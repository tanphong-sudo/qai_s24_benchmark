import ast
import json
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "run_benchmark.py"


def load_functions(*names: str, extra_globals: dict[str, Any] | None = None):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    wanted = set(names)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    namespace = {
        "Any": Any,
        "Path": Path,
        "json": json,
        "np": np,
        "time": time,
    }
    if extra_globals:
        namespace.update(extra_globals)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


class FakeStatus:
    def __init__(self, *, success: bool, code: str):
        self.success = success
        self.failure = not success
        self.finished = True
        self.code = code
        self.message = None if success else "remote job failed"


class ProducerStatusTests(unittest.TestCase):
    def test_failed_model_producer_is_rejected_before_profile_submission(self):
        funcs = load_functions("_status_line", "_model_producer_status", "_require_model_ready")
        failed = FakeStatus(success=False, code="FAILED")
        producer = SimpleNamespace(
            url="https://workbench.aihub.qualcomm.com/jobs/jp_failed",
            get_status=lambda: failed,
            wait=lambda: failed,
        )
        model = SimpleNamespace(
            model_id="mm_failed",
            wait=lambda: False,
            get_producer=lambda: producer,
        )

        with self.assertRaisesRegex(RuntimeError, "failed producer"):
            funcs["_require_model_ready"](model, "Whisper Tiny")


class InferenceStatusTests(unittest.TestCase):
    @staticmethod
    def _globals(job, evidence):
        tensor = SimpleNamespace(name="input", shape=(1,), dtype="float32")
        return {
            "client": SimpleNamespace(submit_inference_job=lambda **kwargs: job),
            "TARGET_DEVICE": SimpleNamespace(name="Samsung Galaxy S24"),
            "TARGET_DEVICE_NAME": "Samsung Galaxy S24",
            "QAIRT_VERSION": "2.39",
            "graph_input_specs": lambda model, graph: [tensor],
            "graph_output_names": lambda model, graph: ["output"],
            "cast_for_tensor": lambda value, spec: np.asarray(value, dtype=np.float32),
            "append_job_evidence": evidence.append,
        }

    def test_failed_inference_is_reported_with_job_url_and_not_recorded_as_success(self):
        evidence = []
        failed = FakeStatus(success=False, code="FAILED")
        job = SimpleNamespace(
            device=SimpleNamespace(name="Samsung Galaxy S24"),
            job_id="jp_failed",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_failed",
            wait=lambda: failed,
            download_output_data=lambda: None,
        )
        funcs = load_functions(
            "_status_line",
            "_wait_job_success",
            "infer_graph",
            extra_globals=self._globals(job, evidence),
        )

        with self.assertRaisesRegex(RuntimeError, "jp_failed"):
            funcs["infer_graph"](
                SimpleNamespace(model_id="mm_model"),
                "encoder",
                [{"input": np.array([1.0], dtype=np.float32)}],
                "failed_inference",
            )

        self.assertEqual(evidence, [])

    def test_successful_inference_records_evidence_after_output_download(self):
        evidence = []
        succeeded = FakeStatus(success=True, code="SUCCESS")
        job = SimpleNamespace(
            device=SimpleNamespace(name="Samsung Galaxy S24"),
            job_id="jp_success",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_success",
            wait=lambda: succeeded,
            download_output_data=lambda: {
                "output": [np.array([1.0]), np.array([2.0])]
            },
        )
        funcs = load_functions(
            "_status_line",
            "_wait_job_success",
            "infer_graph",
            extra_globals=self._globals(job, evidence),
        )

        outputs, _, job_id = funcs["infer_graph"](
            SimpleNamespace(model_id="mm_model"),
            "encoder",
            [
                {"input": np.array([1.0], dtype=np.float32)},
                {"input": np.array([2.0], dtype=np.float32)},
            ],
            "successful_inference",
        )

        self.assertEqual(job_id, "jp_success")
        self.assertEqual([float(row["output"][0]) for row in outputs], [1.0, 2.0])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["status"], "success")


class CacheAndDependencyTests(unittest.TestCase):
    def test_failed_profile_cache_is_retried(self):
        funcs = load_functions("_profile_cache_needs_refresh")
        self.assertIn("_profile_cache_needs_refresh", funcs)
        should_refresh = funcs["_profile_cache_needs_refresh"]

        self.assertTrue(should_refresh(None, "key", ["encoder_latency_us"]))
        self.assertTrue(
            should_refresh(
                {
                    "profile_key": "key",
                    "encoder_latency_us": None,
                    "profile_error": "temporary failure",
                },
                "key",
                ["encoder_latency_us"],
            )
        )
        self.assertFalse(
            should_refresh(
                {"profile_key": "key", "encoder_latency_us": 123, "profile_error": None},
                "key",
                ["encoder_latency_us"],
            )
        )

    def test_corrupt_json_cache_is_quarantined(self):
        funcs = load_functions("_safe_json_load")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cache.json"
            path.write_text("{broken", encoding="utf-8")

            self.assertEqual(funcs["_safe_json_load"](path, {}), {})
            self.assertFalse(path.exists())
            self.assertEqual(len(list(path.parent.glob("cache.json.corrupt_*"))), 1)

    def test_qai_hub_sdk_version_is_pinned(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertRegex(requirements, re.compile(r"^qai-hub==0\.55\.0$", re.MULTILINE))

    def test_windows_launcher_runs_dependency_and_regression_preflight(self):
        setup = (ROOT / "setup_windows.ps1").read_text(encoding="utf-8")
        self.assertIn("-m pip check", setup)
        self.assertIn("-m py_compile run_benchmark.py", setup)
        self.assertIn("-m unittest discover -s tests -v", setup)


if __name__ == "__main__":
    unittest.main()
