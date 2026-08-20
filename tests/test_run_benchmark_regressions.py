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
            "HUB_JOB_RETRIES": 1,
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
            "_qnn_runtime_options",
            "_run_inference_job_with_retries",
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
            "_qnn_runtime_options",
            "_run_inference_job_with_retries",
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
    def test_direct_component_compile_jobs_target_qnn_dlc_without_linking(self):
        submissions = []

        def submit_compile_job(**kwargs):
            submissions.append(kwargs)
            return SimpleNamespace(job_id=f"jp_{len(submissions)}")

        funcs = load_functions("_submit_component_compile_jobs")
        jobs = funcs["_submit_component_compile_jobs"](
            SimpleNamespace(submit_compile_job=submit_compile_job),
            ["encoder_source", "decoder_source"],
            [{"input_features": ((1, 80, 3000), "float32")}, {"input_ids": ((1, 1), "int32")}],
            [["cross_cache"], ["logits"]],
            SimpleNamespace(name="Samsung Galaxy S24"),
            "whisper_small_s24",
            "2.49",
        )

        self.assertEqual(len(jobs), 2)
        self.assertEqual(len(submissions), 2)
        self.assertTrue(all("--target_runtime qnn_dlc" in row["options"] for row in submissions))
        self.assertEqual([row["model"] for row in submissions], ["encoder_source", "decoder_source"])

    def test_inference_retries_a_failed_remote_job(self):
        failed = SimpleNamespace(
            job_id="jp_failed",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_failed",
            device=SimpleNamespace(name="Samsung Galaxy S24"),
            wait=lambda: FakeStatus(success=False, code="FAILED"),
            download_output_data=lambda: None,
        )
        succeeded = SimpleNamespace(
            job_id="jp_success",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_success",
            device=SimpleNamespace(name="Samsung Galaxy S24"),
            wait=lambda: FakeStatus(success=True, code="SUCCESS"),
            download_output_data=lambda: {"output": [np.array([1.0])]},
        )
        jobs = iter([failed, succeeded])
        client = SimpleNamespace(submit_inference_job=lambda **kwargs: next(jobs))
        funcs = load_functions("_status_line", "_run_inference_job_with_retries")

        job, output = funcs["_run_inference_job_with_retries"](
            client,
            SimpleNamespace(model_id="mm_model"),
            SimpleNamespace(name="Samsung Galaxy S24"),
            {"input": [np.array([1.0])]},
            "retry_test",
            "--compute_unit npu",
            "Samsung Galaxy S24",
            2,
        )

        self.assertIs(job, succeeded)
        self.assertIn("output", output)

    def test_single_graph_qnn_dlc_uses_remote_none_keyed_contract(self):
        input_tensor = SimpleNamespace(name="input_features", shape=(1, 80, 3000), dtype="float16")
        output_tensor = SimpleNamespace(name="k_cache_cross_0", shape=(1,), dtype="float16")
        funcs = load_functions(
            "graph_input_specs",
            "graph_output_names",
            extra_globals={"GRAPH_CONTRACTS": {}},
        )
        model = SimpleNamespace(
            input_spec={None: [input_tensor]},
            output_spec={None: [output_tensor]},
        )

        self.assertEqual(funcs["graph_input_specs"](model, "whisper_small_encoder"), [input_tensor])
        self.assertEqual(funcs["graph_output_names"](model, "whisper_small_encoder"), ["k_cache_cross_0"])

    def test_separate_qnn_dlc_runtime_does_not_select_a_context_graph(self):
        funcs = load_functions("_qnn_runtime_options")

        self.assertEqual(
            funcs["_qnn_runtime_options"]("2.49", "encoder", "separate_qnn_dlc"),
            "--compute_unit npu --qairt_version 2.49",
        )
        self.assertIn(
            "context_enable_graphs=encoder",
            funcs["_qnn_runtime_options"]("2.49", "encoder", "linked_context"),
        )

    def test_link_retry_drops_htp_optimization_before_dlc_fallback(self):
        funcs = load_functions("_link_retry_options")
        self.assertIn("_link_retry_options", funcs)

        self.assertEqual(
            funcs["_link_retry_options"]("2.49"),
            [
                "--qairt_version 2.49 --qnn_options default_graph_htp_optimizations=O=2",
                "--qairt_version 2.49 --qnn_options default_graph_htp_optimizations=O=1",
            ],
        )

    def test_link_retry_returns_first_successful_context_binary(self):
        target_model = SimpleNamespace(model_id="mm_context", wait=lambda: True)
        failed_job = SimpleNamespace(
            job_id="jp_o2",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_o2",
            wait=lambda: FakeStatus(success=False, code="FAILED"),
            get_target_model=lambda: None,
        )
        successful_job = SimpleNamespace(
            job_id="jp_o1",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_o1",
            wait=lambda: FakeStatus(success=True, code="SUCCESS"),
            get_target_model=lambda: target_model,
        )
        submitted_options = []
        jobs = iter([failed_job, successful_job])

        def submit_link_job(models, *, device, name, options):
            submitted_options.append(options)
            return next(jobs)

        funcs = load_functions(
            "_status_line",
            "_link_retry_options",
            "_retry_link_jobs",
        )
        job, model, attempts = funcs["_retry_link_jobs"](
            SimpleNamespace(submit_link_job=submit_link_job),
            [SimpleNamespace(model_id="mm_encoder"), SimpleNamespace(model_id="mm_decoder")],
            SimpleNamespace(name="Samsung Galaxy S24"),
            "whisper_small_s24_whisper",
            "2.49",
        )

        self.assertIs(job, successful_job)
        self.assertIs(model, target_model)
        self.assertEqual([attempt["success"] for attempt in attempts], [False, True])
        self.assertIn("O=2", submitted_options[0])
        self.assertIn("O=1", submitted_options[1])

    def test_link_retry_continues_after_submission_error(self):
        target_model = SimpleNamespace(model_id="mm_context", wait=lambda: True)
        successful_job = SimpleNamespace(
            job_id="jp_o1",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_o1",
            wait=lambda: FakeStatus(success=True, code="SUCCESS"),
            get_target_model=lambda: target_model,
        )
        calls = 0

        def submit_link_job(models, *, device, name, options):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary submit failure")
            return successful_job

        funcs = load_functions(
            "_status_line",
            "_link_retry_options",
            "_retry_link_jobs",
        )
        job, model, attempts = funcs["_retry_link_jobs"](
            SimpleNamespace(submit_link_job=submit_link_job),
            [SimpleNamespace(model_id="mm_encoder"), SimpleNamespace(model_id="mm_decoder")],
            SimpleNamespace(name="Samsung Galaxy S24"),
            "whisper_small_s24_whisper",
            "2.49",
        )

        self.assertIs(job, successful_job)
        self.assertIs(model, target_model)
        self.assertEqual([attempt["success"] for attempt in attempts], [False, True])
        self.assertIn("temporary submit failure", attempts[0]["error"])

    def test_cache_supports_separate_qnn_dlc_component_models(self):
        funcs = load_functions("_cached_artifact_model_ids")
        self.assertIn("_cached_artifact_model_ids", funcs)

        self.assertEqual(
            funcs["_cached_artifact_model_ids"](
                {
                    "artifact_mode": "separate_qnn_dlc",
                    "encoder_model_id": "mm_encoder",
                    "decoder_model_id": "mm_decoder",
                }
            ),
            ("separate_qnn_dlc", "mm_encoder", "mm_decoder"),
        )
        self.assertEqual(
            funcs["_cached_artifact_model_ids"]({"linked_model_id": "mm_linked"}),
            ("linked_context", "mm_linked", "mm_linked"),
        )

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
        self.assertIn('$env:QAI_RUN_MODE = "benchmark"', setup)
        self.assertIn('$env:QAI_ARTIFACT_POLICY = "separate_qnn_dlc"', setup)
        self.assertIn('$env:QAI_ENABLE_PROFILING = "0"', setup)

    def test_default_run_keeps_100_samples_and_skips_optional_profiling(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertRegex(source, r'QAI_RUN_MODE",\s*"benchmark"')
        self.assertRegex(source, r'BENCHMARK_N\s*=\s*100')
        self.assertRegex(source, r'QAI_ENABLE_PROFILING",\s*"0"')


if __name__ == "__main__":
    unittest.main()
