import ast
import io
import json
import math
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
COLAB_NOTEBOOK_PATH = ROOT / "QAI_S24_ASR_Benchmark_Colab.ipynb"


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


class AudioCompatibilityTests(unittest.TestCase):
    def test_datasets_audio_constructor_avoids_removed_mono_argument(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")

        self.assertNotRegex(source, r"Audio\(sampling_rate=SR,\s*mono=")

    def test_torchcodec_stereo_audio_is_downmixed_to_mono(self):
        funcs = load_functions(
            "_mono_audio_array",
            "_resample_audio",
            "audio_array",
            extra_globals={"SR": 16000},
        )
        stereo = np.asarray([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32)
        audio = SimpleNamespace(
            get_all_samples=lambda: SimpleNamespace(data=stereo, sample_rate=16000)
        )

        waveform, sampling_rate = funcs["audio_array"](audio)

        np.testing.assert_allclose(waveform, np.asarray([2.0, 4.0], dtype=np.float32))
        self.assertEqual(sampling_rate, 16000)

    def test_datasets_audio_uses_decode_false_to_avoid_torchcodec(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")

        casts = re.findall(r'cast_column\("audio",Audio\(decode=False\)\)', source)
        self.assertEqual(len(casts), 3)
        self.assertNotIn('Audio(sampling_rate=SR)', source)

    def test_audio_bytes_are_decoded_resampled_and_downmixed_without_torchcodec(self):
        stereo = np.column_stack(
            [np.linspace(-0.5, 0.5, 80), np.linspace(0.5, -0.5, 80)]
        ).astype(np.float32)

        def fake_read(source, *, dtype, always_2d):
            self.assertIsInstance(source, io.BytesIO)
            self.assertEqual(dtype, "float32")
            self.assertTrue(always_2d)
            return stereo, 8000

        funcs = load_functions(
            "_mono_audio_array",
            "_resample_audio",
            "audio_array",
            extra_globals={
                "io": io,
                "math": math,
                "sf": SimpleNamespace(read=fake_read),
                "resample_poly": lambda x, up, down: np.repeat(x, up // down),
                "SR": 16000,
            },
        )

        waveform, sampling_rate = funcs["audio_array"](
            {"bytes": b"fake-wav", "path": None}
        )

        self.assertEqual(sampling_rate, 16000)
        self.assertEqual(waveform.ndim, 1)
        self.assertEqual(len(waveform), 160)


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

    def test_profile_component_cache_requires_latency_and_memory(self):
        funcs = load_functions("_profile_component_complete")
        complete = funcs["_profile_component_complete"]

        self.assertFalse(complete({"encoder_latency_us": 123}, "encoder"))
        self.assertTrue(
            complete(
                {
                    "encoder_latency_us": 123,
                    "encoder_inference_peak_memory_bytes": 456,
                },
                "encoder",
            )
        )

    def test_profile_job_retries_a_temporary_remote_failure(self):
        failed = SimpleNamespace(
            job_id="jp_profile_failed",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_profile_failed",
            device=SimpleNamespace(name="Samsung Galaxy S24"),
            wait=lambda: FakeStatus(success=False, code="FAILED"),
        )
        succeeded = SimpleNamespace(
            job_id="jp_profile_success",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_profile_success",
            device=SimpleNamespace(name="Samsung Galaxy S24"),
            wait=lambda: FakeStatus(success=True, code="SUCCESS"),
            download_profile=lambda: {
                "estimated_inference_time": 1234,
                "inference_memory_peak_range": [1024, 2048],
            },
        )
        jobs = iter([failed, succeeded])
        profile_client = SimpleNamespace(
            submit_profile_job=lambda *args, **kwargs: next(jobs),
            get_job_summaries=lambda limit: [],
        )
        funcs = load_functions(
            "_status_line",
            "_find_profile_metric",
            "_range_upper",
            "profile_metrics",
            "_is_memory_allocation_error",
            "_profile_runtime_option_candidates",
            "_run_profile_job_with_retries",
            extra_globals={"client": profile_client},
        )

        job, metrics, _ = funcs["_run_profile_job_with_retries"](
            profile_client,
            SimpleNamespace(model_id="mm_model"),
            SimpleNamespace(name="Samsung Galaxy S24"),
            "profile_retry_test",
            "--compute_unit npu",
            "Samsung Galaxy S24",
            2,
        )

        self.assertIs(job, succeeded)
        self.assertEqual(metrics["latency_us"], 1234)
        self.assertEqual(metrics["inference_peak_memory_bytes"], 2048)

    def test_profile_memory_failure_retries_with_maximum_vtcm(self):
        mem_failed_status = SimpleNamespace(
            success=False,
            failure=True,
            finished=True,
            code="FAILED",
            message="QNN_COMMON_ERROR_MEM_ALLOC: Memory allocation related error.",
        )
        failed = SimpleNamespace(
            job_id="jp_profile_mem_failed",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_profile_mem_failed",
            device=SimpleNamespace(name="Samsung Galaxy S24"),
            wait=lambda: mem_failed_status,
        )
        succeeded = SimpleNamespace(
            job_id="jp_profile_vtcm_success",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_profile_vtcm_success",
            device=SimpleNamespace(name="Samsung Galaxy S24"),
            wait=lambda: FakeStatus(success=True, code="SUCCESS"),
            download_profile=lambda: {
                "estimated_inference_time": 1234,
                "inference_memory_peak_range": [1024, 2048],
            },
        )
        jobs = iter([failed, succeeded])
        submitted_options = []

        def submit_profile_job(*args, **kwargs):
            submitted_options.append(kwargs["options"])
            return next(jobs)

        profile_client = SimpleNamespace(
            submit_profile_job=submit_profile_job,
            get_job_summaries=lambda limit: [],
        )
        funcs = load_functions(
            "_status_line",
            "_find_profile_metric",
            "_range_upper",
            "profile_metrics",
            "_is_memory_allocation_error",
            "_profile_runtime_option_candidates",
            "_run_profile_job_with_retries",
            extra_globals={"client": profile_client},
        )

        job, _, _ = funcs["_run_profile_job_with_retries"](
            profile_client,
            SimpleNamespace(model_id="mm_model"),
            SimpleNamespace(name="Samsung Galaxy S24"),
            "profile_vtcm_retry_test",
            "--compute_unit npu --qairt_version 2.49",
            "Samsung Galaxy S24",
            3,
        )

        self.assertIs(job, succeeded)
        self.assertNotIn("default_graph_htp_vtcm_size", submitted_options[0])
        self.assertIn("default_graph_htp_vtcm_size=0", submitted_options[1])

    def test_profile_vtcm_retry_preserves_context_graph_selection(self):
        funcs = load_functions("_profile_runtime_option_candidates")

        candidates = funcs["_profile_runtime_option_candidates"](
            "--compute_unit npu --qairt_version 2.49 "
            "--qnn_options context_enable_graphs=whisper_small_encoder"
        )

        self.assertEqual(len(candidates), 2)
        self.assertIn("context_enable_graphs=whisper_small_encoder", candidates[1])
        self.assertIn(";default_graph_htp_vtcm_size=0", candidates[1])

    def test_single_graph_context_profile_fallback_compiles_and_links_one_component(self):
        compiled_model = SimpleNamespace(model_id="mm_compiled", wait=lambda: True)
        context_model = SimpleNamespace(model_id="mm_context", wait=lambda: True)
        compile_job = SimpleNamespace(
            job_id="jp_compile",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_compile",
            device=SimpleNamespace(name="Samsung Galaxy S24"),
            wait=lambda: FakeStatus(success=True, code="SUCCESS"),
            get_target_model=lambda: compiled_model,
        )
        link_job = SimpleNamespace(
            job_id="jp_link",
            url="https://workbench.aihub.qualcomm.com/jobs/jp_link",
            device=SimpleNamespace(name="Samsung Galaxy S24"),
            wait=lambda: FakeStatus(success=True, code="SUCCESS"),
            get_target_model=lambda: context_model,
        )
        submissions = []

        def submit_compile_and_link_jobs(**kwargs):
            submissions.append(kwargs)
            return [compile_job], link_job

        funcs = load_functions(
            "_status_line",
            "_link_retry_options",
            "_retry_link_jobs",
            "_compile_single_graph_profile_context",
        )
        model, evidence = funcs["_compile_single_graph_profile_context"](
            SimpleNamespace(submit_compile_and_link_jobs=submit_compile_and_link_jobs),
            SimpleNamespace(model_id="mm_source"),
            {"input_features": ((1, 80, 3000), "float32")},
            ["cross_cache"],
            "whisper_small_encoder",
            SimpleNamespace(name="Samsung Galaxy S24"),
            "whisper_small_encoder_profile_fallback",
            "2.49",
        )

        self.assertIs(model, context_model)
        self.assertEqual(submissions[0]["models"][0].model_id, "mm_source")
        self.assertEqual(submissions[0]["graph_names"], ["whisper_small_encoder"])
        self.assertIn("--quantize_full_type float16", submissions[0]["compile_options"][0])
        self.assertEqual(evidence["profile_model_id"], "mm_context")

    def test_profile_validation_rejects_missing_latency_or_peak_ram(self):
        funcs = load_functions("_validate_required_profile_rows")
        rows = [
            {"model": "Whisper Tiny", "encoder_ms": 1.0, "decoder_ms_per_token": 2.0, "peak_ram_mb": 3.0},
            {"model": "Whisper Small", "encoder_ms": 1.0, "decoder_ms_per_token": np.nan, "peak_ram_mb": 3.0},
            {"model": "PhoWhisper Base", "encoder_ms": 1.0, "decoder_ms_per_token": 2.0, "peak_ram_mb": 3.0},
        ]

        with self.assertRaisesRegex(RuntimeError, "Whisper Small.*decoder_ms_per_token"):
            funcs["_validate_required_profile_rows"](
                rows,
                ["Whisper Tiny", "Whisper Small", "PhoWhisper Base"],
            )

    def test_final_table_validation_rejects_any_blank_submission_cell(self):
        funcs = load_functions("_validate_required_final_rows")
        required_fields = ["VN WER (%) ↓", "Noise ΔWER @0dB (pp) ↓", "Peak RAM on S24 (MB) ↓"]
        rows = [
            {
                "Model": "Whisper Tiny",
                "VN WER (%) ↓": 10.0,
                "Noise ΔWER @0dB (pp) ↓": np.nan,
                "Peak RAM on S24 (MB) ↓": 100.0,
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "Whisper Tiny.*Noise ΔWER"):
            funcs["_validate_required_final_rows"](
                rows,
                ["Whisper Tiny"],
                required_fields,
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
        self.assertIn('$env:QAI_ENABLE_PROFILING = "1"', setup)

    def test_default_run_uses_50_samples_and_requires_s24_profiling(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertRegex(source, r'QAI_RUN_MODE",\s*"benchmark"')
        self.assertRegex(source, r'BENCHMARK_N\s*=\s*50')
        self.assertIn('VIMD_BENCHMARK_REGION_TARGETS = {"North": 17, "Central": 17, "South": 16}', source)
        self.assertRegex(source, r'QAI_ENABLE_PROFILING",\s*"1"')
        self.assertRegex(source, r'PROFILE_REQUIRED\s*=\s*True')

    def test_microbatch_is_tunable_but_keeps_the_safe_default(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertRegex(source, r'QAI_HUB_MICROBATCH",\s*"2"')

    def test_colab_notebook_is_upload_and_run_ready(self):
        self.assertTrue(COLAB_NOTEBOOK_PATH.exists())
        notebook = json.loads(COLAB_NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"colab_cell_{index}", "exec")
        all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )

        for required in (
            "drive.mount",
            "tanphong-sudo/qai_s24_benchmark.git",
            "QAI_RUN_MODE",
            "QAI_HUB_MICROBATCH",
            "QAI_ENABLE_PROFILING",
            "QAI_HF_HOME",
            "QAI_DATA_ROOT",
            "QAI_HUB_API_TOKEN",
            "run_benchmark.py",
            "QAI_S24_BENCHMARK_SUBMISSION.zip",
            "hf_cache",
            "checkpoints",
        ):
            self.assertIn(required, all_source)

        self.assertIn('os.environ["QAI_HUB_MICROBATCH"] = "4"', all_source)
        self.assertIn("'samples_per_benchmark': 50", all_source)
        self.assertNotRegex(all_source, r'QAI_HUB_API_TOKEN\s*=\s*["\'][A-Za-z0-9_-]{20,}')

    def test_colab_can_keep_large_download_caches_off_google_drive(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("QAI_HF_HOME"', source)
        self.assertIn('os.environ.get("QAI_DATA_ROOT"', source)


if __name__ == "__main__":
    unittest.main()
