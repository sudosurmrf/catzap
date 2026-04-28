"""Parser tests for firmware/edge-bench/run_bench.py.

Mirrors server/tests/test_detector.py style: pure parser unit tests with no
binary execution. Given a fixture stdout string, parse_edgebench_stdout +
report_to_result must produce the expected JSON shape consumable by US-011.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# firmware/edge-bench/ is not a proper python package; expose it via sys.path.
_BENCH_DIR = Path(__file__).resolve().parents[3] / "firmware" / "edge-bench"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import run_bench  # noqa: E402


HAPPY_STDOUT = """
[INFO] some MicroPrintf banner
=== EDGEBENCH START ===
edgebench_version: 1
model_path: /tmp/yolov8n_cat_int8.tflite
model_size_bytes: 3220840
runs: 50
schema_version: 3
schema_status: ok
arena_size_bytes: 8388608
allocate_tensors_status: ok
arena_used_bytes: 1843200
input_bytes: 150528
output_bytes: 5145
profiler_overflowed: false
=== OP_BREAKDOWN START ===
op_name: CONV_2D	count: 3000	total_us: 480000
op_name: DEPTHWISE_CONV_2D	count: 1500	total_us: 90000
op_name: ADD	count: 750	total_us: 4500
op_name: RESHAPE	count: 50	total_us: 200
=== OP_BREAKDOWN END ===
timed_invoke_status: ok
total_invoke_runs: 50
total_invoke_us_p50: 11290
total_invoke_us_p95: 11620
total_invoke_us_min: 11100
total_invoke_us_max: 12000
total_invoke_us_mean: 11340
=== EDGEBENCH END ===
some trailing text
""".strip()


FAIL_ALLOC_STDOUT = """
=== EDGEBENCH START ===
edgebench_version: 1
model_path: /tmp/incompatible.tflite
model_size_bytes: 1000
runs: 50
schema_version: 3
schema_status: ok
arena_size_bytes: 8388608
allocate_tensors_status: failed
=== EDGEBENCH END ===
""".strip()


def test_parse_happy_path_keys_and_op_count() -> None:
    parsed = run_bench.parse_edgebench_stdout(HAPPY_STDOUT)
    assert parsed.fields["model_path"] == "/tmp/yolov8n_cat_int8.tflite"
    assert parsed.fields["model_size_bytes"] == 3_220_840
    assert parsed.fields["runs"] == 50
    assert parsed.fields["arena_used_bytes"] == 1_843_200
    assert parsed.fields["allocate_tensors_status"] == "ok"
    assert parsed.fields["total_invoke_us_p50"] == pytest.approx(11290.0)
    assert parsed.fields["profiler_overflowed"] == "false"

    op_names = [op["op_name"] for op in parsed.op_breakdown]
    assert op_names == ["CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "RESHAPE"]


def test_op_percent_sums_to_100() -> None:
    parsed = run_bench.parse_edgebench_stdout(HAPPY_STDOUT)
    total_pct = sum(op["percent"] for op in parsed.op_breakdown)
    assert total_pct == pytest.approx(100.0, abs=0.01)


def test_op_percent_dominant_op_is_largest() -> None:
    parsed = run_bench.parse_edgebench_stdout(HAPPY_STDOUT)
    dominant = max(parsed.op_breakdown, key=lambda o: o["percent"])
    assert dominant["op_name"] == "CONV_2D"
    assert dominant["percent"] > 80.0


def test_top_k_ops_by_total_us_descending() -> None:
    parsed = run_bench.parse_edgebench_stdout(HAPPY_STDOUT)
    top = run_bench.top_k_ops(parsed.op_breakdown, k=3)
    assert [o["op_name"] for o in top] == ["CONV_2D", "DEPTHWISE_CONV_2D", "ADD"]


def test_report_to_result_has_full_schema() -> None:
    parsed = run_bench.parse_edgebench_stdout(HAPPY_STDOUT)
    result = run_bench.report_to_result(
        parsed,
        story_id="US-011-yolov8n",
        binary_path="firmware/edge-bench/build/edgebench",
        tflm_commit="51bee03bed4776f1de88dd87226ff8c260f88e3c",
    )
    expected = {
        "story_id",
        "model_path",
        "model_size_bytes",
        "runs",
        "arena_size_bytes",
        "arena_used_bytes",
        "input_bytes",
        "output_bytes",
        "schema_status",
        "allocate_tensors_status",
        "timed_invoke_status",
        "profiler_overflowed",
        "tflm_compatible",
        "op_breakdown",
        "raw_x86_us_p50",
        "raw_x86_us_p95",
        "raw_x86_us_min",
        "raw_x86_us_max",
        "raw_x86_us_mean",
        "x86_to_s3_multiplier",
        "predicted_s3_us_p50",
        "predicted_s3_us_p95",
        "predicted_s3_fps",
        "binary",
        "tflm_commit",
    }
    assert set(result.keys()) == expected
    assert result["tflm_compatible"] is True
    assert result["tflm_commit"] == "51bee03bed4776f1de88dd87226ff8c260f88e3c"


def test_report_to_result_uses_default_multiplier() -> None:
    parsed = run_bench.parse_edgebench_stdout(HAPPY_STDOUT)
    result = run_bench.report_to_result(
        parsed, story_id="x", binary_path="b", tflm_commit="c"
    )
    assert result["x86_to_s3_multiplier"] == run_bench.DEFAULT_X86_TO_S3_MULTIPLIER
    assert result["predicted_s3_us_p50"] == pytest.approx(
        11290.0 * run_bench.DEFAULT_X86_TO_S3_MULTIPLIER
    )


def test_report_predicted_fps_inverse_of_predicted_us() -> None:
    parsed = run_bench.parse_edgebench_stdout(HAPPY_STDOUT)
    result = run_bench.report_to_result(
        parsed, story_id="x", binary_path="b", tflm_commit="c", multiplier=10.0
    )
    assert result["predicted_s3_us_p50"] == pytest.approx(112900.0)
    assert result["predicted_s3_fps"] == pytest.approx(1_000_000.0 / 112900.0)


def test_failed_alloc_marks_tflm_incompatible() -> None:
    parsed = run_bench.parse_edgebench_stdout(FAIL_ALLOC_STDOUT)
    result = run_bench.report_to_result(
        parsed, story_id="us-011-broken", binary_path="b", tflm_commit="c"
    )
    assert result["allocate_tensors_status"] == "failed"
    assert result["tflm_compatible"] is False


def test_missing_markers_raises() -> None:
    with pytest.raises(ValueError):
        run_bench.parse_edgebench_stdout("nothing useful here")


def test_run_uses_runner_di_seam_and_passes_args(tmp_path: Path) -> None:
    """run() must pass [binary, model, runs, arena_kb] in order; runner is DI."""
    captured: dict = {}

    class FakeProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = HAPPY_STDOUT
            self.stderr = ""

    def fake_runner(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    binary = tmp_path / "edgebench"
    binary.write_bytes(b"")
    model = tmp_path / "m.tflite"
    model.write_bytes(b"")

    out = run_bench.run(
        binary=binary, model=model, runs=42, arena_kb=4096, runner=fake_runner
    )
    assert captured["cmd"] == [str(binary), str(model), "42", "4096"]
    assert "capture_output" in captured["kwargs"]
    assert out == HAPPY_STDOUT


def test_main_writes_json_to_results_dir(tmp_path: Path, monkeypatch) -> None:
    """End-to-end main() with a stubbed binary writes the JSON report."""
    binary = tmp_path / "edgebench"
    binary.write_bytes(b"\x7fELF stub")
    model = tmp_path / "m.tflite"
    model.write_bytes(b"\x00")
    results = tmp_path / "results"

    class FakeProc:
        returncode = 0
        stdout = HAPPY_STDOUT
        stderr = ""

    monkeypatch.setattr(run_bench.subprocess, "run", lambda *a, **k: FakeProc())

    rc = run_bench.main(
        [
            "--model", str(model),
            "--story-id", "US-011-yolov8n",
            "--runs", "50",
            "--binary", str(binary),
            "--results-dir", str(results),
        ]
    )
    assert rc == 0
    out = results / "US-011-yolov8n-tflm.json"
    assert out.exists()
    blob = json.loads(out.read_text())
    assert blob["story_id"] == "US-011-yolov8n"
    assert blob["model_path"] == "/tmp/yolov8n_cat_int8.tflite"
    assert blob["op_breakdown"][0]["op_name"] == "CONV_2D"
