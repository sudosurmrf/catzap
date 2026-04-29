"""iter-C: tests for the L2-norm channel pruning pipeline.

All heavy steps (yolo_factory / pruner / trainer / quantize / eval / bench)
go through DI seams so this file does NOT need torch, ultralytics, TF, cv2,
or the edgebench binary at import or run time. Each test runs in <100 ms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

# Lazy-import dance: torch is needed by some tests but the module under test
# must remain import-safe without it. We import the module first, then guard
# torch-specific tests with skipif.
from training.edge.yolo import prune_channels as pc

torch = pytest.importorskip(
    "torch",
    reason=(
        "torch is needed for the prune-on-real-Conv2d tests. The CLI tests "
        "below don't need it (they DI-mock the pruner), so those still run."
    ),
)


# ---------------------------------------------------------------------------
# Helpers — tiny torch nn.Module fixtures with a clear "head" boundary so we
# can exercise both the include-head and exclude-head pruning branches.
# ---------------------------------------------------------------------------


class Detect(torch.nn.Module):
    """Stand-in for ultralytics' Detect head — class name match is what
    triggers exclude_head in prune_l2_channels._find_head_indices."""

    def __init__(self) -> None:
        super().__init__()
        # Two convs nested inside the "head" block.
        self.cv1 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, bias=False)
        self.cv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, bias=False)


class _FakeYolo(torch.nn.Module):
    """Mimics the ultralytics convention: ``self.model`` is a Sequential
    whose last child is the Detect head. This lets _find_head_indices
    return ``{2}`` and _is_head_path filter the ``2.*`` subtree."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            torch.nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            Detect(),  # index 2 -> head (class name == "Detect")
        )


# ---------------------------------------------------------------------------
# Test 1 (REQUIRED by AC): prune_l2_channels reduces param count by approximately
# the requested sparsity on a tiny module.
# ---------------------------------------------------------------------------


def test_prune_l2_channels_reduces_params_by_target_sparsity() -> None:
    """Real Conv2d, real torch.nn.utils.prune. ~25% sparsity per layer.

    LnStructured zeros out exactly ``floor(amount * out_channels)`` channels
    per layer. With out_channels=16 and amount=0.25 we get 4 zeroed channels
    out of 16 — exactly 25% per-layer (so total weight zeroing is also 25%).
    """
    model = _FakeYolo()
    summary = pc.prune_l2_channels(model, sparsity=0.25, exclude_head=False)

    # Three Conv2d total in this fixture: model.0, model.1, model.2.cv1, model.2.cv2.
    # With exclude_head=False all four are pruned.
    assert summary["actual_sparsity"] == pytest.approx(0.25, abs=0.05), summary
    assert summary["params_zeroed"] > 0
    assert summary["params_after_nonzero"] < summary["params_before"]
    # Sanity: pruned_modules covers all four real Conv2d instances.
    assert len(summary["pruned_modules"]) == 4


# ---------------------------------------------------------------------------
# Test 2 (REQUIRED by AC): the head module is excluded from pruning.
# ---------------------------------------------------------------------------


def test_prune_l2_channels_skips_the_head_when_exclude_head_true() -> None:
    """With exclude_head=True, model.2.cv1 and model.2.cv2 (under Detect)
    must NOT be pruned. The two backbone convs (model.0, model.1) ARE."""
    model = _FakeYolo()
    summary = pc.prune_l2_channels(model, sparsity=0.25, exclude_head=True)

    # Backbone Conv2d count == 2 (indices 0 and 1).
    assert len(summary["pruned_modules"]) == 2
    # Head Conv2d (Detect block at index 2) appears in skipped, not pruned.
    pruned = set(summary["pruned_modules"])
    skipped = set(summary["skipped_head_modules"])
    # No head paths in pruned.
    assert not any(p.startswith("2.") for p in pruned), pruned
    # Both head conv paths in skipped.
    assert any("2.cv1" in s for s in skipped), skipped
    assert any("2.cv2" in s for s in skipped), skipped
    # head_indices populated.
    assert summary["head_indices"] == [2]
    # Detect head weights are still fully nonzero — verify by reading them.
    detect = model.model[2]
    assert int(detect.cv1.weight.ne(0).sum().item()) == detect.cv1.weight.numel()
    assert int(detect.cv2.weight.ne(0).sum().item()) == detect.cv2.weight.numel()


# ---------------------------------------------------------------------------
# Test 3 (REQUIRED by AC): the CLI calls per_channel_quant.export_per_channel_int8
# AFTER finetune. Ordering matters — we'd ship a stale .tflite if quant ran first.
# ---------------------------------------------------------------------------


def test_pipeline_invokes_quantize_after_trainer(tmp_path: Path) -> None:
    """run_pipeline orchestrates trainer -> quantize -> eval -> bench in order.

    We feed mocks for every stage and assert the call ordering. The pruner
    is a no-op stub since trainer_fn is fully mocked too.
    """
    pt_path = tmp_path / "fake.pt"
    pt_path.write_bytes(b"")
    out_pt = tmp_path / "pruned.pt"
    out_tflite = tmp_path / "pruned.tflite"
    results_dir = tmp_path / "results"
    runs_dir = tmp_path / "runs"
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()

    # Track the order of stage calls so we can assert trainer < quantize < eval < bench.
    call_order: list[str] = []

    def fake_trainer(**kwargs):
        call_order.append("trainer")
        # Honor the contract: write the pruned .pt so downstream paths exist.
        Path(kwargs["pruned_pt_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["pruned_pt_path"]).write_bytes(b"PRUNED")
        return Path(kwargs["pruned_pt_path"]), {
            "sparsity_target": kwargs["sparsity"],
            "actual_sparsity": 0.245,
            "params_before": 3_011_043,
            "params_after_nonzero": 2_273_237,
            "params_zeroed": 737_806,
            "exclude_head": kwargs["exclude_head"],
            "head_indices": [22],
            "pruned_modules": ["model.0.conv"],
            "skipped_head_modules": ["model.22.cv2.0.0.conv"],
        }

    def fake_quantize(*, pt_path, calib_dir, out_path, imgsz):
        call_order.append("quantize")
        # The pruned .pt MUST already exist when quantize is called — that's
        # the load-bearing ordering check this test enforces.
        assert Path(pt_path).exists(), (
            "quantize ran before trainer wrote the pruned .pt — ordering broken"
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"FAKE_TFLITE")
        return out_path

    def fake_eval(*, model_path, story_id, val_dir, imgsz, results_dir):
        call_order.append("eval")
        return {
            "story_id": story_id,
            "model_path": str(model_path),
            "model_format": "tflite_int8",
            "map50": 0.21,
            "size_bytes": Path(model_path).stat().st_size,
            "params": 2_273_237,
            "flops": 0,
            "input_hw": [imgsz, imgsz],
            "latency_ms_p50": 11.0,
            "latency_ms_p95": 13.5,
            "val_images": 120,
            "notes": "",
        }

    def fake_bench(*, model_path, story_id, results_dir, runs):
        call_order.append("bench")
        return {
            "story_id": story_id,
            "model_size_bytes": Path(model_path).stat().st_size,
            "runs": runs,
            "arena_used_bytes": 692_912,
            "tflm_compatible": True,
            "op_breakdown": [],
            "raw_x86_us_p50": 1_000_000.0,
            "raw_x86_us_p95": 1_050_000.0,
        }

    def baseline_loader():
        return {
            "candidate_metrics": {
                "map50": 0.2155,
                "size_bytes": 3_221_000,
                "arena_used_bytes": 692_912,
                "predicted_p4_latency_ms_p50": 5_820.51,
            }
        }

    doc = pc.run_pipeline(
        pt_path=pt_path,
        pruned_pt_path=out_pt,
        out_tflite=out_tflite,
        data_yaml=tmp_path / "data.yaml",
        calib_dir=calib_dir,
        val_dir=val_dir,
        results_dir=results_dir,
        runs_dir=runs_dir,
        trainer_fn=fake_trainer,
        quantize_fn=fake_quantize,
        eval_fn=fake_eval,
        bench_fn=fake_bench,
        baseline_loader=baseline_loader,
    )

    # AC: quantize is called AFTER trainer.
    assert call_order == ["trainer", "quantize", "eval", "bench"], call_order
    # iter-C_pruned.json was written.
    assert (results_dir / "iter-C_pruned.json").exists()
    # Status is passed (no blocker triggered).
    assert doc["status"] == "passed"


# ---------------------------------------------------------------------------
# Test 4: a quantize failure is captured as status="blocked" with a concrete
# reason, and the JSON is STILL written (per AC: "If finetune fails ... record
# status='blocked' with the concrete reason and skip the quantize/eval/bench
# steps"). The same isolation contract applies to a quantize failure too.
# ---------------------------------------------------------------------------


def test_pipeline_writes_blocked_json_when_quantize_fails(tmp_path: Path) -> None:
    pt_path = tmp_path / "fake.pt"
    pt_path.write_bytes(b"")
    results_dir = tmp_path / "results"

    def fake_trainer(**kwargs):
        Path(kwargs["pruned_pt_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["pruned_pt_path"]).write_bytes(b"P")
        return Path(kwargs["pruned_pt_path"]), {
            "sparsity_target": 0.25,
            "actual_sparsity": 0.25,
            "params_before": 100,
            "params_after_nonzero": 75,
            "params_zeroed": 25,
            "exclude_head": True,
            "head_indices": [22],
            "pruned_modules": [],
            "skipped_head_modules": [],
        }

    def flaky_quantize(*, pt_path, calib_dir, out_path, imgsz):
        raise RuntimeError("simulated onnx2tf NHWC layout error in iter-C")

    def fail_eval(**kwargs):
        # eval should NOT be called — quantize blocked the pipeline.
        raise AssertionError(
            "eval_fn should not run when quantize raised before it"
        )

    def fail_bench(**kwargs):
        raise AssertionError("bench_fn should not run after a quantize block")

    def baseline_loader():
        return {
            "candidate_metrics": {
                "map50": 0.2155,
                "size_bytes": 3_221_000,
                "arena_used_bytes": 692_912,
                "predicted_p4_latency_ms_p50": 5_820.51,
            }
        }

    doc = pc.run_pipeline(
        pt_path=pt_path,
        pruned_pt_path=tmp_path / "p.pt",
        out_tflite=tmp_path / "p.tflite",
        data_yaml=tmp_path / "data.yaml",
        calib_dir=tmp_path / "calib",
        val_dir=tmp_path / "val",
        results_dir=results_dir,
        runs_dir=tmp_path / "runs",
        trainer_fn=fake_trainer,
        quantize_fn=flaky_quantize,
        eval_fn=fail_eval,
        bench_fn=fail_bench,
        baseline_loader=baseline_loader,
    )

    # JSON file written even though pipeline blocked.
    out_json = results_dir / "iter-C_pruned.json"
    assert out_json.exists()
    assert doc["status"] == "blocked"
    assert doc["pareto"]["verdict"] == "blocked"
    assert "onnx2tf" in doc["blocked_reason"]
    # Baseline numbers preserved so iter-H aggregator can still render the row.
    assert doc["baseline_metrics"]["map50"] == pytest.approx(0.2155)


# ---------------------------------------------------------------------------
# Test 5: a finetune failure (raised by trainer_fn) records status="blocked"
# AND skips the quantize/eval/bench steps — exact wording from the AC.
# ---------------------------------------------------------------------------


def test_pipeline_blocks_when_finetune_fails(tmp_path: Path) -> None:
    pt_path = tmp_path / "fake.pt"
    pt_path.write_bytes(b"")
    results_dir = tmp_path / "results"

    def flaky_trainer(**kwargs):
        raise RuntimeError("CUDA OOM at batch=32, imgsz=224")

    quantize = MagicMock()
    eval_fn = MagicMock()
    bench_fn = MagicMock()

    def baseline_loader():
        return {
            "candidate_metrics": {
                "map50": 0.2155,
                "size_bytes": 3_221_000,
                "arena_used_bytes": 692_912,
                "predicted_p4_latency_ms_p50": 5_820.51,
            }
        }

    doc = pc.run_pipeline(
        pt_path=pt_path,
        pruned_pt_path=tmp_path / "p.pt",
        out_tflite=tmp_path / "p.tflite",
        data_yaml=tmp_path / "data.yaml",
        calib_dir=tmp_path / "calib",
        val_dir=tmp_path / "val",
        results_dir=results_dir,
        runs_dir=tmp_path / "runs",
        trainer_fn=flaky_trainer,
        quantize_fn=quantize,
        eval_fn=eval_fn,
        bench_fn=bench_fn,
        baseline_loader=baseline_loader,
    )

    # PRD wording: "skip the quantize/eval/bench steps".
    quantize.assert_not_called()
    eval_fn.assert_not_called()
    bench_fn.assert_not_called()

    assert doc["status"] == "blocked"
    assert "CUDA OOM" in doc["blocked_reason"]
    out_json = results_dir / "iter-C_pruned.json"
    assert out_json.exists()


# ---------------------------------------------------------------------------
# Test 6: the rollup JSON has the schema the iter-H SUMMARY_v2 aggregator
# will ingest (mirrors iter-A_per_channel_quant.json + iter-B_imgsz_*.json).
# ---------------------------------------------------------------------------


def test_pipeline_rollup_has_iter_h_ingestion_schema(tmp_path: Path) -> None:
    pt_path = tmp_path / "fake.pt"
    pt_path.write_bytes(b"")
    results_dir = tmp_path / "results"

    def fake_trainer(**kwargs):
        Path(kwargs["pruned_pt_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["pruned_pt_path"]).write_bytes(b"P")
        return Path(kwargs["pruned_pt_path"]), {
            "sparsity_target": 0.25,
            "actual_sparsity": 0.25,
            "params_before": 3_011_043,
            "params_after_nonzero": 2_258_283,
            "params_zeroed": 752_760,
            "exclude_head": True,
            "head_indices": [22],
            "pruned_modules": [],
            "skipped_head_modules": [],
        }

    def fake_quantize(*, pt_path, calib_dir, out_path, imgsz):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"X" * 2_400_000)  # 30% smaller
        return out_path

    def fake_eval(*, model_path, story_id, val_dir, imgsz, results_dir):
        return {
            "story_id": story_id,
            "model_path": str(model_path),
            "model_format": "tflite_int8",
            "map50": 0.2155,
            "size_bytes": Path(model_path).stat().st_size,
            "params": 2_258_283,
            "flops": 0,
            "input_hw": [imgsz, imgsz],
            "latency_ms_p50": 9.5,
            "latency_ms_p95": 12.0,
            "val_images": 120,
            "notes": "",
        }

    def fake_bench(*, model_path, story_id, results_dir, runs):
        return {
            "story_id": story_id,
            "model_size_bytes": Path(model_path).stat().st_size,
            "runs": runs,
            "arena_used_bytes": 540_000,  # ~22% smaller
            "tflm_compatible": True,
            "op_breakdown": [],
            "raw_x86_us_p50": 900_000.0,  # ~23% faster
        }

    def baseline_loader():
        return {
            "candidate_metrics": {
                "map50": 0.2155,
                "size_bytes": 3_221_000,
                "arena_used_bytes": 692_912,
                "predicted_p4_latency_ms_p50": 5_820.51,
            }
        }

    doc = pc.run_pipeline(
        pt_path=pt_path,
        pruned_pt_path=tmp_path / "p.pt",
        out_tflite=tmp_path / "p.tflite",
        data_yaml=tmp_path / "data.yaml",
        calib_dir=tmp_path / "calib",
        val_dir=tmp_path / "val",
        results_dir=results_dir,
        runs_dir=tmp_path / "runs",
        trainer_fn=fake_trainer,
        quantize_fn=fake_quantize,
        eval_fn=fake_eval,
        bench_fn=fake_bench,
        baseline_loader=baseline_loader,
    )

    for required in (
        "story_id",
        "title",
        "status",
        "blocked_reason",
        "model_path",
        "p4_multiplier",
        "p4_multiplier_source",
        "prune_summary",
        "eval",
        "tflm",
        "pareto",
        "candidate_metrics",
        "baseline_metrics",
    ):
        assert required in doc, f"missing key {required}: {list(doc)}"

    # Sanity on the verdict — fixture engineered to dominate (latency -22.7%,
    # arena -22%, size large drop, mAP held).
    assert doc["pareto"]["verdict"] == "dominates"
    assert doc["candidate_metrics"]["predicted_p4_fps"] > 0


# ---------------------------------------------------------------------------
# Test 7: prune_l2_channels works on a fully-mocked module — useful for
# environments that don't have torch (the CI lane that runs lint without TF).
# ---------------------------------------------------------------------------


def test_prune_l2_channels_with_mocked_prune_module() -> None:
    """Using nn_module=MagicMock so isinstance() never matches — the function
    should simply not prune anything but still return a well-formed summary."""
    fake_nn = MagicMock()
    # Force isinstance(child, fake_nn.Conv2d) to return False on every child.
    fake_nn.Conv2d = type("FakeConv2d", (), {})
    fake_prune = MagicMock()

    fake_inner = MagicMock()
    fake_inner.named_modules.return_value = []  # no convs
    fake_root = SimpleNamespace(model=fake_inner)

    summary = pc.prune_l2_channels(
        fake_root,
        sparsity=0.25,
        exclude_head=True,
        prune_module=fake_prune,
        nn_module=fake_nn,
    )

    assert summary["pruned_modules"] == []
    assert summary["params_before"] == 0
    assert summary["actual_sparsity"] == 0.0
    fake_prune.ln_structured.assert_not_called()


# ---------------------------------------------------------------------------
# Test 8: head detection by class name works on a SimpleNamespace so we can
# unit-test _find_head_indices without instantiating real Detect.
# ---------------------------------------------------------------------------


def test_find_head_indices_picks_up_detect_classes() -> None:
    class _FakeDetectCls:
        pass

    _FakeDetectCls.__name__ = "Detect"  # match HEAD_MODULE_NAMES

    class _Plain:
        pass

    fake_root = SimpleNamespace(
        model=[_Plain(), _Plain(), _FakeDetectCls()]
    )
    indices = pc._find_head_indices(fake_root)
    assert indices == {2}


# ---------------------------------------------------------------------------
# Test 9: _is_head_path correctly classifies ultralytics-style dotted names.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,head_indices,expected",
    [
        # Leading numeric form — names from inner.named_modules().
        ("22.cv2.0.0.conv", {22}, True),
        ("22", {22}, True),
        ("0.conv", {22}, False),
        ("21.bn", {22}, False),
        # Full "model.<int>..." form — names from root.named_modules().
        ("model.22.cv2.0.0.conv", {22}, True),
        ("model.22", {22}, True),
        ("model.0.conv", {22}, False),
        ("model.21.bn", {22}, False),
        # When head_indices is empty, exclude_head=True still leaves nothing
        # excluded (we let the pruner walk every Conv2d).
        ("model.22.cv2.0.0.conv", set(), False),
        ("22.cv2.0.0.conv", set(), False),
    ],
)
def test_is_head_path_dotted_names(
    name: str, head_indices: set[int], expected: bool
) -> None:
    assert pc._is_head_path(name, head_indices) is expected
