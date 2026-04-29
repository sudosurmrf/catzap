"""Run the edgebench binary on a .tflite and write a TFLM benchmark JSON.

Usage:
    python firmware/edge-bench/run_bench.py \\
        --model training/edge/models/yolov8n_cat_int8.tflite \\
        --story-id US-011-yolov8n \\
        --runs 50

Output:
    training/edge/results/<story-id>-tflm.json with shape:
      {
        "story_id": "...",
        "model_path": "...",
        "model_size_bytes": int,
        "runs": int,
        "arena_size_bytes": int,
        "arena_used_bytes": int,
        "input_bytes": int,
        "output_bytes": int,
        "schema_status": "ok" | "...",
        "allocate_tensors_status": "ok" | "failed",
        "tflm_compatible": bool,
        "op_breakdown": [
          {"op_name": "CONV_2D", "count": int, "total_us": int, "percent": float},
          ...
        ],
        "raw_x86_us_p50": int,
        "raw_x86_us_p95": int,
        "raw_x86_us_min": int,
        "raw_x86_us_max": int,
        "raw_x86_us_mean": float,
        "predicted_s3_us_p50": float,
        "predicted_s3_us_p95": float,
        "predicted_s3_fps": float,
        "x86_to_s3_multiplier": float,
        "binary": "<path>",
        "tflm_commit": "<sha or unknown>",
      }

The x86 -> ESP32-S3 multiplier is documented in firmware/edge-bench/README.md.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Default x86 -> ESP32-S3 latency multiplier. Sourced from public TFLM
# benchmarks (see firmware/edge-bench/README.md "Multiplier rationale" for
# citation and uncertainty bounds).
DEFAULT_X86_TO_S3_MULTIPLIER = 8.0

START_MARKER = "=== EDGEBENCH START ==="
END_MARKER = "=== EDGEBENCH END ==="
OP_START = "=== OP_BREAKDOWN START ==="
OP_END = "=== OP_BREAKDOWN END ==="

_KV_RE = re.compile(r"^([a-z_][a-z0-9_]*):\s*(.+?)\s*$")
_OP_RE = re.compile(
    r"^op_name:\s*(\S+)\s+count:\s*(\d+)\s+total_us:\s*(\d+)\s*$"
)

_INT_KEYS = {
    "model_size_bytes",
    "runs",
    "arena_size_bytes",
    "arena_used_bytes",
    "input_bytes",
    "output_bytes",
    "schema_version",
    "total_invoke_runs",
}
_FLOAT_KEYS = {
    "total_invoke_us_p50",
    "total_invoke_us_p95",
    "total_invoke_us_min",
    "total_invoke_us_max",
    "total_invoke_us_mean",
}
_STR_KEYS = {
    "edgebench_version",
    "model_path",
    "schema_status",
    "resolver_status",
    "allocate_tensors_status",
    "warmup_invoke_status",
    "timed_invoke_status",
    "profiler_overflowed",
}


@dataclass
class ParsedReport:
    fields: dict
    op_breakdown: list[dict]


def parse_edgebench_stdout(stdout: str) -> ParsedReport:
    """Parse the machine-readable section of edgebench stdout.

    Lines outside the START/END markers (e.g. MicroPrintf output) are ignored.
    Unknown keys are kept as strings under fields. Order of op rows is
    preserved.
    """
    if START_MARKER not in stdout or END_MARKER not in stdout:
        raise ValueError(
            "edgebench stdout did not contain start/end markers; "
            "binary likely failed before printing"
        )
    body = stdout.split(START_MARKER, 1)[1].split(END_MARKER, 1)[0]
    lines = body.splitlines()
    fields: dict = {}
    ops: list[dict] = []
    in_ops = False
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        if line == OP_START:
            in_ops = True
            continue
        if line == OP_END:
            in_ops = False
            continue
        if in_ops:
            m = _OP_RE.match(line)
            if not m:
                continue
            ops.append(
                {
                    "op_name": m.group(1),
                    "count": int(m.group(2)),
                    "total_us": int(m.group(3)),
                }
            )
            continue
        m = _KV_RE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if key in _INT_KEYS:
            fields[key] = int(val)
        elif key in _FLOAT_KEYS:
            fields[key] = float(val)
        else:
            fields[key] = val
    total_us = sum(o["total_us"] for o in ops) or 1
    for o in ops:
        o["percent"] = round(100.0 * o["total_us"] / total_us, 3)
    return ParsedReport(fields=fields, op_breakdown=ops)


def report_to_result(
    parsed: ParsedReport,
    *,
    story_id: str,
    binary_path: str,
    tflm_commit: str,
    multiplier: float = DEFAULT_X86_TO_S3_MULTIPLIER,
) -> dict:
    """Convert a ParsedReport into the on-disk JSON shape."""
    f = parsed.fields
    p50 = f.get("total_invoke_us_p50", 0.0)
    p95 = f.get("total_invoke_us_p95", 0.0)
    pred_p50 = float(p50) * multiplier
    pred_p95 = float(p95) * multiplier
    fps = (1_000_000.0 / pred_p50) if pred_p50 > 0 else 0.0
    alloc_ok = f.get("allocate_tensors_status") == "ok"
    timed_ok = f.get("timed_invoke_status", "ok") == "ok"
    schema_ok = f.get("schema_status") == "ok"
    return {
        "story_id": story_id,
        "model_path": f.get("model_path", ""),
        "model_size_bytes": int(f.get("model_size_bytes", 0)),
        "runs": int(f.get("runs", 0)),
        "arena_size_bytes": int(f.get("arena_size_bytes", 0)),
        "arena_used_bytes": int(f.get("arena_used_bytes", 0)),
        "input_bytes": int(f.get("input_bytes", 0)),
        "output_bytes": int(f.get("output_bytes", 0)),
        "schema_status": f.get("schema_status", "unknown"),
        "allocate_tensors_status": f.get("allocate_tensors_status", "unknown"),
        "timed_invoke_status": f.get("timed_invoke_status", "unknown"),
        "profiler_overflowed": f.get("profiler_overflowed", "false") == "true",
        "tflm_compatible": bool(alloc_ok and timed_ok and schema_ok),
        "op_breakdown": parsed.op_breakdown,
        "raw_x86_us_p50": float(p50),
        "raw_x86_us_p95": float(p95),
        "raw_x86_us_min": float(f.get("total_invoke_us_min", 0.0)),
        "raw_x86_us_max": float(f.get("total_invoke_us_max", 0.0)),
        "raw_x86_us_mean": float(f.get("total_invoke_us_mean", 0.0)),
        "x86_to_s3_multiplier": float(multiplier),
        "predicted_s3_us_p50": pred_p50,
        "predicted_s3_us_p95": pred_p95,
        "predicted_s3_fps": fps,
        "binary": binary_path,
        "tflm_commit": tflm_commit,
    }


def top_k_ops(op_breakdown: list[dict], k: int = 3) -> list[dict]:
    """Helper used by US-011 aggregation: top-k ops by total_us, descending."""
    return sorted(op_breakdown, key=lambda o: o["total_us"], reverse=True)[:k]


def _resolve_binary(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
    else:
        p = Path(__file__).parent / "build" / "edgebench"
    if not p.exists():
        raise SystemExit(
            f"edgebench binary not found at {p}. Run firmware/edge-bench/build.sh first."
        )
    return p


def _read_tflm_commit() -> str:
    """Return the pinned TFLM commit recorded by build.sh, if available."""
    third_party = Path(__file__).parent / "third_party" / "tflite-micro"
    git_head = third_party / ".git" / "HEAD"
    if not git_head.exists():
        return "unknown"
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        out = subprocess.run(
            [git, "-C", str(third_party), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def run(
    *,
    binary: Path,
    model: Path,
    runs: int,
    arena_kb: int,
    runner: callable | None = None,
) -> str:
    """Invoke the binary and return its stdout. `runner` is a DI seam for tests."""
    cmd = [str(binary), str(model), str(runs), str(arena_kb)]
    if runner is None:
        runner = subprocess.run
    proc = runner(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        # We still try to parse — if the binary printed up to EDGEBENCH END
        # before failing, those fields are useful for diagnosing TFLM-incompat.
    return proc.stdout


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_bench")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--story-id", required=True)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--arena-kb", type=int, default=8192)
    ap.add_argument("--binary", default=None)
    ap.add_argument(
        "--multiplier",
        type=float,
        default=DEFAULT_X86_TO_S3_MULTIPLIER,
        help="x86 -> ESP32-S3 latency multiplier (see README.md)",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("training/edge/results"),
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    binary = _resolve_binary(args.binary)
    stdout = run(
        binary=binary, model=args.model, runs=args.runs, arena_kb=args.arena_kb
    )
    parsed = parse_edgebench_stdout(stdout)
    result = report_to_result(
        parsed,
        story_id=args.story_id,
        binary_path=str(binary),
        tflm_commit=_read_tflm_commit(),
        multiplier=args.multiplier,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{args.story_id}-tflm.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
