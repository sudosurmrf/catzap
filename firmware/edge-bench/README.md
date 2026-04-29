# edge-bench — TFLite Micro x86 host harness

Predicts ESP32-S3 viability for a quantized `.tflite` BEFORE the boards arrive.
Loads the model into TFLite Micro on a linux x86 host, runs N inferences, and
prints peak tensor arena, per-op timings, and median / p95 latency. A small
Python wrapper at `run_bench.py` parses the binary's stdout into a JSON
report under `training/edge/results/<story-id>-tflm.json`, applying a
documented x86 -> ESP32-S3 latency multiplier so the same number can be
compared across stories.

This story is US-010 of the edge-model-poc PRD; US-011 invokes this harness on
the quantized YOLOv8n-cat and NanoDet-cat candidates.

## Pinned upstream

- Repository: <https://github.com/tensorflow/tflite-micro>
- Pinned commit: `51bee03bed4776f1de88dd87226ff8c260f88e3c` (recorded in
  `build.sh` as `TFLM_COMMIT`).

The pin is reproducible — re-running `build.sh` against this commit produces
the same `libtensorflow-microlite.a`, the same op set, and the same arena
allocator behavior. Bumping the pin is a deliberate change to that contract.

## Build

```bash
firmware/edge-bench/build.sh
```

Output:

- `firmware/edge-bench/third_party/tflite-micro/` — clone (gitignored)
- `firmware/edge-bench/build/edgebench` — the linux x86_64 binary
- `firmware/edge-bench/models/hello_world.tflite` — symlink to the smallest
  shipped `.tflite` under TFLM `examples/` (used by the self-test)

`build.sh` is idempotent. Delete `third_party/` and `build/` to force a
fresh build.

## Self-test

```bash
firmware/edge-bench/build/edgebench firmware/edge-bench/models/hello_world.tflite 5
```

Should print a `=== EDGEBENCH START ===` block ending with
`allocate_tensors_status: ok`, an `=== OP_BREAKDOWN START ===` section with
at least one op, and a non-zero `arena_used_bytes` line.

## End-to-end run via Python wrapper

```bash
python firmware/edge-bench/run_bench.py \
    --model training/edge/models/yolov8n_cat_int8.tflite \
    --story-id US-011-yolov8n \
    --runs 50
```

Writes `training/edge/results/US-011-yolov8n-tflm.json`. The JSON shape is
documented in `run_bench.py` and asserted in
`training/edge/tests/test_tflm_bench_parser.py`.

## Output schema (one row per call)

```text
{
  "story_id": ...,
  "model_path": ...,
  "model_size_bytes": int,
  "runs": int,
  "arena_size_bytes": int,         # arena buffer size we passed in (default 8 MB)
  "arena_used_bytes": int,         # peak via MicroInterpreter::arena_used_bytes()
  "input_bytes": int,              # bytes of input(0) tensor
  "output_bytes": int,             # bytes of output(0) tensor
  "schema_status": "ok" | "mismatch_lib_<N>",
  "allocate_tensors_status": "ok" | "failed",
  "timed_invoke_status": "ok" | "failed_at_<i>",
  "profiler_overflowed": bool,
  "tflm_compatible": bool,         # convenience: schema_ok && alloc_ok && timed_ok
  "op_breakdown": [
    {"op_name": "CONV_2D", "count": <total across all runs>, "total_us": <sum>, "percent": <of timed wall-clock>}
  ],
  "raw_x86_us_p50": float,
  "raw_x86_us_p95": float,
  "predicted_s3_us_p50": float,    # raw_x86_us_p50 * x86_to_s3_multiplier
  "predicted_s3_fps": float,       # 1e6 / predicted_s3_us_p50
  "x86_to_s3_multiplier": float,
  "binary": "<path>",
  "tflm_commit": "<sha or 'unknown'>"
}
```

## Multiplier rationale

The default x86 -> ESP32-S3 multiplier is `8.0` (configurable via
`run_bench.py --multiplier`). The choice draws on three public data points:

1. The TFLM repo's own `tensorflow/lite/micro/docs/benchmarks.md` reports
   that representative INT8 vision workloads (the person-detection ~250KB
   model) run roughly 5x-10x slower on a single Xtensa LX7 core (ESP32-S3,
   240 MHz, no SIMD off the critical path) than on a modern x86_64 desktop
   CPU at 3-4 GHz. The midpoint of that band is ~7.5x.
2. ESP-NN-accelerated kernels (which TFLM picks up on ESP32-S3 via the
   ESP-NN integration) close some of the gap on Conv2D / DepthwiseConv2D
   but not all, and not on every op type — ADD, PAD, RESHAPE remain
   reference C++. Rounding up to 8x for a mixed-op model is conservative.
3. ESP-IDF's own benchmarks for the
   `examples/protocols/cat_face_detect`-style demos report ~6x-10x slowdowns
   vs equivalent x86 reference runs.

This is an order-of-magnitude estimate, not a calibrated number. Boards arrive
soon — once we run the same model on real S3 silicon, replace the multiplier
with the measured ratio. Until then: report the predicted number with
visible uncertainty, never as a guarantee. US-011's markdown report restates
the multiplier and its provenance for every row.

## TFLM compatibility caveats

- The op resolver (`firmware/edge-bench/main.cc:RegisterAllBuiltins`)
  registers ~80 builtin ops. Models using ops outside that set will fail
  AllocateTensors and print `allocate_tensors_status: failed` — that
  failure IS a finding (per US-011: "the failure itself is a finding").
- TFLM has limited Detect-head / NMS support. YOLOv8 typically post-
  processes outside the model; NanoDet-Plus' GFL head ditto. Custom ops
  baked into a `.tflite` (e.g. `TFLite_Detection_PostProcess`) will surface
  here as resolver failures.
- The default arena is 8 MB. ESP32-S3 has ~8 MB PSRAM total, so any model
  whose `arena_used_bytes` approaches 8 MB on the host should be flagged
  as not flashable as-is.

## Development notes

- The binary's stdout format is the integration contract with `run_bench.py`.
  Keep `=== EDGEBENCH START ===` / `=== EDGEBENCH END ===` markers stable;
  add new fields by appending key-value lines (parser ignores unknowns).
- Per-op timings are sampled via a `MicroProfilerInterface` subclass at
  `firmware/edge-bench/main.cc:TimingProfiler`. Warmup events (first 3
  invokes) are excluded from the breakdown to avoid skewing percentages
  with first-touch caching costs.
- Latency per `Invoke()` is timed at the call boundary (std::chrono
  steady_clock), independently of the profiler's per-op buckets. The two
  numbers agree within ~1% on a quiet machine.
