// edgebench — TFLite Micro x86 host harness for predicting ESP32-S3 viability.
//
// Loads a .tflite, runs N inferences, and prints a machine-parsable report:
//   - peak tensor arena bytes used (after AllocateTensors)
//   - per-op breakdown: tag, count across all runs, total micro-seconds
//   - median / p95 / min / max / mean per-Invoke latency in micro-seconds
//
// Usage: edgebench <model.tflite> [runs=50] [arena_kb=8192]
//
// The output format is consumed by firmware/edge-bench/run_bench.py and the
// parser tested in training/edge/tests/test_tflm_bench_parser.py.
//
// The op resolver registers a wide builtin set (see RegisterAllBuiltins below)
// so the same binary can run YOLOv8n-cat INT8 and NanoDet-cat INT8 without a
// rebuild. If a model uses an op not in the list, AllocateTensors() fails and
// the binary prints `allocate_tensors_status: failed` plus the missing-op log
// from the underlying MicroPrintf — that failure IS a finding (per US-011).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

uint64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Records one (tag, start_ns, end_ns) per BeginEvent/EndEvent pair so we can
// aggregate per op type across all Invoke() calls. ~32 bytes per event; 1M
// events ≈ 32 MB — far more than enough for 50 runs of any model in the PoC.
class TimingProfiler : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char* tag) override {
    if (tag == nullptr) tag = "(null)";
    if (events_.size() >= kMaxEvents) {
      overflowed_ = true;
      return kInvalidHandle;
    }
    Event e;
    e.tag = tag;
    e.start_ns = NowNs();
    e.end_ns = e.start_ns;
    events_.push_back(std::move(e));
    return static_cast<uint32_t>(events_.size() - 1);
  }

  void EndEvent(uint32_t handle) override {
    if (handle == kInvalidHandle || handle >= events_.size()) return;
    events_[handle].end_ns = NowNs();
  }

  struct Event {
    std::string tag;
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
  };

  const std::vector<Event>& events() const { return events_; }
  bool overflowed() const { return overflowed_; }

 private:
  static constexpr uint32_t kInvalidHandle = 0xFFFFFFFFu;
  static constexpr size_t kMaxEvents = 1u << 20;
  std::vector<Event> events_;
  bool overflowed_ = false;
};

// Register a comprehensive builtin set so YOLOv8n-cat INT8 and NanoDet-cat
// INT8 both load. Order doesn't matter; capacity is set to 128 which is well
// above the ~80 ops we register.
TfLiteStatus RegisterAllBuiltins(tflite::MicroMutableOpResolver<128>& r) {
  if (r.AddAdd() != kTfLiteOk) return kTfLiteError;
  if (r.AddArgMax() != kTfLiteOk) return kTfLiteError;
  if (r.AddArgMin() != kTfLiteOk) return kTfLiteError;
  if (r.AddAveragePool2D() != kTfLiteOk) return kTfLiteError;
  if (r.AddBatchToSpaceNd() != kTfLiteOk) return kTfLiteError;
  if (r.AddCast() != kTfLiteOk) return kTfLiteError;
  if (r.AddCeil() != kTfLiteOk) return kTfLiteError;
  if (r.AddConcatenation() != kTfLiteOk) return kTfLiteError;
  if (r.AddConv2D() != kTfLiteOk) return kTfLiteError;
  if (r.AddCos() != kTfLiteOk) return kTfLiteError;
  if (r.AddDepthToSpace() != kTfLiteOk) return kTfLiteError;
  if (r.AddDepthwiseConv2D() != kTfLiteOk) return kTfLiteError;
  if (r.AddDequantize() != kTfLiteOk) return kTfLiteError;
  if (r.AddDiv() != kTfLiteOk) return kTfLiteError;
  if (r.AddElu() != kTfLiteOk) return kTfLiteError;
  if (r.AddEqual() != kTfLiteOk) return kTfLiteError;
  if (r.AddExp() != kTfLiteOk) return kTfLiteError;
  if (r.AddExpandDims() != kTfLiteOk) return kTfLiteError;
  if (r.AddFill() != kTfLiteOk) return kTfLiteError;
  if (r.AddFloor() != kTfLiteOk) return kTfLiteError;
  if (r.AddFloorDiv() != kTfLiteOk) return kTfLiteError;
  if (r.AddFloorMod() != kTfLiteOk) return kTfLiteError;
  if (r.AddFullyConnected() != kTfLiteOk) return kTfLiteError;
  if (r.AddGather() != kTfLiteOk) return kTfLiteError;
  if (r.AddGatherNd() != kTfLiteOk) return kTfLiteError;
  if (r.AddGreater() != kTfLiteOk) return kTfLiteError;
  if (r.AddGreaterEqual() != kTfLiteOk) return kTfLiteError;
  if (r.AddHardSwish() != kTfLiteOk) return kTfLiteError;
  if (r.AddL2Normalization() != kTfLiteOk) return kTfLiteError;
  if (r.AddLeakyRelu() != kTfLiteOk) return kTfLiteError;
  if (r.AddLess() != kTfLiteOk) return kTfLiteError;
  if (r.AddLessEqual() != kTfLiteOk) return kTfLiteError;
  if (r.AddLog() != kTfLiteOk) return kTfLiteError;
  if (r.AddLogicalAnd() != kTfLiteOk) return kTfLiteError;
  if (r.AddLogicalNot() != kTfLiteOk) return kTfLiteError;
  if (r.AddLogicalOr() != kTfLiteOk) return kTfLiteError;
  if (r.AddLogistic() != kTfLiteOk) return kTfLiteError;
  if (r.AddMaximum() != kTfLiteOk) return kTfLiteError;
  if (r.AddMaxPool2D() != kTfLiteOk) return kTfLiteError;
  if (r.AddMean() != kTfLiteOk) return kTfLiteError;
  if (r.AddMinimum() != kTfLiteOk) return kTfLiteError;
  if (r.AddMirrorPad() != kTfLiteOk) return kTfLiteError;
  if (r.AddMul() != kTfLiteOk) return kTfLiteError;
  if (r.AddNeg() != kTfLiteOk) return kTfLiteError;
  if (r.AddNotEqual() != kTfLiteOk) return kTfLiteError;
  if (r.AddPack() != kTfLiteOk) return kTfLiteError;
  if (r.AddPad() != kTfLiteOk) return kTfLiteError;
  if (r.AddPadV2() != kTfLiteOk) return kTfLiteError;
  if (r.AddPrelu() != kTfLiteOk) return kTfLiteError;
  if (r.AddQuantize() != kTfLiteOk) return kTfLiteError;
  if (r.AddReduceMax() != kTfLiteOk) return kTfLiteError;
  if (r.AddRelu() != kTfLiteOk) return kTfLiteError;
  if (r.AddRelu6() != kTfLiteOk) return kTfLiteError;
  if (r.AddReshape() != kTfLiteOk) return kTfLiteError;
  if (r.AddResizeBilinear() != kTfLiteOk) return kTfLiteError;
  if (r.AddResizeNearestNeighbor() != kTfLiteOk) return kTfLiteError;
  if (r.AddRound() != kTfLiteOk) return kTfLiteError;
  if (r.AddRsqrt() != kTfLiteOk) return kTfLiteError;
  if (r.AddShape() != kTfLiteOk) return kTfLiteError;
  if (r.AddSin() != kTfLiteOk) return kTfLiteError;
  if (r.AddSlice() != kTfLiteOk) return kTfLiteError;
  if (r.AddSoftmax() != kTfLiteOk) return kTfLiteError;
  if (r.AddSpaceToBatchNd() != kTfLiteOk) return kTfLiteError;
  if (r.AddSpaceToDepth() != kTfLiteOk) return kTfLiteError;
  if (r.AddSplit() != kTfLiteOk) return kTfLiteError;
  if (r.AddSplitV() != kTfLiteOk) return kTfLiteError;
  if (r.AddSqrt() != kTfLiteOk) return kTfLiteError;
  if (r.AddSquare() != kTfLiteOk) return kTfLiteError;
  if (r.AddSquaredDifference() != kTfLiteOk) return kTfLiteError;
  if (r.AddSqueeze() != kTfLiteOk) return kTfLiteError;
  if (r.AddStridedSlice() != kTfLiteOk) return kTfLiteError;
  if (r.AddSub() != kTfLiteOk) return kTfLiteError;
  if (r.AddSum() != kTfLiteOk) return kTfLiteError;
  if (r.AddSvdf() != kTfLiteOk) return kTfLiteError;
  if (r.AddTanh() != kTfLiteOk) return kTfLiteError;
  if (r.AddTranspose() != kTfLiteOk) return kTfLiteError;
  if (r.AddTransposeConv() != kTfLiteOk) return kTfLiteError;
  if (r.AddUnpack() != kTfLiteOk) return kTfLiteError;
  if (r.AddZerosLike() != kTfLiteOk) return kTfLiteError;
  return kTfLiteOk;
}

bool ReadFile(const char* path, std::vector<char>* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  std::streamsize sz = f.tellg();
  if (sz <= 0) return false;
  f.seekg(0, std::ios::beg);
  out->resize(static_cast<size_t>(sz));
  return static_cast<bool>(f.read(out->data(), sz));
}

double Percentile(std::vector<uint64_t> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  double idx = p * (v.size() - 1);
  size_t lo = static_cast<size_t>(idx);
  size_t hi = std::min(lo + 1, v.size() - 1);
  double frac = idx - lo;
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <model.tflite> [runs=50] [arena_kb=8192]\n",
                 argv[0]);
    return 2;
  }
  const char* model_path = argv[1];
  const int runs = (argc >= 3) ? std::atoi(argv[2]) : 50;
  const int arena_kb = (argc >= 4) ? std::atoi(argv[3]) : 8192;
  if (runs <= 0 || arena_kb <= 0) {
    std::fprintf(stderr, "runs and arena_kb must be > 0\n");
    return 2;
  }

  std::vector<char> model_buf;
  if (!ReadFile(model_path, &model_buf)) {
    std::fprintf(stderr, "cannot read %s\n", model_path);
    return 3;
  }

  std::printf("=== EDGEBENCH START ===\n");
  std::printf("edgebench_version: 1\n");
  std::printf("model_path: %s\n", model_path);
  std::printf("model_size_bytes: %zu\n", model_buf.size());
  std::printf("runs: %d\n", runs);

  const tflite::Model* model = tflite::GetModel(model_buf.data());
  std::printf("schema_version: %u\n", model->version());
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    std::printf("schema_status: mismatch_lib_%d\n", TFLITE_SCHEMA_VERSION);
    std::printf("=== EDGEBENCH END ===\n");
    return 4;
  }
  std::printf("schema_status: ok\n");

  static tflite::MicroMutableOpResolver<128> resolver;
  if (RegisterAllBuiltins(resolver) != kTfLiteOk) {
    std::printf("resolver_status: register_failed\n");
    std::printf("=== EDGEBENCH END ===\n");
    return 5;
  }

  const size_t arena_size = static_cast<size_t>(arena_kb) * 1024u;
  std::vector<uint8_t> arena(arena_size);
  std::printf("arena_size_bytes: %zu\n", arena_size);

  TimingProfiler profiler;
  tflite::MicroInterpreter interpreter(model, resolver, arena.data(),
                                       arena_size, /*resource_variables=*/nullptr,
                                       &profiler);
  TfLiteStatus alloc_status = interpreter.AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    std::printf("allocate_tensors_status: failed\n");
    std::printf("=== EDGEBENCH END ===\n");
    return 6;
  }
  std::printf("allocate_tensors_status: ok\n");
  std::printf("arena_used_bytes: %zu\n", interpreter.arena_used_bytes());

  TfLiteTensor* in = interpreter.input(0);
  TfLiteTensor* out = interpreter.output(0);
  std::printf("input_bytes: %zu\n", in ? in->bytes : 0u);
  std::printf("output_bytes: %zu\n", out ? out->bytes : 0u);
  if (in && in->data.raw) std::memset(in->data.raw, 0, in->bytes);

  // Warmup
  constexpr int kWarmup = 3;
  for (int i = 0; i < kWarmup; ++i) {
    if (interpreter.Invoke() != kTfLiteOk) {
      std::printf("warmup_invoke_status: failed_at_%d\n", i);
      std::printf("=== EDGEBENCH END ===\n");
      return 7;
    }
  }

  // Drop warmup events; aggregate over the timed window only.
  std::vector<TimingProfiler::Event> warmup_events = profiler.events();
  // Reset by destructive copy — TimingProfiler doesn't expose Clear, so we
  // walk the timed events as everything past the warmup count.
  const size_t warmup_event_count = warmup_events.size();

  std::vector<uint64_t> per_invoke_us;
  per_invoke_us.reserve(runs);
  for (int i = 0; i < runs; ++i) {
    uint64_t t0 = NowNs();
    if (interpreter.Invoke() != kTfLiteOk) {
      std::printf("timed_invoke_status: failed_at_%d\n", i);
      std::printf("=== EDGEBENCH END ===\n");
      return 8;
    }
    uint64_t t1 = NowNs();
    per_invoke_us.push_back((t1 - t0 + 500) / 1000);
  }
  std::printf("timed_invoke_status: ok\n");

  // Aggregate per-op stats across the timed window only.
  struct Agg { uint64_t count = 0; uint64_t total_ns = 0; };
  std::map<std::string, Agg> agg;
  const auto& events = profiler.events();
  for (size_t i = warmup_event_count; i < events.size(); ++i) {
    const auto& e = events[i];
    auto& a = agg[e.tag];
    a.count += 1;
    if (e.end_ns >= e.start_ns) a.total_ns += (e.end_ns - e.start_ns);
  }

  std::printf("profiler_overflowed: %s\n", profiler.overflowed() ? "true" : "false");
  std::printf("=== OP_BREAKDOWN START ===\n");
  for (const auto& kv : agg) {
    uint64_t total_us = (kv.second.total_ns + 500) / 1000;
    std::printf("op_name: %s\tcount: %llu\ttotal_us: %llu\n",
                kv.first.c_str(),
                static_cast<unsigned long long>(kv.second.count),
                static_cast<unsigned long long>(total_us));
  }
  std::printf("=== OP_BREAKDOWN END ===\n");

  double p50 = Percentile(per_invoke_us, 0.5);
  double p95 = Percentile(per_invoke_us, 0.95);
  uint64_t mn = *std::min_element(per_invoke_us.begin(), per_invoke_us.end());
  uint64_t mx = *std::max_element(per_invoke_us.begin(), per_invoke_us.end());
  uint64_t total = 0;
  for (uint64_t v : per_invoke_us) total += v;
  double mean = static_cast<double>(total) / per_invoke_us.size();

  std::printf("total_invoke_runs: %zu\n", per_invoke_us.size());
  std::printf("total_invoke_us_p50: %.0f\n", p50);
  std::printf("total_invoke_us_p95: %.0f\n", p95);
  std::printf("total_invoke_us_min: %llu\n",
              static_cast<unsigned long long>(mn));
  std::printf("total_invoke_us_max: %llu\n",
              static_cast<unsigned long long>(mx));
  std::printf("total_invoke_us_mean: %.0f\n", mean);
  std::printf("=== EDGEBENCH END ===\n");
  return 0;
}
