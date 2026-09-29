#!/usr/bin/env bash
# Build the frozen source on the pod. No credentials; no datasets; no checkpoints.
set -euo pipefail
source /venv/main/bin/activate
R=/workspace/results
mkdir -p /workspace/src "$R"
tar -xzf /workspace/kit/source.tar.gz -C /workspace/src
cd /workspace/src
export PYTHONPATH="$PWD/python:$PWD/oracle:$PWD" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -c 'import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name()); assert torch.cuda.is_available()'
if ! command -v g++ >/dev/null; then apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential; fi
need=("pybind11>=2.12" "pytest>=8" "hypothesis>=6")
command -v cmake >/dev/null || need+=("cmake>=3.24")
command -v ninja >/dev/null || need+=("ninja")
python -m pip install --disable-pip-version-check -q "${need[@]}"
python -m pip freeze > "$R/pip-freeze.txt"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v python)"
cmake --build build -j 16
./build/cpp/tests/gd_tests > "$R/gd-tests.log" 2>&1 && echo "gd_tests ok"
sha256sum python/gd/*.so > "$R/engine-sha256.txt"
python -m infra.cpu_budget --facts "$R/host-facts.json"
cat /proc/self/cgroup > "$R/cgroup.txt" || true
nvidia-smi > "$R/nvidia-smi.txt" || true
# CUDA/Triton bitwise gates, including >= 512-stream cases, before any timing.
# A failed gate turns that optimization off in the sweep (never report a wrong path).
gate() {  # name, test files...
  local name=$1; shift
  NVIDIA_TF32_OVERRIDE=0 timeout 600 python -m pytest -q -p no:cacheprovider "$@" \
    --junitxml "$R/gate-$name.xml" > "$R/gate-$name.log" 2>&1
  echo "gate $name exit $?"
}
gate triton tests/test_history_triton_cache.py tests/test_history_host_cache.py
gate graphs tests/test_history_cuda_graphs.py tests/test_history_graph_intervals.py \
  tests/test_history_host_rollout.py
gate wide tests/test_history_wide_projection.py
gate learner tests/test_history_learner_attention.py
# Merged snapshot inference (this branch): CUDA precision, sampling stream, slot
# refresh, and whole-rollout equality with the production CUDA options. -s keeps
# the printed max |merged - per-identity| log-prob differences in the log.
NVIDIA_TF32_OVERRIDE=0 timeout 900 python -m pytest -q -s -p no:cacheprovider \
  tests/test_history_snapshot_batch.py --junitxml "$R/gate-snapshot.xml" > "$R/gate-snapshot.log" 2>&1
echo "gate snapshot exit $?"
# Allocator cache trim (this branch): bitwise-identical training with and without
# the trim on CUDA, reserved never grows across a trim, and switching the merged
# arm on releases the snapshot private graphs' pools.
NVIDIA_TF32_OVERRIDE=0 timeout 900 python -m pytest -q -s -p no:cacheprovider \
  tests/test_history_trim.py --junitxml "$R/gate-trim.xml" > "$R/gate-trim.log" 2>&1
echo "gate trim exit $?"
# Data-parallel ranks (CPU tests; informational, not a gate).
timeout 600 python -m pytest -q -p no:cacheprovider tests/test_history_ddp.py > "$R/ddp-tests.log" 2>&1
echo "ddp tests exit $?"
python - "$R" <<'PYEOF'
import json, sys
import xml.etree.ElementTree as ET
from pathlib import Path
root = Path(sys.argv[1])
# Each gate needs zero failures and real CUDA cases that ran (not skipped).
required = {"triton": ["test_large_batch_encode_matches_eager_bits[cuda-520-None]",
                       "test_large_batch_encode_matches_eager_bits[cuda-520-97]",
                       "test_large_batch_encode_matches_eager_bits[cuda-1100-None]"],
            "graphs": ["test_private_graph_large_batch_matches_eager_bits[512-96-False]",
                       "test_private_graph_large_batch_matches_eager_bits[1024-33-True]"],
            "wide": ["test_wide_projection_matches_exact_path[cuda-0-257-64-64]"],
            "learner": ["test_batched_matches_per_match_values_and_gradients[cuda-64-9-130-513-4]"],
            "snapshot": ["test_production_size_step_matches_each_identity[cuda]",
                         "test_merged_log_probs_match_each_identity[cuda-multi-decision-auxiliary]",
                         "test_sampling_matches_per_identity_act_and_generator[cuda-one-per-stream]",
                         "test_sampled_choices_follow_each_identitys_distribution[cuda]",
                         "test_head_slots_refresh_on_every_resident_set_change[cuda]",
                         "test_collector_merged_matches_per_identity_rollout[cuda-True]",
                         "test_collector_refreshes_heads_when_snapshots_change[cuda]",
                         "test_production_cuda_options_with_merged_snapshots"],
            "trim": ["test_trim_leaves_training_bitwise_unchanged_on_cuda[on-first]",
                     "test_trim_leaves_training_bitwise_unchanged_on_cuda[off-first]",
                     "test_switching_the_arm_on_releases_snapshot_graph_pools",
                     "test_trim_leaves_training_bitwise_unchanged_on_cpu[true]"]}
gates = {}
for name, needed in required.items():
    path = root / f"gate-{name}.xml"
    record = dict(passed=False, file=path.name)
    if path.exists():
        cases = list(ET.parse(path).getroot().iter("testcase"))
        status = {}
        for case in cases:
            outcome = "passed"
            for tag in ("failure", "error", "skipped"):
                if case.find(tag) is not None:
                    outcome = tag
            status[case.get("name")] = outcome
        failed = [n for n, o in status.items() if o in ("failure", "error")]
        cuda_passed = sum(1 for n, o in status.items()
                          if o == "passed" and ("cuda" in n.lower() or "large_batch" in n))
        missing = [n for n in needed if status.get(n) != "passed"]
        record.update(tests=len(cases), failed=failed[:20], cuda_or_large_passed=cuda_passed,
                      required_missing=missing, passed=not failed and not missing)
    gates[name] = record
(root / "gates.json").write_text(json.dumps(gates, indent=2) + "\n")
print(json.dumps({k: v["passed"] for k, v in gates.items()}))
PYEOF
