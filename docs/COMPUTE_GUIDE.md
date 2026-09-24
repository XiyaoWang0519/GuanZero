# Compute guide: getting results faster

Suggestions, not rules. Hosts and inventory change from day to day; when the
preferred shape is unavailable, take the closest reasonable option, write down
what you got (`infra.cpu_budget --facts`), and move on. None of this replaces
the spend and lifecycle rules in `docs/TRAINING.md`.

Written 2026-09-24 from the reports cited below. Nothing here has been run as
a dedicated experiment yet; the gains are estimates.

## What limits us

Training is CPU-bound, not GPU-bound. The C++ engine and opponent play in
rollout collection dominate; the policy is about 5M parameters.

- B11 main spent 86% of each update in collection
  ([perf scan](reports/perf-scan-2026-09-23.md)).
- A 5090 with 64 vCPU kept its GPU only 44–73% busy
  ([compute options](reports/compute-options.md)).
- M1 reached about 63,200 decisions/s ([M1 report](reports/M1-runpod.md));
  the overnight main run on a 4090 with a 13.6–17.85 CPU quota averaged about
  32,000 ([overnight results](reports/overnight-results-2026-09-24.md),
  [4090 validation](reports/perf-gpu-2026-09-24.md)).

So a bigger GPU (H100, DGX Spark) is not the lever. More usable CPU per run,
and more runs in parallel, are.

## Suggestions, in order of payoff

### 1. Run experiment arms in parallel, one per pod

Prefer launching independent arms (control, filter, union, seeds) on separate
pods at the same time over running them in sequence on one GPU. Billing is by
the hour, so N pods for T hours costs about the same as one pod for N×T hours,
and results arrive N times sooner. Overnight, three 4,500 s arms ran in
sequence (about 3.75 h); in parallel they would have taken about 1.25 h.

Keep each pod's allowance and deadline under the usual per-experiment bounds.
If fewer pods are available than arms, co-locate arms and size threads with
`infra.cpu_budget --arms N`, as today. Matched arms should still share source,
seed, start checkpoint and opponent pool; note in the report when they ran on
different hosts, since host speed varies (the 4090 validation saw 28k–41k
decisions/s on unchanged code).

### 2. Prefer hosts with real CPU, and check it at startup

The listed vCPU count is not what a pod gets. The two 4090 pods exposed 64
logical CPUs but had 17.85 and 13.6 CPU quota. Aim for a host whose quota is
roughly 28 or more for a training arm; 16 is workable for evaluation and
smoke runs.

- Filter by vCPU when choosing (`--min-cpus`; the anonymous `lowestPrice`
  query with `minVcpuCount` shows vCPU-filtered prices). RTX 5090 with 64 vCPU
  was the best shape seen (community about $0.69/h, secure about $0.99/h).
- On the first poll, run `python -m infra.cpu_budget --facts` and look at the
  quota. If it is far below what you need and a better host is available,
  consider releasing the pod early and trying again; if not, keep it, scale
  the thread settings down, and record the quota in the report.
- A 3090 or A40 with 32 vCPU is fine for evaluation-only jobs.

### 3. Vast.ai, after RunPod credits are used up

Stay on RunPod while its credit lasts. Afterwards, Vast.ai is the planned
provider. Suggested filters: on-demand (not interruptible) for multi-hour
runs, `cpu_cores_effective >= 32`, `reliability > 0.98`, verified hosts,
4090/5090 class. Vast.ai offers a CLI and can self-destroy from inside the
instance, which covers the no-API-key-on-pod termination requirement
differently from `infra/runpod.py guard`; wire that up before the first
unattended run. Run a short `bench/ppo_throughput.py` on the first host to get
cost per million decisions against the RunPod baseline. `infra/runpod.py` does
not cover Vast.ai; it needs its own launch and guard path.

### 4. Longer term: separate actors from the learner

If one run is still too slow after 1–3, the structural fix is CPU-only actor
machines (RunPod CPU pods were about $0.03/vCPU-h) feeding one GPU learner.
This is a real engineering project (transport, staleness, failure handling)
and changes results, so it needs its own design note first.

### 5. Remaining code speedups

`docs/PERF_TODO.md` lists smaller wins (actor processes, rollout data kept on
the GPU, 4096×32 league shape), estimated at 1.1–1.35×. Worth doing, but
smaller than 1 and 2.

## Not recommended for training

- DGX Spark or other ARM desktops: roughly one 4090 pod's throughput for this
  workload (20 ARM cores), about $4k up front. Useful for local evaluation and
  larger-model supervised experiments, not a training speedup.
- Colab, Kaggle, Lightning free tier, Salad: 2–4 CPUs.
- Datacenter GPUs (H100, A100) on their own: pays for GPU we cannot use.
