# Compute guide: getting results faster

Suggestions, not rules. Hosts and inventory change from day to day; when the
preferred shape is unavailable, take the closest reasonable option, write down
what you got (`infra.cpu_budget --facts`), and move on. None of this replaces
the spend and lifecycle rules in `docs/TRAINING.md`.

Written 2026-09-24 from the reports cited below and revised the same day
after the collection profile. Nothing here has been run as a dedicated
experiment yet; the gains are estimates.

## What limits us

Written first as "training is CPU-bound", then as "inference-bound" after a
Mac profile; the RTX 4090 profile of September 24
([GPU pass](reports/perf-gpu-2026-09-24b.md)) settles it. On CUDA, one
Python process runs the engine, host-side gathers, uploads, forwards and
buffer writes one after another, so the GPU waits on the host most of the
time. The limit is host work per step and per update, and how much of it can
overlap.

- Steady-state profiles on 4090 pods: the GPU was busy only 40% of a
  collect step on the frozen arm and 36-43% on the league arm, with 281 and
  342 kernel launches per step, and 53-57% during learn. Collect and learn
  took about equal time per update.
- The engine's `pending` is the largest single host item (about 4.4 ms of a
  15 ms frozen step) and already runs on the engine's thread pool; buffer
  writes, uploads and launch overhead follow. On the league arm, copying
  opponent features for every opponent row was the other large item
  (2.7-4.7 ms per step) until they became lazy (league collect 15% faster).
  On a Mac CPU the same collect is instead dominated by the forwards
  (reference 56%, policy 30%, engine 3%;
  [learner pass](reports/perf-learner-2026-09-24.md)).
- More CPU helps when it runs in parallel: four actor processes at four
  engine threads each cut collect time 1.4-1.8x on that pod (statistically
  equivalent, not bitwise; see PERF_TODO). Raising engine threads within
  one process from 4 to 16 took the frozen arm only from 49k to 60k
  decisions/s ([4090 validation](reports/perf-gpu-2026-09-24.md)).
- Shared hosts are noisy. With identical code and optimizer-step counts,
  one learner step took 19-70 ms from run to run at a host load of about 70.
  Compare code only on one pod, alternating runs, and report medians and
  minimums.
- A 5090 with 64 vCPU kept its GPU only 44–73% busy
  ([compute options](reports/compute-options.md)); that "64 vCPU" was
  `os.cpu_count()`, never the quota. The two 4090 pods of September 24 had
  a 31-CPU quota; earlier ones had 13.6 and 17.85.
- M1 reached about 63,200 decisions/s ([M1 report](reports/M1-runpod.md));
  the overnight main run on a 4090 with a 13.6–17.85 CPU quota averaged about
  32,000 ([overnight results](reports/overnight-results-2026-09-24.md)).

So a bigger GPU on its own will not help. What helps is real CPU quota used
in parallel (actor processes, parallel arms) and less host work per step.

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
logical CPUs but had 17.85 and 13.6 CPU quota. Aim for about 16 dedicated
CPUs per training arm (N×16 when co-locating N arms); 16 is workable for evaluation and
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
runs, `num_gpus=1` and `gpu_frac=1` (the whole machine; a fraction of a
multi-GPU host only reserves its GPU share of the CPU, like RunPod),
`cpu_cores_effective >= 32` (threads, not cores), `reliability >= 0.98`,
verified hosts, 4090/5090 class. On Sept. 24 such whole machines listed at
about $0.47/h (4090) and $0.60/h (5090) with 48-64 threads
([CPU isolation survey](reports/compute-cpu-isolation-2026-09-24.md)). Vast.ai offers a CLI and can self-destroy from inside the
instance, which covers the no-API-key-on-pod termination requirement
differently from `infra/runpod.py guard`; wire that up before the first
unattended run. Run a short `bench/ppo_throughput.py` on the first host to get
cost per million decisions against the RunPod baseline. `infra/runpod.py` does
not cover Vast.ai; it needs its own launch and guard path.

### 4. Longer term: separate actors from the learner

Not with CPU-only actors. The forwards are GPU-shaped work (on a CPU they
are 86% of collection), so a CPU machine is the slowest and most expensive
place for them (RunPod CPU is about
$0.03/vCPU-h, so 64 vCPU costs more than a whole 5090 pod), and one Mac
(about 80k decisions/s frozen, about 28k league) could not keep up with a
4090 learner for league arms, let alone a faster GPU. If one GPU still
cannot keep up after same-GPU actor processes, the design is several cheap
GPU actor pods feeding one learner, sending deal seeds and action sequences
rather than feature rows (about 340 MB of rows per update against under
1 MB). It changes results (at least one update of staleness), so it needs
its own design note and a matched evaluation first.

### 5. Remaining code speedups

`docs/PERF_TODO.md` lists them. The exact ones are done and measured on a
4090: rollout data kept on the device for learning, the reference skip (no
GPU effect), and lazy opponent features (league collect 15% faster). The
largest measured lever left is four actor processes per arm (1.15–1.3× end
to end on the frozen arm, single runs); it is statistically equivalent, not
bitwise, so it needs a paired measurement and a decision before it becomes a
default. Changes that alter learning (4096×32 league shape, async
collection) wait for a matched evaluation.

## Not recommended for training

- DGX Spark or other ARM desktops: roughly one 4090 pod's throughput for this
  workload (20 ARM cores), about $4k up front. Useful for local evaluation and
  larger-model supervised experiments, not a training speedup.
- Colab, Kaggle, Lightning free tier, Salad: 2–4 CPUs.
- Datacenter GPUs (H100, A100) on their own: pays for GPU we cannot use.
