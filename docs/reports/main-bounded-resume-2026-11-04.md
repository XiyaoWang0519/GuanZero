# Main bounded resume — November 4, 2026

Velocity job 19346 resumed main from u111202 (7,287,734,272 decisions).
The durable and staged checkpoint both matched the SHA-256 in the
[DanLM endpoint report](main-league-danlm-2026-11-04.md). The source was the
existing main `train-src` deployment, staged as `train-src-669c1de738fbca96`.

The run uses three allocated L40S GPUs, eight ranks, 128 environments and
eight minibatch matches per rank, the existing fast-lg2 mode and page-span.
MPS is off. The target is u120000 with a single 12-hour Slurm allocation
and an 11.5-hour trainer deadline. No automatic successor is configured.
The sync loop is configured for five-minute intervals and final sync.

Startup observation: node-local metrics advanced to u111204 and
7,287,865,344 decisions. This confirms training progress at startup.
Durable sync and target completion remained pending at this observation.
Remote launcher output: `~/guanzero/runs/main/slurm-19346.out`.

## User-authorized automatic continuation

Job 19347 was queued with `afterok:19346`. The existing allocation retains
its original u120000 stop and wall-clock limit. Starting with 19347, there
is no experimental update target. Each successor uses a 12-hour allocation
and submits one dependent successor after a normal deadline stop, an
advancing finite checkpoint and successful durable sync. STOP, abnormal
exit or failed handoff break the chain. The first successor also accepts a
clean target exit from 19346. Its first handoff has not yet executed.
The remote script is `~/guanzero/main-auto-2026-11-04.sbatch`.

The staged main source still forces math SDPA in the CUDA gradient path
that caused the October 15 incident. In 30 live updates u111209–u111238,
actor gradient norm ranged 0.398–0.467, encoder 0.0041–0.0579, critic
11.17–15.72, and all observed losses and KL were finite. A later node-local
check reached u111258. This is a startup health observation.
