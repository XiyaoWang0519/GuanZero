# Search distillation: plan (October 6, 2026)

Goal: move part of the test-time search gain into the plain network, so the
model plays better without search and search starts from a stronger blueprint.
This is a plan, not a result. Nothing below is implemented yet.

## Why now

- The search gain against DanLM has stayed flat while the model improved:
  +0.31 (u9989), +0.30 (u15094), six KV-path checkpoints pooled +0.32, and
  **m1 (u27900) +0.32 [+0.14, +0.50]** on 200 paired deals. Training has not
  absorbed what search adds.
- +0.3 levels/round is worth about 0.7 doublings of training at the current
  slope (+0.44 per doubling from m0 to m1), which is roughly one more day of
  4-GPU training from 2.1B decisions. Distillation is worth it if it keeps a
  substantial share of that for less.
- A larger search budget does not help: on m1, 128 worlds vs 32 worlds is
  −0.025 [−0.21, +0.16] paired, at 4x the cost. Labels can use the standard
  32-world search.

## Decisions needed from the user

1. **Design boundary.** DESIGN.md 1.1 forbids distillation from old MLPs, and
   section 3 lists engine search as "a separate, deferred mechanism". Stage 1
   below is an offline experiment and changes no training entrypoint.
   Stage 2 needs a DESIGN.md amendment: search labels made by current or past
   Transformer versions of this lineage become a training signal. Nothing
   from the MLPs or DanLM.
2. **Local training.** Stage 1 fine-tunes on the Mac (MPS). The time is not
   measured yet; the estimate is 1 to 2 hours for all three arms.
3. **GPU split for stage 2.** See "Stage 2" for the 2+2 option.

## Labels

Self-play with the parent in all four seats, house rules, the same heuristic
tribute as training, the standard search config (`search.json`: trigger
`hand_or_unsure`, top 8 candidates, 32 uniform worlds, 10 s, mean selection,
margin 0.5, KV rollouts). Every triggered decision records the candidates,
their rollout means, the world count, the blueprint's choice and the search's
choice. Untriggered decisions of the same matches are kept unlabeled for the
anchor term below. No DanLM games: the target is general strength, not a
DanLM exploiter.

Parent: **m1 (u27900)**, which is already on the Mac with DanLM and search
numbers. The experiment does not wait for the main run.

Storage: per match, the deal seed and the full action list; per searched
decision, the label row. The trainer rebuilds the history streams by
replaying the match through the engine, the same path the history policy
uses at evaluation. Shards with a manifest and a match-disjoint train/val
split, as in `eval/collect_critic.py`.

Volume: m1 with w32 ran 8,023 searches in 17.6 minutes with 10 CPU workers,
plain arm included, so about 27,000 labeled decisions an hour. One 8-hour
night gives about 200,000, about 10% of them overrides (784 of 8,023 on m1).
Server CPU jobs (short partition, 4 h) can add more if needed.

Two label variants from the same data:

- **H (hard):** the action the search played, which is the baseline when it
  did not override. This is exactly what produced the +0.32.
- **S (soft):** over the candidates, `q(a) ∝ pi_parent(a) · exp(mean(a) / tau)`,
  with tau chosen on the validation split so that argmax q agrees with the
  search choice on at least 95% of decisions.

## Stage 1: offline fine-tune (the pilot)

From m1, actor only (critic and aux heads frozen), learning rate 3e-5, 1 to
3 epochs with early stopping on validation loss over override decisions:

`loss = CE(label, pi) on searched decisions + beta · KL(pi_m1 || pi) on all decisions`

The KL anchor keeps the policy close to m1 away from searched positions and
guards against entropy collapse from hard labels.

Arms, all on the same data:

| Arm | Labels | Purpose |
|---|---|---|
| C | the blueprint's own choice everywhere | control: does the fine-tune itself change strength? |
| H | search choice (hard) | main arm |
| S | soft search distribution | label-noise check |

Measurements:

- Against DanLM, 4,000 deals on the Mac (about 4 minutes each): m1, C, H, S.
  The 95% interval is about ±0.04 per point, and differences on the same deals
  are tighter.
- Internal: lineage_eval against m1 (2,000 deals) and V1T-linux (4,000), on
  the server.
- Search on top of the best arm: 200 deals with w32. Does search still add,
  and how much?
- Diagnostics: agreement with the search choice on validation overrides;
  entropy change against m1.

Pass: the best arm gains **at least +0.10 against DanLM over m1** (a third of
the search gain), with no loss beyond −0.05 against m1 internally or against
V1T, and C within ±0.05 of m1. If every arm is within noise, offline
distillation does not carry the gain. Then look at label quality first (the
hidden-hand sampler, the margin) before building stage 2.

## Stage 2: online, only if stage 1 passes

A branch from the latest main milestone. CPU workers produce labels from the
branch's latest snapshot, refreshed on a fixed update cadence, into a rolling
buffer. The learner adds `lambda · CE` on a buffer minibatch each update,
next to PPO, the same way the auxiliary heads enter the loss
(`train/history_ppo.py`, `policy_total`). Compare the branch with the main
continuation at equal decisions over two milestones (500M each).

GPUs: the per-user cap is 4. A first 2-GPU slice (job 4611, Oct 6) ran at
about 14,500 decisions/s against about 18,400 on 4 GPUs. That was measured
over only 52 updates; a 124-update window is needed. If it holds, a 2+2 split
costs the main run about 20% of its speed and gives the branch about 80%.
Labels are not the bottleneck at this scale: one update is 65,536
decisions, and the buffer is reused across updates.

## Risks

- **Strategy fusion.** The search averages rollouts over sampled hidden hands
  and can prefer moves that look good world by world but are not a single
  consistent choice. Distillation would copy that. The DanLM, internal and
  V1T checks are there to catch it.
- **Label noise.** With 128 worlds the search overrode half as often (404 vs
  784) for the same gain, so many 32-world overrides look like noise that is
  neutral on average. Arm S and the margin are the levers.
- **Entropy collapse** from hard labels: the anchor term and the entropy check.
- **Off-policy positions.** Labels come from m1's positions. Fine for one
  improvement step; stage 2 refreshes them.

## Timeline and cost

| When | What | Where |
|---|---|---|
| Day 0 | label collector (`eval/collect_search_labels.py`), replay loader, fine-tune script, tests (replayed blueprint probabilities match the recorded ones) | Mac |
| Night 1 | labels, about 200,000 | Mac, plus server CPUs if useful |
| Day 1 | three fine-tunes, then evaluations (about 1 hour) | Mac and server |
| Decision | stage 2 or stop | user |

Money: none (Mac and Max-Velocity).
