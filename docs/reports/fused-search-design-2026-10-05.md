# Fused test-time search: design (October 5, 2026)

Goal: make one search several times cheaper without losing its gain, so that
search can later be distilled back into the policy. The design combines our
current search with two ideas from the literature: GS2's value-diverse choice
of hidden worlds (Ge et al., NeurIPS 2023) and depth-limited rollouts with a
learned value at the leaf (GS2; Ataraxos, Nature 2026). It adds a third idea
of our own: the privileged critic as a control variate.

Status: design only. Nothing here is implemented or measured yet.

## 1. Where we start

Current search (`eval/search.py`, config of every published search number):
trigger `hand_or_unsure` (any seat at 10 cards or fewer, or the blueprint's
own choice below probability 0.6); root candidates are the blueprint's top 8;
32 hidden worlds dealt uniformly; every (world, candidate) pair rolled out to
the end of the round with the greedy blueprint in all four seats; the mean
return per candidate decides, with a 0.5 margin before overriding the
blueprint; 10 s budget.

Measured against the strongest published agent (paired deals):

| Blueprint | Decisions | Gain from search | Override rate | Searches per decision |
|---|---|---|---|---|
| u2623 | 172M | +0.43 ± 0.16 (200 deals) | 14.0% | 44% |
| u9989 | 655M | +0.31 ± 0.15 (300 deals) | 13.4% | 44% |
| u15094 | 989M | +0.30 ± 0.14 (300 deals) | 11.6% | 45% |

Cost: median 2.3 s per search, p95 8.8 s, 2-3% of searches hit the 10 s
budget; about 120 s per deal with 3 workers on the Mac. The search gain is
worth roughly 450M training decisions on the current curve, but at this cost
it cannot feed training.

## 2. What the literature adds

* GS2 (the strongest published Guandan search agent; search at play time
  only). Samples m = 1,000 hidden worlds, keeps k = 10-30 chosen to span the
  range of counterfactual values, builds a depth-10 tree (8 when opponents
  hold more than 27 cards) over the top 2 actions per hand type, values the
  leaves by greedy rollouts of its DMC Q function, and solves the small tree.
  Its ablation is the key fact for us: with k = 10 worlds picked at random
  the search gains nothing (-0.014 ± 0.18 vs 0.000 ± 0.16); picked for
  diversity, it gains.
* Ataraxos (Stratego): depth-limited rollouts, value network at the leaf,
  search regularized toward the policy (an unregularized search step fell far
  below the raw network). Search only at test time; search inside training
  was tried and the data-generation slowdown outweighed the benefit.
* Successful search-in-training (AlphaZero; DORA and Diplodocus in 7-player
  Diplomacy) all use cheap, shallow search with a value network. Failures
  (DeltaDou in DouDizhu, Stratego) used expensive rollouts. Cost per search
  target is the deciding factor.

## 3. Why GS2's selection does not transfer as is

GS2 solves a small game tree, where covering the spread of opponent
situations matters. Our search compares candidates inside the same worlds:
the value of a world is common to all candidates and cancels in the
comparison. Stratifying on the world's value V(w) therefore removes little
of the variance that matters. What matters is the spread of the candidate
differences D_a(w) = R_a(w) - R_0(w) across worlds (R_a: rollout return of
candidate a; candidate 0 is the blueprint's choice). Selection and variance
reduction should key on a cheap proxy of D_a(w), not on V(w).

## 4. The fused search

Same trigger, candidates, margin and budget as now. Per search:

1. Deal M uniform worlds (M = 256 to start). Dealing is cheap (C++).
2. Cheap proxy. For every world w and candidate a, apply a and evaluate the
   afterstate with the critic: C_a(w) = V(afterstate), always from the root
   team's point of view (negate when the seat to move belongs to the other
   team). The critic is the trainer's `HistoryCritic`: an MLP over the
   acting seat's observation plus the three hidden hands' card counts, which
   the world supplies. M x 8 rows in one batched forward pass, no history
   encoder. Proxy difference c_a(w) = C_a(w) - C_0(w).
3. Expensive evaluation on k of the M worlds (k = 8 to start): full greedy
   rollouts as now, giving D_a(w) for those k worlds.
4. Estimate per candidate, three variants to compare:
   * **uniform-k**: plain mean of D_a over k randomly chosen worlds (what we
     get by simply cutting worlds; the baseline every variant must beat).
   * **diverse-k** (GS2-style): choose the k worlds to span the proxy
     differences (sort worlds by their proxy vector's first principal
     component, cut into k equal strata, draw one world per stratum at
     random). Equal strata keep the plain mean unbiased.
   * **control variate**: mean over M of c_a(w), plus the mean over the k
     rolled-out worlds of (D_a(w) - c_a(w)). Unbiased for the uniform-world
     mean whatever the critic's error; the variance falls with the
     correlation between c and D. Combinable with diverse-k.
5. Decide as now (mean, 0.5 margin). Record the per-candidate estimates and
   their standard errors for every search: these are the soft targets a later
   distillation step would train on.

Depth-limited rollouts (stage E2): stop rollouts after d plies (d = 10, as
GS2) and score the leaf with the critic, except when few cards remain, where
full rollouts are short anyway (rule: full rollout when the round would end
within d plies on the blueprint's path, or when every seat holds at most 10
cards). This cuts rollout cost further, but unlike the control variate it is
biased by the critic's error.

## 5. Experiments

Stage E0, offline truth set (Mac, one night). Run plain self-play and the
reference-agent arena with the current search trigger; at about 2,000
triggered decisions save the serialized state, the public history and the
candidate list. For each saved state compute a high-fidelity reference: 256
uniform worlds with full rollouts. These states are the test bed for every
variant, so variants are compared on identical inputs.

Stage E1, estimator quality (Mac, CPU/MPS, hours). For each variant at k = 4,
8, 16 and M = 256: on every saved state, (a) does it pick the same move as
the reference, (b) mean squared error of its candidate differences against
the reference, (c) wall time. Paired across variants on the same states, this
is far more sensitive than game outcomes. Success: some variant at k = 8
matches 32 uniform worlds in agreement with the reference at a quarter of the
rollouts or less.

Stage E2, depth limit. Same protocol with d = 6, 10, 16 on top of the best E1
variant. Success: agreement within a few points of the full-rollout variant,
at a further 2x or better saving.

Stage E3, games. The winning configuration against the strongest published
agent on the same 300 deals as the existing search numbers, paired against
both no search and the current search. Success: gain not distinguishable
from the current +0.30 at a third of the time or less.

Stage E4, distillation pilot (only after E3). Collect soft targets on
triggered decisions from self-play, fine-tune the policy with an added
cross-entropy toward the search's preferences (only where the search's
standard errors separate candidates), and measure the plain policy against
the reference agent. Needs a GPU for any meaningful number of targets.

## 6. Risks and checks

* Critic error. The critic's residual is about 31% its own error and 69%
  action-sampling noise (critic noise split, October 3). The control variate
  stays unbiased under critic error; diverse-k and the depth limit do not.
  E1 measures the correlation between c and D directly; if it is near zero,
  the control variate and diverse-k collapse to uniform-k and the plan stops
  at E1.
* The critic was trained on the lineage's own self-play states. Searched
  worlds are uniform deals, so some are unlikely under real play; the critic
  may be off there. Check the c-D correlation separately for late and
  "unsure" searches.
* Perspective bugs. Afterstate values must be taken for the root team; a
  sign error would silently invert the proxy. Unit test on a constructed
  state where one candidate wins the round outright.
* Comparability. Published search numbers use 32 uniform worlds and a 10 s
  budget; every new variant keeps the same trigger, candidates, margin and
  budget, and is reported next to the current search on the same deals.
* Opponent-model mismatch. Rollouts assume all seats play like the
  blueprint. This limits search against other agents whatever the estimator
  (in self-play the gain grew +0.5 to +0.75 with model strength; against the
  reference agent it did not grow). Opponent modelling is a separate
  question; this design does not address it.

## 7. Code touch points

* `eval/search.py`: new `SearchConfig` fields (`estimator`:
  uniform | diverse | control, `proxy_worlds` M, `rollout_worlds` k,
  `depth_limit` d); the defaults reproduce the current search exactly.
* `eval/search_rollout.py`: depth-limited rollout and leaf evaluation.
* New loader for the critic next to `load_policy` (reuse
  `load_history_checkpoint` as in `eval/critic_noise.py`).
* New `eval/search_truth.py`: state collection (E0) and the offline estimator
  comparison (E1, E2).
