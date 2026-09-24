# C3 residual-shuffle endgame search v0 — bounded CPU probe

Date: September 24, 2026. This is an engineering prototype and a small paired
inference probe. It is **not** the C3 learned-belief arm or the G9 acceptance
test in `STAGE_C_TODO.md`.

## Implementation and information boundary

`MatchState.determinize_uniform(observer, seed)` clones a Play state with the
other three hands resampled. It first fixes public known-card lower bounds,
then uniformly shuffles residual card copies into open seats. This does not
imply a uniform distribution over complete legal hidden-hand assignments or
a calibrated conditional belief. The C++ sampler reads only the observer's hand,
publicly played cards, each seat's public remaining count, and public tribute
and anti-tribute records. It fixes known cards, shuffles the remaining unseen
copies, and rejects inconsistent card multiplicities or seat sizes. It never
reads another seat's true card identities. Search rollouts run `Engine.apply`
on these sampled states and call the frozen blueprint for each acting seat;
that blueprint obtains that seat's own observation and legal actions. The
search wrapper only accepts the audited built-in policy classes, preventing
an arbitrary policy from inspecting the live state's other hands.

`SearchPolicy` triggers only in Play when unseen cards are at most 12. Its
defaults consider the blueprint action and at most three alternatives over at
most four complete sampled worlds, with a 120-step rollout cap and a 500 ms
search deadline. A partial world is discarded. If fewer than two worlds finish,
the original blueprint action is returned. Completed values use the round's
seat return and a hand-set 70% prior on the blueprint's chosen action, with
the rest spread evenly over sampled alternatives. The tabular correction is
KL-regularized relative to this artificial prior, not the blueprint's actual
policy distribution. An alternative also needs a 0.5-return margin. The scalar arena accepts
`search:<checkpoint>`. The existing batched evaluator still accepts only its
previous policy classes; this search prototype uses `--eval-backend scalar`.

## Local probe

Frozen B11 checkpoint SHA-256 model digest:
`8e872130189af48adb262a0b218bcc82e711b31f5804272c1dd50854b629bc3e`.
Two PyTorch CPU threads, 64 paired duplicate deals, seed 20260924, house rules,
same frozen B11 checkpoint as the opponent. No training or cloud resources.
Timing below is additional search time after the blueprint root inference.

| Search deadline | Triggers | Complete | Fallback | Overrides | p50 / p95 / max extra ms | Net levels/round [bootstrap 95%] |
|---|---:|---:|---:|---:|---:|---:|
| 500 ms | 665 | 665 | 0 | 10 | 11.15 / 35.68 / 56.26 | +0.0078 [−0.0432, +0.0547] |
| 20 ms | 665 | 649 | 16 | 17 | 10.62 / 20.13 / 20.44 | +0.0625 [0, +0.1406] |

The intervals include zero; these small numbers do not establish strength.
The 20 ms and 500 ms runs produce different stochastic rollout outcomes, so
their score difference is not a reliable budget ranking. Raw paired results
and timings: `stage-c-search-v0-500ms.json` and
`stage-c-search-v0-20ms.json`. A two-deal scalar `eval.arena` smoke report is
`stage-c-search-v0-arena-smoke.json`.

Reproduce from this checkout after building its Python module:

```sh
PYTHONPATH=python:oracle:. /Users/xiyaowang/Documents/Projects/GuanZero/.venv/bin/python -m eval.search \
  --checkpoint /Users/xiyaowang/Documents/Projects/GuanZero/.work/runpod-b11/results/runs/main/run/latest.pt \
  --deals 64 --seed 20260924 --time-ms 500 --threads 2 \
  --output docs/reports/stage-c-search-v0-500ms.json
```

Run this from the search worktree after building its Python module. The
residual-shuffle sampler is a control, not the marginal or learned belief
requested by C3. Candidate selection is
bounded and does not evaluate the full legal set. A G9 claim requires the
learned-belief arm, three measured budgets, and the prescribed larger
duplicate/held-out comparison.

## Verification

- `build-search/cpp/tests/gd_tests`: 81 cases, 0 failures.
- Targeted Python suites (`test_search`, Stage B baseline, tribute evaluation,
  engine invariants): 47 passed.
- Same observable information with different true hidden hands gives the same
  sampled worlds for each seed and the same search action. Separate tests
  verify two-deck conservation, known tribute cards, anti-tribute red joker
  holders, non-anti private flags, clone replay, finite configuration and budget
  fallback.
