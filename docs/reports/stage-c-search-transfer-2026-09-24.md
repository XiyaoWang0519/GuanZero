# Residual-shuffle search: scoring audit and DanLM transfer probe

Date: September 24, 2026. This evaluates the fixed `e8fc714` search v0
configuration; it is not the learned-belief C3 arm or G9 acceptance test.

## Audit of the internal paired result

The upstream development file `search-1000.json` (seed 2026092427) and
independent file `search-4000-heldout.json` (seed 2026092463) were read from
`.work/overnight-20260924/`. The held-out file has SHA-256
`f60361195847a7af0f3661d776829d82254cb1a57a56dc3aa0afa13a5d003926`.
For all 5,000 deal pairs, recomputing
`(first.seat_return[0] - swapped.seat_return[0]) / 2` from the raw legs exactly
matches each stored pair score. The means reproduce +0.042 and +0.049
levels/round respectively. The 4,000-pair file has 326 positive, 102 negative
and 3,572 zero differences. Repeating its 500-replicate, whole-deal bootstrap
with the recorded seed exactly reproduces [0.0381125, 0.06014375]. Its
reported extra-search p95 latency (30.41 ms) is only an aggregate in that
file, so it cannot be independently recomputed from per-decision times.

The scalar duplicate harness resets the same deal for both legs and seeds each
seat identically in those legs. The B11 blueprint is deterministic argmax.
Search draws its candidate subset and hidden worlds from the evaluation RNG,
without using the true hidden hands or state hash as a seed. The C++ sampler
uses own cards, public played cards, public seat sizes and public tribute facts;
the wrapper accepts only audited built-in blueprints. Existing tests also
compare sampled worlds and actions when true hidden hands change but the acting
seat's information stays fixed. This is an information-boundary and scoring
audit, not independent reproduction of all 4,000 games.

## External DanLM transfer, same deals for both arms

`eval.search_danlm_transfer` ran 500 new duplicate deals, seed 2026092491,
with 50% tribute setup. Each arm played both seat assignments against the
frozen DanLM v1 checkpoint. DanLM's engine was the referee. Both arms used
the same frozen B11 checkpoint (model digest
`8e872130189af48adb262a0b218bcc82e711b31f5804272c1dd50854b629bc3e`),
one Python worker and one PyTorch CPU thread. Search retained the default
12-unseen-card trigger, at most four actions and four sampled worlds, 120-step
rollout cap and 500 ms deadline. There was no training or cloud usage.

| Arm | Mean net levels/round | Bootstrap 95% interval |
|---|---:|---:|
| B11 | −2.017 | [−2.107, −1.928] |
| Search:B11 | −2.001 | [−2.094, −1.909] |
| Paired search minus B11 | +0.016 | [0, +0.035] |

All 500 deals (1,000 legs) scored in both arms; neither arm had a mirror
failure. In the direct paired difference, 11 deals improved, 1 worsened, and
488 tied. Both arms logged the same 26 `our_candidate_unmatched` and 3
`their_choice_reading` divergences, with no reward mismatch. The raw records
and summary are `search-danlm-transfer-500.raw.json.gz` (SHA-256
`fa47714ab8ec67a052617ac6aa368b70870f82abcd20fda051f9b994b63220f0`)
and `search-danlm-transfer-500.summary.json` beside this report.

The confidence interval for the transfer difference touches zero. The
observed advantage over B11 in internal self-play therefore does not yet
establish a reliable improvement against DanLM, and both policies lose about
two net levels per round to DanLM. Search adds inference work; this is a
bounded deployment comparison, not an equal-compute training comparison.
The transfer raw records omit per-decision search latency and fallback counts;
those are known only from the separate internal run. The residual shuffle is
also not a calibrated hidden-hand posterior, and the action correction uses
an artificial prior on the blueprint's chosen action.

## Reproduction and boundaries

The transfer harness is `eval/search_danlm_transfer.py`. From this worktree,
after building its CPython 3.12 `gd` extension in `build-cp312`:

```sh
PYTHONPATH=python:. DANLM_ROOT=/Users/xiyaowang/Documents/Projects/GuanZero/.work/external/DanLM \
  /Users/xiyaowang/Documents/Projects/GuanZero/.work/external/danlm-venv/bin/python \
  -m eval.search_danlm_transfer \
  --checkpoint /Users/xiyaowang/Documents/Projects/GuanZero/.work/runpod-b11/results/runs/main/run/latest.pt \
  --deals 500 --seed 2026092491 --chunk 100 --max-seconds 1400 \
  --output-prefix docs/reports/search-danlm-transfer-500
```

The harness writes raw legs after each completed chunk and bootstraps only
whole deals whose two legs succeeded in both arms. This run completed in
337.7 seconds. The CPython 3.12 extension was built only in this worktree;
the main checkout's compiled module was untouched.

## Independent 4,000-deal transfer confirmation

After the 500-deal pilot, the unchanged search configuration was evaluated on
new seed 2026092497 with 4,000 paired deals. Both policies used the same B11
checkpoint. The run completed in 2,649.46 seconds on one CPU inference thread.
Two deals were excluded because of mirror failures; 3,998 common valid deals
remain. Their raw rewards were independently joined and rescored after the run.

| Arm | Mean net levels/round | Bootstrap 95% interval |
|---|---:|---:|
| B11 | −2.04077 | [−2.07166, −2.00849] |
| Search:B11 | −2.02489 | [−2.05578, −1.99236] |
| Paired search minus B11 | +0.01588 | [+0.00925, +0.02251] |

There were 99 improved deals, 36 worse deals, and 3,863 ties. This confirms a
small transfer gain against this frozen DanLM opponent under the tested
protocol, while leaving the large absolute gap to DanLM. It does not establish
full-match strength, unfamiliar-partner performance, or G9. No search latency
claim for this external matchup is inferred from the internal timing sample.
The final raw legs and summary are retained alongside this report as
`search-danlm-transfer-4000.raw.json.gz` and
`search-danlm-transfer-4000.summary.json`.
