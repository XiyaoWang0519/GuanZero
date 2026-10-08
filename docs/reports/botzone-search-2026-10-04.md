# Botzone search deployment — October 4, 2026

## Model and configuration

- Main-line checkpoint: u20264, `.work/lr-decay-2026-10-03/kits/main/download/results/segments/lr-main/latest.pt`.
- Actor checkpoint ID: `3191f5826b2d6453c2671337ed074067e40b2e08531210737552e82cfc2c6367`.
- Export: `.work/botzone/dist/gz_actor_u20264.npz`, 5,457,722 bytes; uploaded to separate Botzone user-storage filename. The u15094 weights remain available for old versions.
- Platform submission page confirms Python 3.6.5: **6 seconds per turn** in traditional mode, **4 seconds** after startup in keep-running mode. Uses traditional mode. Code limit 4 MB; models are in user storage.
- Budget: 6 seconds minus 0.4 seconds reserve and process CPU time already consumed by interpreter/import/model load/input. The remaining duration bounds the decision, including mirror reconstruction and root scoring. Search uses a wall-clock deadline, conservatively counting scheduling delays. This is a bounded budget, not a guarantee against all platform stalls.
- Search trigger: any seat has <=10 cards, or root probability <0.6; top eight expressible policy candidates; rollout with the same actor to round end; minimum 16 complete paired worlds; switch only for >=0.5 levels improvement. Continue sampling until deadline, with a defensive cap of one million worlds.
- Pure NumPy incremental causal KV cache, branch-local arrays; no PyTorch or compiled extension on Botzone. Prefix includes public actions only. Hidden worlds are reconstructed from the acting seat's Botzone log, respecting public tribute/anti-tribute constraints. This constrained sampler differs from the offline residual-uniform shuffle and is not claimed to be a uniform posterior. Partial worlds are discarded.
- Search errors preserve the root policy action. Debug contains checkpoint prefix, world count, candidates, means, chosen/baseline indices, time and memory.

## Verification

- 20 Botzone tests passed, including incremental/full encoder parity, cache isolation, partial-world cancellation, expired-budget fallback, protocol and pure-Python engine checks.
- u20264 no-search parity judge: 30 games, 2,748 decisions, 1,090/1,090 comparable plays agree with PyTorch, no illegal responses or fallback notes. `.work/botzone/judge-u20264-parity.json`.
- Full-search smoke game against greedy: 74 decisions, 29 searches, 18–332 complete worlds per search, 2 overrides, no errors/fallbacks, max decision 5,300.8 ms with a 5.3-second local budget. This run had min_worlds=2 during development, but every actual search completed >=18; the released threshold is 16. `.work/botzone/search-smoke.json`.
- Final ZIP under existing `gz-botzone-py36:latest` Docker image, network disabled, one CPU, 256 MB: checkpoint prefix confirmed, 98 worlds, 5,485.6 ms decision, VmHWM 40 MB / VmPeak 99 MB on a late-round fixture.
- Plain Python source form in same container: 101 worlds, 5,480.7 ms, VmHWM 39 MB / VmPeak 98 MB. Package embeds compressed module source, not model weights.
- Neither smoke nor parity games establish a playing-strength gain for u20264 search.

## Deployment

The original bot was GuanZero v2, u15094 without search. Initial observed rank score was 1011.08; it changed to **1016.67 before this deployment** on v2 matches, so that movement is not evidence for the upgrade.

Two ZIP submissions displayed `服务器或网络错误：undefined` but were later found to have created versions 3 and 4. The site's online editor then accepted the verified plain Python source as **v5**, with explicit `创建成功`. Latest version and bot description both show u20264 search. No ladder exit/re-entry was performed (which would reset score).

Source artifact actually submitted: `.work/botzone/dist/gz_bot_u20264_search.py`.
Reproducible equivalent packaging:

```sh
PYTHONPATH=python:oracle:. .work/external/danlm-venv/bin/python -m eval.botzone.pack \
  .work/botzone/u20264.npz .work/botzone/dist/gz_bot_u20264_search.py \
  --single-file --search --weights-name gz_actor_u20264.npz
```

Online game/runtime and post-upgrade ranking observations are recorded below when available.

Latest observed leaderboard: **rank 42, score 1016.67, version 5**. Screenshot: `.work/botzone/deployed-v5-rank42.png`. This score still matches the pre-deployment v2 score; no gain attributable to v5 yet.

## Online validation and first ladder result

- Authorized test match: https://botzone.org.cn/match/6ac2c971e2453c4f471ec83b . Seats 0/2 used v5; seats 1/3 used v2. Completed with v5 team winning one point; no platform timeout/runtime-error termination was shown. Single-game result is not strength evidence.
- Visible v5 debug output confirms checkpoint `3191f5826b2d`, 117 complete worlds, 2 candidates, decision 5453.4 ms / search 5435.6 ms, resident high-water mark 38 MB. VmPeak reports 276 MB virtual address space; this is not resident memory. Saved `.work/botzone/online-v5-debug.txt`. This sampled debug entry does not establish the maximum runtime over every move.
- Refreshed ladder: **rank 42, score 1023.58**, up **6.91** from the immediate pre-deployment score 1016.67. Bot management shows the latest paired ladder games used **GuanZero v5** (site-displayed time 2026-10-5 5:45:03), one win and one loss, with aggregate +6.91. Rank has not moved yet; this small sample cannot isolate search benefit from checkpoint improvement or matchup variation.
- Final targeted regression: 20 passed in 10.36 seconds; `git diff --check` passed.

## Online observation, October 5, 2026

Score reached **1049.80, rank 34**. The 20 latest rated games (the platform's maximum list) were downloaded to `.work/botzone/v5_logs/` and audited with `.work/botzone/audit_logs.py`.

- 6 of the 20 games (Oct 4 17:16–17:27) still ran the old plain u15094 code (no checkpoint tag, ~200 ms decisions): 2–4. The other 14 (from 17:45) ran v5: **11–3, +22 points**. One-sided binomial vs 50%: p = 0.029; one-sided Fisher vs all 26 plain games (11–15): p = 0.029. Every game is a single deal at level 2 without tribute; partners and opponents are random ladder bots, so this is weak evidence.
- Paired seat-swapped games under v5: 4 pairs won both, 3 split, none lost both.
- Health: 331 v5 decisions, 0 non-OK verdicts, 0 fallback notes, platform time max 5,616 ms (limit 6,000).
- Search did little: 58 decisions skipped (confident early), 176 ran, but platform CPU gave a median of only **20 worlds** (10th percentile 5, local Docker gave ~100). 70 of 176 searches stayed below the 16-world minimum and kept the policy action. Only **5 of 331 decisions** were overridden.
- So the 11–3 is mostly u20264's own policy (it replaced u15094 in the same upload) or luck, not search. Search on Botzone is compute-starved; a cheaper search (fewer candidates, especially at 5–8 candidates where medians are 5–11 worlds) would be needed for it to matter there.

## Upgrade to m2 u35588, October 6, 2026

- Checkpoint: Velocity milestone m2, `.work/velocity-milestones/m2-u35588.pt` (2.332B decisions; −0.869 vs DanLM plain on 4,000 deals). Actor ID `1705fd905f860b08caef46987c231ba360849a9d9145ad9d3ec7b473465d8f91`.
- Export `.work/botzone/u35588.npz` (5,457,558 bytes), uploaded to Botzone storage as `data/gz_actor_u35588.npz` (size matches). Older weight files kept.
- Parity judge (30 games vs greedy): 2,572 decisions, 1,071/1,071 comparable plays agree with PyTorch, 0 illegal, max 20 ms. `.work/botzone/judge-u35588-parity.json`.
- Package: same search code and settings as v5 (`pack --single-file --search --weights-name gz_actor_u35588.npz`) → `.work/botzone/dist/gz_bot_u35588_search.py`. Docker py3.6 smoke (1 CPU, 256 MB, no network): checkpoint `1705fd905f86`, 107 worlds, 5,488.6 ms, VmHWM 39 MB.
- Submitted as **GuanZero v6** by file upload (Python 3.6.5), "Successfully created"; bot description updated. Score at switch **1036.89**. No ladder exit/re-entry.
- Score reset (user request, October 6): the 20 latest rated games (all v5; 2026-10-6 14:45 to 2026-10-7 0:38 site time) were saved to `.work/botzone/v5_logs_oct6/` first. Then opted out of the ranklist and back in (star button; both steps use a native confirm dialog the user clicked). Now **v6, score 1000.00, on the ranklist**. Ratings from here reflect v6 games only.
- October 8, 2026 (user-reported, from the Botzone bot page): v6 at **rank 33, score about 1077**, up from 1000.00 at the October 6 reset. The rating curve rose to about 1050 in the first half of the games and then moved between about 1048 and 1077. Only the rank and the rating curve were observed; there is no per-game audit and no search-on/off split, so this is not evidence of a search gain.
