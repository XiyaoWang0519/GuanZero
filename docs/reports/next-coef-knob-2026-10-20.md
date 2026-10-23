# Train-time `next_coef` knob (2026-10-20)

Status: **implemented, not yet trained.** Lands the single-knob next-token
coefficient proposed by
[aux-loss-balance-screen-2026-10-20](aux-loss-balance-screen-2026-10-20.md).
No GPU arm has run; holdout eval is deferred.

## Change

Belief and outcome keep shared `aux_coef` (default 0.1, with existing warmup).
Next uses:

| field | role |
|---|---|
| `next_coef` | start weight; `None` → same as warmed `aux_coef` (bit-identical default) |
| `next_coef_final` | anneal end (default 0) |
| `next_anneal_updates` | linear anneal length; 0 = off; requires explicit `next_coef` |

Actor loss applies `next_coefficient * next_loss + aux_coefficient * (belief + outcome)`.
Resume overrides: `--resume-set next_coef=…,next_coef_final=…,next_anneal_updates=…`.

## Planned Velocity arms (short, mature resume)

Fixed healthy resume (m10 / tip once chosen); `aux_coef=0.1`; one variable = next:

1. `next_coef=0.1` (control)
2. `next_coef=0.03`
3. anneal `0.1 → 0` over ~1k updates
4. `next_coef=0` (off; heads remain)

Log learner rows / decisions / GPU-hours. Holdout eval later. **Do not take the
only free GPU ahead of Branch A (mature league).**

## Checks

- `tests/test_history_aux.py` — default equality with `aux_coef`, anneal path,
  resume-set parse, `next_coef=0` with belief/outcome still at 0.1
