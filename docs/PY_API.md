# Python API of the `gd` package

The pybind11 module is `gd._gd_core`; `python/gd/__init__.py` re-exports it.
The suites in `tests/` are the specification: whatever makes them pass is right.

## Oracle-shaped free functions

These mirror `oracle/gd_reference.py` exactly in argument and return shape so
that the cross-check tests can compare with `==`. Types are the oracle's
strings: `Single Pair Triple FullHouse Straight Tube Plate Bomb StraightFlush
JokerBomb Pass`. A key is an `int`, except for `Bomb` where it is the tuple
`(size, power)`.

```python
gd.card_id("HT") -> int                      # -1 when unparsable
gd.card_str(8) -> "ST"
gd.cards("S3 D3 H7") -> list[int]
gd.power(rank: int, level: int) -> int
gd.interpret(cards: Sequence[int], level: int) -> set[tuple[str, int | tuple]]
gd.best_readings(cards: Sequence[int], level: int) -> set[tuple[str, int | tuple]]
gd.beats(cand: tuple[str, key], top: tuple[str, key] | None) -> bool
gd.legal_actions(hand: Sequence[int], level: int,
                 top: tuple[str, key] | None,
                 canonical: bool = False) -> set[tuple[str, key, tuple[int, ...]]]
gd.abstract_id(action_or_reading) -> int
gd.abstract_action_count() -> int            # 393
gd.tribute_choices(hand, level) -> set[int]
gd.back_tribute_choices(hand, level) -> set[int]
```

## Object API used by training

```python
gd.RuleConfig()          # fields mirror cpp/include/gd/config.h; .house() / .ogd()
gd.ActionConfig()        # .full() classmethod
gd.Action                # .type (str) .key .bomb_size .wilds .cards (list[int]) .abstract_id
gd.DealSpec              # .hands (4 lists) .level .team_levels .fails .owner .leader .prev_order
gd.Engine(rules, actions)
gd.MatchState            # .levels .fails .owner .round_index .winner .phase .to_move .hands .hash()
gd.VecEnv(num_envs, num_threads=1, seed=0, rules=gd.RuleConfig.house(),
          actions=gd.ActionConfig(), encode=True,
          log_public_actions=False, log_env_limit=-1)
```

`VecEnv` follows DESIGN.md 5.3:

```python
env = gd.VecEnv(num_envs=64, num_threads=4, seed=1)
env.reset()                   # or env.reset(deals) with a list[DealSpec]
b = env.pending()             # DecisionBatch
b.obs        -> np.ndarray [B, gd.OBS_DIM]  float32, aliases engine memory
b.cand       -> np.ndarray [sum_k, gd.ACT_DIM] float32
b.offsets    -> np.ndarray [B + 1] int32
b.env_id, b.seat, b.phase -> np.ndarray [B] int32
b.round_index, b.greedy_choice -> np.ndarray [B] int32
b.match_id   -> np.ndarray [B] int64
b.hidden_counts -> np.ndarray [B, 3, 54] uint8
env.step(choices)             # np.ndarray [B] int32, one index per pending row
env.drain_finished_rounds()   # list[RoundResult]
env.drain_public_actions()    # list[PublicActionEvent], only when logging enabled
env.fork(env_id, copies)      # list[int] of new env ids
```

`hidden_counts` contains the actual remaining hand counts of relative seats
`+1` (LHO), `+2` (partner), and `+3` (RHO), in card-id order. These are
**privileged training targets**, never policy inputs. `obs` remains limited to
the acting player's hand and public information. `greedy_choice` is a local
candidate index from the in-engine greedy player, including the tribute
heuristic. Reading it does not advance any environment RNG.

The observation's known-holdings block contains one public-presence bit per
card and other seat. It tracks lower bounds from tribute and back-tribute
transfers in order, then subtracts public plays. Returning a received card
removes its guarantee at the returning seat, even when an unobserved second
copy remains there. This correction preserves the observation layout and
dimensions; archived checkpoints load unchanged, but their inputs can differ
on these exchange cases when evaluated with the corrected encoder.

All batch arrays are read-only views backed by ordinary CPU memory. The batch
keeps its environment alive, but its views must be copied before the next
`pending()`, `step()`, `reset()`, or `fork()`. This memory is **not CUDA-pinned**;
CUDA learners explicitly copy into pinned staging tensors when using
asynchronous transfers. With `encode=False`, `obs` and `cand` have shapes
`[0, OBS_DIM]` and `[0, ACT_DIM]`; metadata and private labels are still present.

Call `reset()` before `pending()`, and call `pending()` before each `step()`.
Choices must be a contiguous, one-dimensional NumPy `int32` array of exactly
`B` valid local candidate indices. Invalid input raises an exception before
any environment advances. Reusing a consumed batch, malformed deal fields,
invalid environment/row indices, and negative fork counts are rejected.
`fork()` adds slots and invalidates the old pending batch; call `pending()` to
include the new slots before stepping again.

Round completion is processed by `pending()`. Drain results immediately after
that call, before assigning the new batch's decisions to trajectories:

```python
b = env.pending()
for result in env.drain_finished_rounds():
    key = (result.env_id, result.match_id, result.round_index)
    # Label all stored decisions for key with result.seat_return[acting_seat].
```

`RoundResult` exposes `env_id`, `match_id`, `round_index`, `order`,
`num_finished_seats`, `winning_team`, `gain`, `levels`, `fails`, `next_owner`,
`match_winner`, `round_level`, and `seat_return`. A standalone `Engine.end_round`
result has `env_id=-1`. `round_index` is zero based within a match and uses
32-bit storage, so matches longer than 127 rounds do not wrap. `match_id` is a
per-environment generation starting at zero and increments on automatic match
restart. Both identifiers reset on `reset()`; forks inherit the source match
identifier and use their new environment id. Raw `MatchState.serialize()`
blobs are build-local snapshots, not a portable or versioned checkpoint format.

For belief-probe datasets, use `log_public_actions=True`. Set `log_env_limit=4`
to record only environment ids 0 through 3; `-1` records all ids. Logging
preserves the same decisions and outcomes and includes automatically skipped
passes. Drain frequently to bound memory. Events are ordered within each
environment and contain:

| Field | Meaning |
|---|---|
| `env_id`, `match_id`, `round_index` | Source trajectory |
| `seat`, `phase` | Actor and phase before the action |
| `step` | Zero-based play step before the action; tribute actions do not increment it |
| `action` | Public concrete `Action`, including exact cards |
| `encoded_action` | Owned float32 `[ACT_DIM]` array; private tribute flags `[146:]` are always zero |
| `cards_left` | Actor's public remaining card count after the action |
| `forced` | True for an automatically skipped pass |

Tribute payments are settled together after all payers choose, so the first
payer's `cards_left` can still be 27 at its selection event. The play-phase
stream is the complete ordered play/pass history used by the v2 belief probe.
History contains no hidden hands; `hidden_counts` must be stored separately
as labels. `reset()` clears undrained results and events.
