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
gd.VecEnv(num_envs, num_threads, rules, actions, seed)
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
env.step(choices)             # np.ndarray [B] int32, one index per pending row
env.drain_finished_rounds()   # list[RoundResult]
env.fork(env_id, copies)      # list[int] of new env ids
```

Buffers returned by `pending()` stay valid until the next `pending()` or `step()`.
