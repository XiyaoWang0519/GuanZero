# Round and match state machine: the implementation contract

Companion to `cpp/include/gd/state.h` and RULES.md sections 7, 8 and 9.

## Phases

`Deal -> Tribute -> BackTribute -> Play -> RoundEnd`, then either the next
round or `MatchEnd`. Round 1 of a match skips both tribute phases, and so does
an anti-tribute round.

`needs_decision` is true in `Tribute`, `BackTribute` and `Play`. The engine
resolves forced moves itself: when a seat's only legal play is pass, it passes
and records it without the caller being asked. `legal_actions` therefore never
returns a single-element pass list during `Play`.

## Playing a trick (RULES.md 7.2)

`top` of type `Pass` means the trick is open and `to_move` must play.
Otherwise the seat may pass or beat `top`.

A play sets `top`, `holder` and resets `passes` to 0. A pass increments
`passes`. After a play that empties a hand, append the seat to `order`, set
`finish_pos`, and test the round-end conditions of 7.3 **before** advancing.

The trick ends when `passes` equals the number of active seats other than
`holder`, counting `holder` only if it is still active. Then:

1. `holder` still active: `holder` leads.
2. `holder` finished with that play: `partner(holder)` leads, the teammate
   lead. The partner is always active at that point.
3. Defensive fallback: the next active seat after `holder`.

Clear `top`, `holder` and `passes` at every trick start. `to_move` always
advances to the next **active** seat.

## End of a round (RULES.md 7.3)

The round ends as soon as both members of one team have finished, or three
players have finished. Seats that never ran out keep their cards; their
relative order in `order` follows `double_win_tail`:

* `NextFollower` (house): by seat, starting from `next(Follower)`.
* `AscendingSeat` (ogd): by ascending seat index.

No rule depends on it, but the two profiles must log it identically to their
references or the replay diff reports phantom divergences.

## Levels, ownership and level A (RULES.md 8)

`end_of_round` is normative and mirrors `gd_reference.end_of_round`, extended
by two config fields:

* `a_fail_on_loss` (house true, ogd false): whether a *lost* owned round at
  level A counts as a failure. Both profiles count an owned A round won with
  Banker and Dweller.
* `a_fail_limit` (house 3, ogd 4): the reset to `a_fail_reset_level` fires when
  the count reaches it. The simulator increments first and compares with `> 3`,
  which is why its effective limit is 4.

Order of operations, which the oracle pins: decide the match win, then count
the failure, then promote the round winner, then apply the reset.

`seat_return` is the DMC target of DESIGN.md 8.2: `+3`, `+2` or `+1` for both
seats of the Banker's team by the partner's finishing position and the negative
for the other team, except that in a level-A round owned by the Banker's team
and won with Banker and Dweller every seat scores 0, because that result cannot
pass A. The four values always sum to zero.

## Tribute (RULES.md 9)

Let `B, F, T, D` be the previous round's finishing order.

**Single tribute.** If `D` holds both red jokers: anti-tribute, no cards move,
`B` leads. Otherwise `D` gives one card to `B`, `B` returns one to `D`, and `D`
leads. This holds even when `D` is `B`'s partner (`tribute_between_partners`).

**Double tribute.** If the two losers hold both red jokers between them, in any
split: anti-tribute and `B` leads. Otherwise both losers choose a tribute card,
then both receivers return one.

Who receives which card depends on `tribute_pairing`:

* `Power` (house): the higher card goes to `B`, the other to `F`.
* `SeatGeometry` (ogd): the loser at `(B + 1) % 4` always pays `B` and the
  loser at `(B + 3) % 4` always pays `F`, whatever the cards are.

The payer of the higher card leads under both profiles. On a tie the leader is
the Banker's upstream seat `(B + 3) % 4` under `Upstream`, or the seat recorded
last in the previous finishing order under `LastFinisher`.

Decision order inside the phase, so that the batch interface stays uniform:
both tribute decisions first, in payer order, then both back-tribute decisions
in receiver order. Record every movement in `tribute_moves`, which is public
and feeds the observation's tribute block and the known-holdings block.

Hands are back to 27 cards each when `Play` begins.

## DealSpec

`set_deal` accepts four hands, the round level, both team levels, the fail
counts and the owner. With `leader` set the round opens directly in `Play`.
With `prev_order` set it opens in `Tribute`, which is what the tribute
experiments of DESIGN.md 8.3 and the duplicate-deal evaluator need. Setting
both is an error; setting neither means a random leader from the seed.

## Hashing and serialization

`hash` must cover everything that affects future play: all four hands, the
played sets, the level, phase, seat to move, top play, holder, passes,
finishing order, tribute state and the match-level levels, fails and owner. It
must not depend on padding, on pointer values or on the machine, because
invariant 8 requires the same seed and action sequence to hash identically
everywhere. Serialization is a byte copy of the trivially copyable state plus a
version tag, and must round trip the hash.
