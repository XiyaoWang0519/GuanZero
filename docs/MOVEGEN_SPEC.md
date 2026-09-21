# Move generation: the implementation contract

Companion to `cpp/include/gd/movegen.h` and RULES.md section 11. Written before
the code so that the two enumeration modes have one definition, not two.

## The required-rank multiset

Every abstract action names a multiset `R` of natural ranks it needs.

| Abstract action | `R` | Jokers |
|---|---|---|
| Single, power `p` | `{rank(p): 1}` | `p = 13` is `BJ`, `p = 14` is `RJ` |
| Pair, power `p` | `{rank(p): 2}` | a joker pair needs two natural jokers of one colour |
| Triple, power `p` | `{rank(p): 3}` | never |
| Bomb, size `n`, power `p` | `{rank(p): n}`, `4 <= n <= 10` | never |
| Joker bomb | `{BJ: 2, RJ: 2}` | all four, no substitution |
| Full house, triple `r`, pair `q` | `{r: 3, q: 2}`, `q != r` | `q` may be `BJ` or `RJ`, and then needs two natural jokers |
| Straight, window `w` | one of each of the window's 5 ranks | never |
| Straight flush, window `w`, suit `s` | the same 5 ranks, every card of suit `s` | never |
| Tube, window `w` | two of each of the window's 3 ranks | never |
| Plate, window `w` | three of each of the window's 2 ranks | never |

`rank(p)` inverts `power`: `p == 12` is the round level `L`, `p < 12` is `p` when
`p < L` and `p + 1` otherwise, `p == 13` is `BJ` and `p == 14` is `RJ`.

`avail[r]` is the number of natural cards of rank `r` in hand. The wild cards
are excluded from it: `HandView::rank_count` already does this.

**Deficit.** `deficit = sum over r of max(0, R[r] - avail[r])`. The abstract
action is feasible when `deficit <= wilds` and no slot needing a joker is short,
because a wild card may never stand for a joker.

## Layer 1: which abstract actions to visit

When leading, visit every abstract action. When following, visit only those that
beat the top play (`beats_reading`), which is what makes most follow decisions
cheap: non-bomb tops admit only higher keys of the same type plus the bomb class.

## Layer 2: concretization

For each feasible abstract action, pick the actual cards.

For one rank `r` needing `k` cards, the candidate card ids are the suits of `r`
that the hand holds, each with multiplicity 1 or 2, and the wild id is not among
them. Let `take` be the number of natural cards used at `r`.

* `wild_usage = minimal`: `take = min(k, avail[r])`, so wild cards only ever
  fill a genuine shortfall. Straight flushes restrict the candidates to suit `s`
  before this is applied.
* `wild_usage = all`: `take` ranges over `min(k, avail[r])` down to
  `max(0, k - remaining_wilds)`, which is what lets a player spend a wild card
  to keep a natural one. Only reachable in full mode.

Within a rank, enumerate every multiset of `take` cards from that rank's card
ids. Across ranks, take the cross product. The wild cards fill what is left.

## The three canonical reductions

Each is behind its own flag in `ActionConfig` and each must be individually
switchable, because M1 ablates them.

1. `prune_dominated_readings`: group the generated actions by card multiset and,
   within one type, keep only the highest key. Keep every type: a player may have
   a cooperative reason to declare the weaker type. For bomb-class types compare
   `strength`, not the raw key.
2. `suit_dedup`: two actions are equivalent when they share the abstract id, the
   wild count and the sub-multiset of their SF-relevant cards. Keep the member
   whose sorted card id vector is lexicographically smallest. A card of rank `r`
   and suit `s` is SF-relevant when some straight window containing `r` could
   still become a straight flush in suit `s` out of this hand's own cards plus
   its wild cards.
3. `wild_usage = minimal`: as above.

## Invariants the tests enforce

* Full mode equals the oracle's `legal_actions` on hands of 12 cards or fewer,
  for random levels and random tops (`tests/test_movegen_crosscheck.py`).
* Canonical mode is a subset of full mode, and every full-mode action is
  represented either by its abstract id or by a stronger reading of the same
  cards.
* Leading never emits pass; following always does.
* Nothing allocates per call beyond growing the caller's output vector.

## Tribute candidates

`generate_tribute` offers the distinct card ids of the highest power in hand,
wild cards excluded (RULES.md 9.3). `generate_back_tribute` offers every card of
natural rank 2 to 10 that is not a level card of any suit; when nothing
qualifies and `back_tribute_fallback` is set, it offers every card of the lowest
power instead. Both emit single-card actions of type `Tribute` / `BackTribute`
so that they travel the same batch path as plays.
