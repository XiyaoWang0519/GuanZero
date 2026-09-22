# Guandan rules specification

Status: v0.2 for milestone M0. House rules decided by the owner on Sept. 20, 2026 (section 13). Owner: Irvin. Audience: whoever implements or tests the engine, human or agent.

This document is the acceptance standard for the rules engine. If the engine and this document disagree, one of them has a bug. Every example in section 12 has been executed against the Python oracle in `gd_reference.py` (run `python3 test_gd_reference.py`), so the examples and the oracle agree with each other today.

## 1. Sources and alignment policy

The reference ruleset is the one implemented by OpenGuanDan (arXiv 2602.00676, code at https://github.com/GameAI-NJUPT/OpenGuanDan). It matches the rules described in DanZero (arXiv 2210.17087) and the rules used in the Chinese Guandan AI competition hosted by NJUPT. We align to it for one practical reason: its simulator ships seven baseline agents, and our evaluation is only comparable if both sides play the same game.

The papers leave some details unspecified. Section 13 records how each one was resolved. The owner decided the ones that are real rule choices. Those decisions are the `house` rule profile, which is the default for training, internal evaluation and human play.

A second profile, `ogd`, mirrors whatever the OpenGuanDan simulator does. M0 fills it in by replaying logged OpenGuanDan games through our engine (DESIGN.md section 9.2), and the differential tests run under it. Inside the OpenGuanDan simulator its own rules apply anyway, since it is the referee there. The two profiles can only differ in rare tribute ties and in match-level rules around level A, so training under `house` costs nothing measurable against the benchmark agents. Every item in section 13 sits behind a named config field so that a profile is just a set of values.

## 2. Glossary

| Term in this repo | Chinese | Notes |
|---|---|---|
| Match | 一场 | Play from level 2 until a team passes A. The papers call this a game or an episode. |
| Round | 小局 | One deal of 108 cards, played until the finishing order is decided. |
| Trick | 一轮 | From one lead until every other active player has passed. |
| Banker, Follower, Third, Dweller | 头游, 二游, 三游, 末游 | Finishing positions 1 to 4 in a round. Names follow the papers. |
| Double win | 双下 | Banker and Follower are teammates. |
| Level | 级数 | Per team, from 2 up to A. |
| Round level | 本局级牌点数 | The level the current round is played at. |
| Level card | 级牌 | Any of the 8 cards whose rank equals the round level. |
| Wild card | 逢人配 | The 2 heart level cards. |
| Tribute, back-tribute, anti-tribute | 进贡, 还贡, 抗贡 | Card exchange before rounds 2 and later. |
| Teammate lead | 接风 | The partner of a player who just finished inherits the lead. |
| Single, pair, triple | 单张, 对子, 三同张 | |
| Full house | 三带二 | |
| Straight | 顺子 | Exactly 5 cards. |
| Tube | 三连对 (木板) | Exactly 3 consecutive pairs. |
| Plate | 钢板 | Exactly 2 consecutive triples. |
| Bomb | 炸弹 | 4 to 10 cards of one rank. |
| Straight flush | 同花顺 | |
| Joker bomb | 天王炸 | All 4 jokers. |
| Black joker (BJ), red joker (RJ) | 小王, 大王 | RJ is the stronger one. |

## 3. Cards and encoding

The deck is two standard 54-card decks: 108 cards, 54 distinct card ids, 2 copies of each.

| Item | Encoding |
|---|---|
| Rank index | `2=0, 3=1, 4=2, 5=3, 6=4, 7=5, 8=6, 9=7, T=8, J=9, Q=10, K=11, A=12` |
| Suit index | `S=0, H=1, C=2, D=3` |
| Card id | `rank * 4 + suit` for ids 0 to 51, `52 = BJ`, `53 = RJ` |
| Text form | Suit letter then rank character: `S2`, `HT`, `DA`. Jokers are `SB` (black) and `HR` (red). |
| Hand | 54 counts, each 0, 1 or 2 |

The text form follows the NJUPT platform convention. The OpenGuanDan paper confirms the suit plus rank pattern (`S2`, `HA`, `D2`). The ten character and the joker strings are from memory of the older platform and need confirmation (parity check O9).

Seats are numbered 0 to 3. `next(s) = (s + 1) % 4` is the play order, which corresponds to counterclockwise at a physical table. `team(s) = s % 2` and `partner(s) = (s + 2) % 4`.

## 4. Level cards, wild cards and the two orderings

Each round has a round level `L`, a rank index from 0 to 12.

There are two orderings and mixing them up is the classic engine bug.

**Power order.** Used for singles, pairs, triples, the triple inside a full house and the rank of a bomb. The level rank is lifted above the ace.

```
power(rank, L):
    rank 13 (BJ) -> 13, rank 14 (RJ) -> 14
    rank == L    -> 12
    rank <  L    -> rank
    rank >  L    -> rank - 1
```

With `L = 7` the power order is `2 3 4 5 6 8 9 T J Q K A 7 BJ RJ`.

**Sequence order.** Used for straights, straight flushes, tubes and plates. It is the natural rank order and the round level has no effect. The ace may sit below the 2 or above the king. Sequences never wrap: `K A 2` is not consecutive. Jokers never appear in sequences. A level card inside a sequence sits at its natural rank.

| Type | Windows, lowest to highest | Count | Key |
|---|---|---|---|
| Straight, straight flush | `A2345, 23456, ..., TJQKA` | 10 | window index 0 to 9 |
| Tube | `AA2233, 223344, ..., QQKKAA` | 12 | window index 0 to 11 |
| Plate | `AAA222, 222333, ..., KKKAAA` | 13 | window index 0 to 12 |

Window index 0 is always the ace-low window.

**Wild cards.** The two heart cards of rank `L` are wild.

1. Inside a combination of two or more cards, a wild card may stand for any card of rank 2 to A in any suit. It may never stand for a joker.
2. Played alone, a wild card is an ordinary level card: a single of power 12.
3. A combination made only of wild cards is read at the level rank. Two wild cards are a pair of power 12.
4. Wild substitution has no copy limit. Eight natural 9s plus both wild cards form a 10-card bomb of 9s.
5. Using a heart level card as itself needs no special case. It is substitution where the target happens to be the same card.

Natural level cards in the other three suits are ordinary cards of rank `L`.

## 5. Combination types

A play is a multiset of cards from the player's hand together with a declared reading `(type, key)`. The same cards can have several readings when wild cards are involved.

| Type | Size | Composition after wild substitution | Key |
|---|---|---|---|
| Single | 1 | Any card | power |
| Pair | 2 | Two cards of one rank, `BJ BJ` or `RJ RJ`. `BJ RJ` is not a pair. | power |
| Triple | 3 | Three cards of one rank. Jokers cannot form a triple. | power |
| Full house | 5 | A triple plus a pair of a different rank. The pair may be a joker pair (decided, O3). | power of the triple |
| Straight | 5 | One card in each rank of a straight window, not all one suit | window index |
| Tube | 6 | Two cards in each rank of a tube window | window index |
| Plate | 6 | Three cards in each rank of a plate window | window index |
| Bomb | 4 to 10 | `n` cards of one rank from 2 to A. Sizes 9 and 10 need wild cards. The level rank tops out at 8. | `(n, power)` |
| Straight flush | 5 | A straight window with all five cards in one suit | window index |
| Joker bomb | 4 | `BJ BJ RJ RJ` | none |
| Pass | 0 | No cards. Illegal when leading. | none |

Any other multiset is not a legal play. In particular a lone `BJ BJ RJ`, five cards with a joker in a sequence, a 6-card straight and an 11-card bomb are all illegal.

## 6. When one play beats another

Let `T` be the current top play of the trick and `C` the candidate.

1. If the player is leading, any non-pass play is legal.
2. Pass is always legal when following.
3. Bomb, straight flush and joker bomb form the bomb class. A bomb class play beats every play outside the bomb class.
4. Inside the bomb class, compare strength tuples. Larger wins and equal never wins.

```
strength(bomb of n cards)  = (n, power)
strength(straight flush)   = (5.5, window index)
strength(joker bomb)       = (99, 0)
```

   This yields: 4-bomb < 5-bomb < straight flush < 6-bomb < ... < 10-bomb < joker bomb. Suits never break ties between straight flushes.
5. Outside the bomb class, `C` beats `T` only if both have the same type and `C.key > T.key`. The seven non-bomb types are mutually incomparable.

## 7. Round flow

Phases: `DEAL -> TRIBUTE -> BACK_TRIBUTE -> PLAY -> ROUND_END`. Round 1 of a match skips both tribute phases. Anti-tribute skips them too.

### 7.1 Dealing and the first leader

Shuffle 108 cards and deal 27 to each seat. In round 1 the first leader is a uniformly random seat drawn from the environment seed (O6). In later rounds the leader comes from section 9.

The engine must also accept an explicit `DealSpec`: four hands, round level, team levels and either a leader or the previous finishing order. Giving the previous order makes the round start with its tribute phase, which the tribute experiments in DESIGN.md section 8.3 need. Duplicate-deal evaluation and the scenario tests depend on it.

### 7.2 Playing a trick

State needed: four hands, `top` (the top play or none), `holder` (the seat that played `top`), `passes` since `top`, the finishing order so far and the seat to move.

1. The seat to move is always active, meaning it still holds cards.
2. When `top` is none the seat leads and must play.
3. Otherwise the seat passes or plays something that beats `top`. A play replaces `top`, sets `holder` and resets `passes` to 0. A pass increments `passes`.
4. After a play that empties the hand, append the seat to the finishing order and check section 7.3.
5. Advance to the next active seat in play order.
6. The trick ends when every active seat other than `holder` has passed since `top`. Equivalently `passes` equals the number of active seats, not counting `holder` if `holder` is still active.
7. When the trick ends: if `holder` is still active, `holder` leads the next trick. If `holder` finished with that play, `partner(holder)` leads (teammate lead). The partner is always active at that point, because the round would already be over otherwise. As a defensive fallback the next active seat after `holder` leads.

Note that after a player goes out, everyone else, the partner included, still gets a chance to beat that last play before the teammate lead applies.

### 7.3 End of a round

The round ends as soon as either condition holds:

1. Both members of one team have finished. If they are Banker and Follower this is a double win and the other two players keep their cards. Their relative order is irrelevant to every rule (O5, logging only).
2. Three players have finished. The remaining player is the Dweller.

## 8. Levels and match result

1. Only the Banker's team gains levels: 3 if the partner is the Follower, 2 if Third, 1 if Dweller.
2. `new level = min(level + gain, A)`. The ace cannot be skipped.
3. The next round is played at the winning team's new level.
4. The team that won the previous round owns the current round, because the round is played at its level. Round 1 has no owner.
5. A team passes A and wins the match only in a round it owns: the round level is A, the team has the Banker and the Banker's partner is not the Dweller (decided, O4).
6. If the other team is also at level A and wins such a round, it does not win the match, whatever its finishing positions. It stays at A and owns the next round, which is then its own attempt.
7. Every owned round at level A that does not pass counts as one failure for the owner, whether the owner lost the round or won it with Banker and Dweller. The count is cumulative. At 3 the owner's level resets to 2 and its count resets to 0 (decided, O4). `a_fail_limit = 0` disables the reset.
8. Order of operations at round end: decide the match win, then count the failure, then promote the round winner, then apply the reset. The oracle's `end_of_round` is normative.

Both papers agree on items 1 to 3 and DanZero states the Follower or Third condition of item 5. Ownership and the failure count are house decisions. The OpenGuanDan text is internally inconsistent here, so its behavior is recorded in the `ogd` profile during M0.

## 9. Tribute

Applies from round 2 on. Let `B, F, T, D` be the previous round's Banker, Follower, Third and Dweller.

### 9.1 Single tribute (no double win)

1. If `D` holds both red jokers: anti-tribute. No cards move and `B` leads.
2. Otherwise `D` gives one tribute card to `B`. Then `B` gives one back-tribute card to `D`. `D` leads.
3. This applies even when `D` is `B`'s partner (decided, O10).

### 9.2 Double tribute (previous round was a double win)

1. If the two losers together hold both red jokers, whether 2 and 0 or 1 and 1: anti-tribute. `B` leads.
2. Otherwise each loser selects a tribute card. The card with higher power goes to `B` and the other goes to `F`.
3. Each receiver returns a back-tribute card to the player whose card they received.
4. The payer of the higher tribute leads.
5. If both tribute cards have equal power, the Banker's upstream seat pays the Banker and leads. Upstream means the seat that plays right before the Banker, `(B + 3) % 4`. The other loser pays `F` (decided, O1).

### 9.3 Which cards may move

**Tribute card.** The payer must give a card of the highest power in hand, ignoring wild cards. If several distinct cards tie, for instance two aces of different suits, the payer chooses. The legal tribute actions are therefore the distinct card ids at that top power.

**Back-tribute card.** Any card of natural rank 2 to 10 that is not a level card. Level cards rank above the ace, so they do not count as ten or lower even when their face value is small (decided, O2). If no card qualifies, which is nearly impossible, any card of the lowest power in hand.

All four cards involved are public information. Hands are back to 27 cards each when play starts.

## 10. What each player can observe

| Public | Private |
|---|---|
| Every play and pass, in order, with the exact cards | Own hand |
| Number of cards left in every hand (see the note below) | |
| Round level and both team levels | |
| Finishing order so far | |
| Tribute and back-tribute cards with payer and receiver | |
| Anti-tribute events, including which seats held the red jokers (the benchmark broadcasts `antiPos`) | |

**Card counts (decided, O7).** The house rule is the table rule: a player must declare once they hold ten cards or fewer, and counts above ten are not announced. This rule only governs what a human interface shows to a human player. It does not change what an agent may know. Every play is public and every hand starts the play phase at 27 cards, so a hand's size is always 27 minus the cards that seat has played. The agent's features therefore carry exact counts at all times, which is ordinary card counting and also matches the `publicInfo` field of the benchmark.

## 11. Action model

### 11.1 Abstract actions

An abstract action is a `(type, key)` pair with suits and wild usage erased. There are 393:

| Type | Count | Notes |
|---|---|---|
| Pass | 1 | |
| Single | 15 | powers 0 to 14 |
| Pair | 15 | |
| Triple | 13 | |
| Full house | 182 | 13 triple ranks times 14 pair choices (12 other ranks, BJ pair, RJ pair) |
| Straight | 10 | |
| Tube | 12 | |
| Plate | 13 | |
| Bomb | 91 | 13 ranks times sizes 4 to 10. Some are unreachable, such as a 9-card bomb of the level rank. |
| Straight flush | 40 | 10 windows times 4 suits |
| Joker bomb | 1 | |

Note the full house row: the abstract id records the pair rank even though the key ignores it, because which pair you give up matters strategically.

### 11.2 Concrete actions and the two enumeration modes

A concrete action is `(type, key, card multiset)` and the engine derives `wilds_used` from the cards.

**Full mode.** Every distinct `(type, key, card multiset)` that is legal. Used for differential testing against OpenGuanDan and against the oracle. Expect up to the order of 10^4 actions on an opening hand.

**Canonical mode.** Used for training and play. It applies three reductions, each behind its own flag:

1. `prune_dominated_readings`: for one card multiset keep only the highest key per type. Readings of different types are all kept, because a player may have a cooperative reason to declare the weaker type (letting the partner overtake cheaply).
2. `suit_dedup`: a card of rank `r` and suit `s` is SF-relevant if some straight window containing `r` could still become a straight flush in suit `s` using the hand's own cards plus the wild cards it holds. Two concrete actions are equivalent if they share the abstract action, use the same number of wild cards and contain the same SF-relevant cards. Keep the member with the lexicographically smallest id vector. This is lossless up to opponent belief effects, since suits influence nothing else.
3. `wild_usage = minimal`: only enumerate plays that use the fewest wild cards possible for the chosen natural cards. The alternative value `all` exists for experiments. The one known cost of `minimal` is that the agent cannot spend a wild card to keep a natural SF-relevant card.

Canonical mode must always be a subset of full mode. The guarantee it gives, corrected in M0, is:

1. Every type available in full mode is available in canonical mode.
2. For every type, the best reading available in canonical mode is the best one available in full mode. No play is ever weakened.
3. Whenever a full mode action has no counterpart in canonical mode, canonical mode holds a strictly stronger reading of the same type.

The earlier wording, that every full mode action survives as the same abstract action or as a stronger reading of the same cards, is too strong and the engine cannot satisfy it. Reduction 1 keeps only the highest key per type for one multiset, and with a wild card in hand every multiset that reads as a low full house also reads as a higher one, so the low key disappears outright rather than surviving on some other multiset. What is lost is the ability to declare a deliberately weak play of a type the player can also play strongly. Reduction 1 already accepts that cost for keys; it keeps the different types precisely because the cooperative argument applies across types.

## 12. Test vectors

All verified by `test_gd_reference.py`. `W` means a wild card in prose. Keys are shown as returned by the oracle.

### Ordering and singles

| ID | Level | Input | Expectation |
|---|---|---|---|
| T-ORD-01 | 7 | single `S7` over single `SA` | beats |
| T-ORD-02 | 7 | `SB` over `S7`, `HR` over `SB` | both beat. Keys 13 and 14. |
| T-ORD-03 | 7 | `H7` alone | single of power 12, equal to `S7`, so neither beats the other |
| T-ORD-04 | 7 | power of 8 and of 6 | 5 and 4 |
| T-ORD-05 | A | power of A and of K | 12 and 11 |

### Pairs and triples

| ID | Level | Cards | Readings |
|---|---|---|---|
| T-PAIR-01 | 7 | `H7 SA` | pair, power 11 |
| T-PAIR-02 | 7 | `H7 SB` | none, a wild card cannot be a joker |
| T-PAIR-03 | 7 | `SB HR` | none |
| T-PAIR-04 | 7 | `SB SB`, `HR HR` | pair 13, pair 14 |
| T-PAIR-05 | 7 | `H7 H7` | pair 12 only |
| T-PAIR-06 | 7 | `H7 S7` | pair 12 |
| T-TRI-01 | 7 | `H7 H7 S9` | triple, power 6 |
| T-TRI-02 | 7 | `SB SB H7` | none |

### Full houses

| ID | Level | Cards | Readings |
|---|---|---|---|
| T-FH-01 | 7 | `S5 D5 C5 SB SB` | full house, key 3 (joker pair allowed) |
| T-FH-02 | 7 | `S5 D5 C5 SB H7` | none |
| T-FH-03 | 7 | `S5 D5 H7 S9 D9` | full house key 3 and key 6. Canonical keeps key 6. |
| T-FH-04 | 7 | `H7 H7 S9 D9 C9` | 5-bomb of 9s and full house key 6. Canonical keeps both. |
| T-FH-05 | 7 | `S7 D7 C7 S2 D2` over `SA DA CA SK DK` | key 12 beats key 11 |

### Straights and straight flushes

| ID | Level | Cards | Readings |
|---|---|---|---|
| T-STR-01 | 7 | `SA D2 C3 S4 H5` | straight 0 |
| T-STR-02 | 7 | `SK DA C2 S3 H4` | none, no wraparound |
| T-STR-03 | 7 | `ST DJ CQ SK HA` | straight 9 |
| T-STR-04 | 7 | `S5 D6 S7 C8 D9` | straight 4, level card at its natural rank |
| T-STR-05 | 7 | `SA S2 S3 S4 S5` | straight flush 0 |
| T-STR-06 | 2 | `H2 S6 D7 C8 D9` | straight 4 and straight 5. Canonical keeps 5. |
| T-STR-07 | 2 | `S3 S4 S5 S6 S7 S8` | none, six cards |
| T-STR-08 | 2 | `ST SJ SQ SK SB` | none, joker in a sequence |
| T-STR-09 | 7 | `H7 S3 S4 S5 S6` | straight 1 and 2, straight flush 1 and 2. Canonical keeps straight 2 and straight flush 2. |
| T-STR-10 | A | `HT HJ HQ HK HA` | `HA` is wild here: straight 8 and 9, straight flush 8 and 9 |

### Tubes and plates

| ID | Level | Cards | Readings |
|---|---|---|---|
| T-SEQ-01 | 7 | `SQ DQ SK DK SA DA` | tube 11 |
| T-SEQ-02 | 7 | `SK DK SA DA S2 D2` | none |
| T-SEQ-03 | 7 | `SA DA S2 D2 S3 D3` | tube 0 |
| T-SEQ-04 | 7 | `SK DK CK SA DA CA`, `SA DA CA S2 D2 C2` | plate 12, plate 0 |
| T-SEQ-05 | 7 | `H7 H7 S3 D3 S4 D4` | tube 1, tube 2 and plate 2. Canonical keeps tube 2 and plate 2. |

### Bombs

| ID | Level | Input | Expectation |
|---|---|---|---|
| T-BOMB-01 | 7 | all eight 7s | bomb `(8, 12)` |
| T-BOMB-02 | 7 | eight 9s plus `H7 H7` | bomb `(10, 6)`. Adding an 11th card gives none. |
| T-BOMB-03 | any | 5-bomb of 2s over 4-bomb of aces | beats, size first |
| T-BOMB-04 | any | straight flush 0 over 5-bomb of level cards. 6-bomb of 2s over straight flush 9. | both beat |
| T-BOMB-05 | any | joker bomb over a 10-bomb | beats. The reverse does not. |
| T-BOMB-06 | 7 | `SB SB HR`, `SB SB HR H7` | none. `SB SB HR HR` is the joker bomb. |
| T-BOMB-07 | any | 4-bomb of 2s over full house key 12 | beats |
| T-BOMB-08 | any | straight flush 3 over straight flush 3 | does not beat |

### Mismatched types and legal sets

| ID | Level | Input | Expectation |
|---|---|---|---|
| T-MIS-01 | any | pair over single, tube over plate, straight over full house | never beats |
| T-LEG-01 | 7 | hand `S3 D3 H7 SB`, top is a pair of 2s | pass, plus pair of 3s as `S3 D3`, `S3 H7` and `D3 H7` |
| T-LEG-02 | 7 | same hand, leading | no pass. Readings available: singles 1, 12, 13, pair 1, triple 1. |

### Round bookkeeping and tribute

| ID | Input | Expectation |
|---|---|---|
| T-RND-01 | finish orders `[0,2,..]`, `[0,1,2,3]`, `[0,1,3,2]`, `[1,0,3,2]` | team 0 gains 3, 2, 1. Team 1 gains 2. |
| T-RND-02 | K plus 3, Q plus 3, A plus 1 | all give A |
| T-TRB-01 | level 7, payer holds `H7 S7 SA DA` | must give `S7` |
| T-TRB-02 | level 7, payer holds `H7 SA DA SK` | may give `SA` or `DA` |
| T-TRB-03 | level 7, payer holds `HR SB SA` | must give `HR` |
| T-TRB-04 | level 5, receiver holds `S5 S3 DT SJ` | may return `S3` or `DT`. The level card `S5` is excluded. |
| T-TRB-05 | double tribute, Banker 0, both cards of power 13 | seat 3 pays the Banker and leads, seat 1 pays the Follower |
| T-TRB-06 | double tribute, Banker 0, seat 1 gives power 14, seat 3 gives 13 | seat 1 pays the Banker and leads |
| T-OBS-01 | Seat 3 tributes `S9` to 0; seat 0 returns that `S9` | Known-holdings feature marks `S9` at 0 before the return, and only at 3 afterward. |
| T-OBS-02 | Same exchange, but seat 0 privately already held another `S9` | After returning one copy, the remaining private copy is not marked as publicly known. |
| T-OBS-03 | Seat 3 tributes `S9` to 0; seat 0 returns `S3`; seat 3 plays `S4`, then 0 plays `S9` | The different return preserves both known holdings. Playing `S4` preserves known `S3`; playing `S9` removes its guarantee. |
| T-OBS-04 | Seat 0 privately holds `S9`, receives another `S9`, returns `S3`, then plays one `S9` | One physical copy remains, but no publicly guaranteed copy remains. |
| T-MATCH-01 to 06 | match bookkeeping cases in `test_match_bookkeeping` | as asserted there, matching T-FLOW-09 and T-FLOW-11 to 13 |

### Flow scenarios to script in the engine tests

These need the state machine, so the oracle does not cover them.

| ID | Scenario | Expectation |
|---|---|---|
| T-FLOW-01 | 0 leads a pair, 1 and 2 pass, 3 beats it, 0, 1 and 2 pass | 3 leads next |
| T-FLOW-02 | 0 plays its last card, 1, 2 and 3 pass | 2 leads (teammate lead) |
| T-FLOW-03 | 0 plays its last card, 1 beats it, 2 and 3 pass | 1 leads |
| T-FLOW-04 | 0 finishes first, later 2 finishes | round ends at once, team 0 gains 3, seats 1 and 3 keep their cards |
| T-FLOW-05 | finish order 0, 1, then 3 | round ends, 2 is Dweller, team 0 gains 1 |
| T-FLOW-06 | previous order `0,1,2,3`, seat 3 holds `HR HR` | anti-tribute, 0 leads |
| T-FLOW-07 | previous double win by 0 and 2, seats 1 and 3 hold one `HR` each | anti-tribute, 0 leads |
| T-FLOW-08 | previous double win by 0 and 2, 1 gives `HR`, 3 gives `SB` | 0 receives `HR` from 1, 2 receives `SB` from 3, each returns to its payer, 1 leads |
| T-FLOW-09 | owner at A wins its round with Banker and Dweller | no match win, level stays A, failure count 1, keeps ownership |
| T-FLOW-10 | previous double win by 0 and 2, seats 1 and 3 both give `SB` | tie: seat 3, upstream of the Banker, pays 0 and leads. Seat 1 pays 2. |
| T-FLOW-11 | both teams at A, team 1 owns the round, team 0 finishes first and second | no match win. Team 1 failure count 1. Team 0 owns the next round. |
| T-FLOW-12 | owner at A with 2 failures loses the round | owner resets to level 2 with count 0, the winners promote |
| T-FLOW-13 | team at A wins a round played at the other team's level K | no match win and no failure. It owns the next round, played at A. |

## 13. Decisions and remaining parity checks

Items marked decided are house rules chosen by the owner. They are final for the `house` profile. M0 still records what the benchmark does for each of them in the `ogd` profile. Items marked parity are not rule choices. They only concern matching the simulator's output and are settled by trace replay.

| ID | Question | Resolution | Status | Config field |
|---|---|---|---|---|
| O1 | Double tribute with equal cards | The Banker's upstream seat pays the Banker and leads. The other loser pays the Follower. | Decided | `tribute_tie = upstream` |
| O2 | Level cards as back-tribute | Not allowed. Natural rank 2 to 10 and not a level card. Fallback if nothing qualifies: any lowest power card. | Decided | `back_tribute_level_cards = false` |
| O3 | Joker pair inside a full house | Allowed. The simulator agrees: probed at M0, it offers the `ThreeWithTwo` with both `SB SB` and `HR HR` as the pair. | Decided, parity closed | `full_house_joker_pair = true` |
| O4 | Passing A | Only in a round the team owns. Every owned A round that does not pass is a failure, cumulative. Three failures reset the team to level 2. | Decided | `pass_a_requires_owner = true`, `a_fail_limit = 3`, `a_fail_reset_level = 2` |
| O7 | Card count visibility | Declaration at ten cards or fewer. Governs the human interface only. Agents use exact counts derived from public plays. | Decided | `ui_count_visibility = le10` |
| O10 | Dweller paying tribute to a Banker on the same team | Tribute and back-tribute proceed as usual | Decided | `tribute_between_partners = true` |
| O5 | Order of the last two players after a double win | By seat from `next(Follower)`. No rule depends on it. | Parity | none |
| O6 | First leader in round 1 | Uniform random from the seed | Parity | `first_leader = random` |
| O8 | Does the reference list dominated readings, such as the lower of two straights for the same cards? | Yes, it lists them all and prunes nothing. Our full mode is therefore the right side of the comparison and canonical mode is ours alone. Probed at M0. | Parity closed | none |
| O9 | Exact strings for the ten, the jokers and the type names | Confirmed at M0: `T`, `SB`, `HR`, and the 13 type strings `Single, Pair, Trips, ThreePair, ThreeWithTwo, TwoTrips, Straight, StraightFlush, Bomb, FourKings, tribute, back, PASS`. The joker bomb is `FourKings`, which this table previously did not list. | Parity closed | `eval/ogd_adapter/normalize.py` |
| O11 | Does the simulator enumerate every wild card substitution? | No. A wild card keeps its own identity when that identity already fits, so the simulator lists fewer readings than we do. Characterized by probe and by trace replay, not emulated: our set is a strict superset, so no legal play is ever missing. | Parity closed, superset accepted | none |

### 13.1 What the OpenGuanDan simulator does (the `ogd` profile)

Recorded in M0 by reading the simulator's sources and by probing its move
generator. Evidence per item is in `docs/ogd_profile_findings.md` and
`docs/ogd_parity_probes.md`. These values are the `ogd` profile in
`cpp/include/gd/config.h`. They do not change any house decision; they exist so
that the differential tests of DESIGN.md 9.2 can run under the simulator's own
rules.

| ID | House | OpenGuanDan | Field |
|---|---|---|---|
| O1 pairing | The higher tribute card goes to the Banker | Fixed by seat: the Banker's downstream seat `(B + 1) % 4` always pays the Banker, the upstream seat always pays the Follower, whatever the cards are | `tribute_pairing` |
| O1 leader | The payer of the higher card leads; on a tie the Banker's upstream seat | Payer of the higher card leads, which agrees; on a tie the seat recorded last in the finishing order leads, which does not | `tribute_tie` |
| O2 | Natural rank 2 to 10, no level card of any suit; fall back to any lowest-power card | The same restriction, but no fallback path exists at all | `back_tribute_fallback` |
| O4 count | Every owned A round that does not pass is a failure, including a plain loss | Only an owned A round won with Banker and Dweller counts. A plain loss counts nothing and simply hands ownership over. | `a_fail_on_loss` |
| O4 limit | Reset to level 2 on the third failure | The check runs after the increment and compares with `> 3`, so the reset lands on the fourth | `a_fail_limit` |
| O4 draw | No such rule | After 50 resets the match is decided by accumulated victory count | `shuffle_limit` |
| O5 | By seat from `next(Follower)` | By ascending seat index. Logging only, no rule depends on it. | `double_win_tail` |
| O6 | Uniform random from the seed | Uniform random, `randint(0, 3)`. Agrees. | `first_leader` |
| O9 | See the adapter mapping table | Type strings `Single, Pair, Trips, ThreePair, ThreeWithTwo, TwoTrips, Straight, StraightFlush, Bomb, FourKings, tribute, back, PASS`. The joker bomb is `FourKings`, which this document previously did not list. Card strings `T`, `SB` and `HR` are confirmed. | `eval/ogd_adapter/normalize.py` |
| Card counts | Declaration at ten or fewer, interface only | `publicInfo.rest` carries exact counts to every player at all times. Agrees with the note in section 10. | `ui_count_visibility` |

### 13.2 Wild card substitution: the one divergence left (O11)

Replaying 1,000 logged matches, 1,783,202 decisions and 13,341 round ends
through our engine under the `ogd` profile leaves exactly one class of
difference, on 3,784 decisions, or 0.21%. In every case our legal set is a
strict superset: we never miss a play the simulator offers, and every action
the simulator's players actually chose was legal in our engine.

The simulator does not enumerate a wild card standing for something weaker than
what the card already is. Three probes of its move generator isolate it:

| Hand | Level | Simulator | Us and the oracle |
|---|---|---|---|
| `H2 H3 H4 H6 HA` | A | straight flush only | straight flush and straight |
| `S3 S4 S5 S6 H7` | 7 | straight flush and straight, both windows | the same |
| `SA HA HA DA DA` | A | no full house | full house, key 12 |

The wild card is the heart of the round level. In the first hand it is already
a heart among hearts, so the simulator reads the flush and stops; we also read
the wild as another suit, which gives the weaker plain straight that RULES.md
section 5 allows, since the five cards are then not all one suit. The second
hand is the control: the wild is a heart among spades, its own identity does
not fit, and the simulator enumerates both readings exactly as we do. In the
third the two wild cards are aces at level A, and the simulator will not let
them stand for a pair of another rank.

This is not emulated behind a config field. Emulating it would add a special
case to move generation for no gain in play strength, and nothing depends on
byte-exact action lists: the adapter matches the simulator's chosen action by
its content, not by its index into `actionList`.

Anti-tribute, the level gain of 3, 2 and 1, and the cap at A agree with this
document. O3 and O8 were settled by probing the move generator, which lives in a
compiled jar and cannot be read; see `docs/ogd_parity_probes.md`. The
simulator also passes the wild count to the generator as a caller-supplied
`heartsNum` rather than deriving it, which the adapter must get right.

## 14. Engine invariants for property tests

1. Conservation: for every card id, copies in the four hands plus copies played equals 2, after every step and after tribute.
2. When leading, the legal set is non-empty and has no pass. When following, it contains pass.
3. Soundness: every generated action is accepted by the oracle's `interpret` and `beats`.
4. Completeness: on random hands of 12 cards or fewer, the full mode legal set equals the oracle's `legal_actions`, for random levels and random tops.
5. Canonical mode is a subset of full mode, offers every type that full mode offers, and for each type offers the same best reading (section 11.2).
6. Termination: every round of random play ends in fewer than 600 steps.
7. The finishing order never repeats a seat and a finished seat never moves again.
8. Determinism: the same seed and the same action sequence give the same state hash on every platform.
9. Serialization round trip preserves the state hash.
10. Levels never decrease except through the A fail reset.
