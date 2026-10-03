# Botzone bot, steps 1, 2 and 4 (October 3, 2026)

Goal: enter the main lineage (u15094, `aux-full/latest.pt`, id `98b489f5…`) on
the Botzone GuanDan ladder, where DanLM ranked first in April 2026. Done here:
the protocol layer (1), a NumPy forward pass (2) and a local judge (4). Not
done: step 3, running without our compiled `gd` extension (Botzone's sandbox
is Python 3.6 with NumPy), and any upload.

## What exists

| File | Role |
|---|---|
| `eval/botzone/protocol.py` | Botzone card ids ↔ gd faces, levels, `claim` building and parsing, the per-round request log. Stdlib only, Python 3.6 grammar. |
| `eval/botzone/numpy_actor.py` | The actor's inference path in NumPy (stream encoder, private query, candidate head). Python 3.6 grammar. |
| `eval/botzone/export.py` | Checkpoint → `.npz` with the policy weights only. u15094: 1,357,953 parameters, 5.5 MB. |
| `eval/botzone/mirror.py` | Rebuilds the round in gd from what one seat was told, then replays it into a public token stream. |
| `eval/botzone/bot.py` | The bot: traditional mode (default) and `--keep-running`. |
| `eval/botzone/judge.py` | Local referee in gd speaking the Botzone protocol, with a torch reference check. |
| `tests/test_botzone.py` | 9 tests. |

The bot plays exactly like our evaluations do. Tribute and back-tribute use
the engine heuristic, and play is the greedy argmax over the canonical
candidates. The one difference: candidates Botzone cannot express are
masked. The only known class is a full house whose triple is the level rank,
with both wild cards as the pair (T-FH-04).

## Protocol facts that shaped the code

- The real judge sends `history` as four positional slots: slot *i* holds the
  latest move of seat `(me + i) % 4` since our last move. This comes from
  FableDan's platform notes; the wiki instead shows a list of dicts, and both
  formats are parsed.
- Because of this, a jiefeng inside one window (the lead jumps to a finished
  seat's partner) overwrites the passes that closed the trick. The replay
  restores at most one pass per seat per window, and only for a seat that
  moves again later in that window. When the history cannot tell a jiefeng
  lead from the partner beating its finished partner's last card, the replay
  reads it as the jiefeng lead.
- The unseen hands are filled consistently with the public record: plays,
  tribute caps, and big jokers for anti-tribute. Most-constrained cards are
  placed first. A test checks that the actor's inputs (observation,
  candidates, stream) are identical across different fillings.
- In traditional mode a decision takes about 10 ms in process and at most
  283 ms including process start. The limit is about 6 s for Python.

## Verification

- NumPy and torch logits agree within `2e-5` absolute on a small random actor
  with every head type (test).
- On 600 judge games, u15094 played seats 0/2 or 1/3 alternately against
  greedy, with an even mix of no tribute, single and double tribute (65
  anti-tributes). Results: 55,611 decisions, **0 illegal responses, 0 fallbacks,
  and 22,126 of 22,126 multi-candidate bot plays equal to the torch
  `HistoryPolicy` on the referee's true state and stream**. Score: +2.39
  levels per round, 558 of 600 rounds won. Report:
  `.work/botzone/judge-u15094-greedy-600.json`.
- Subprocess runs in both keep-running and traditional mode played without
  errors.

## Open risks before upload

1. **Double-tribute tie.** gd's `house` profile (which matches DanLM, RULES.md
   13.3) routes the Banker's downstream neighbour's card to the Banker. The
   Botzone wiki says the first finisher takes the last finisher's card. When
   the two disagree and the tied cards differ in suit, our mirror of our own
   hand is wrong; when they are the same face, only the physical id
   differs. The local judge follows gd, so it cannot detect this. The first
   real logs will settle it; if Botzone follows the wiki, the fix is a
   routing option in the bot.
2. **Protocol details taken second-hand.** The positional history,
   `tribute_cards` values given as lists, and the ordering rules come from
   the wiki and FableDan's notes, not from logs of our own. The local judge
   reproduces our reading of them, not the platform.
3. **Step 3.** `mirror.py` and `bot.py` need `gd` (move generation,
   observation and action encoding, tribute heuristic). Botzone cannot load
   our extension, so these need a pure-Python port, checked against gd
   bit for bit, before the bot can run there.
