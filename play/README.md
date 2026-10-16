# Play against the model in a browser

`play/server.py` serves a single-page Guandan table (`play/web/`). One person
sits at the bottom seat; the partner and both opponents are the same history
Transformer checkpoint (default `m7-u73780`, the latest milestone as of
October 15, 2026). A game is a full match from level 2 through A, under the
`house` rules. Model seats play greedily from their own observation plus the
shared public stream, and use the engine tribute heuristic, as in evaluation.

```sh
play/share.sh            # local + LAN link
play/share.sh --public   # also a temporary https://*.trycloudflare.com link
```

`CHECKPOINT=... play/share.sh` swaps the model; `PORT` changes the port
(default 18688). Games live in server memory: a refresh resumes the same
match, a server restart loses it.

Controls: tap or swipe across cards to select; 理牌 puts the selection into
its own column, 还原 restores the automatic order, 同花顺 cycles through
straight flushes in hand, 提示 cycles legal plays in the model's own
preference order for the player's seat, and the AI 概率 panel (on by
default) lists the model's own probability for each legal play from the
player's seat (the canonical list, negligible ones summarized); tapping a
row selects those cards. When selected cards with a wild
card have several readings, the player picks one. Keyboard: Enter plays,
Space passes, H hints. `?fast` in the URL shortens animations for testing.

The human seat matches selected cards against the engine's legal list with
suit de-duplication off, so any concrete suit choice is accepted; model seats
use the canonical list they were trained on. Forced passes are automatic.
