"""Sample complete rounds for the later supervised belief experiment."""
import os
from pathlib import Path

import numpy as np

from train.buffer import Decision

TOKEN_DIM = 4 + 154 + 28


def public_token(event: object) -> np.ndarray:
    token = np.zeros(TOKEN_DIM, dtype=np.uint8)
    token[int(event.seat)] = 1
    token[4:158] = event.encoded_action
    # Tribute structure is private, even if an older engine emits those flags.
    token[4 + 146:4 + 154] = 0
    token[158 + int(event.cards_left)] = 1
    return token


def save_round(path: Path, decisions: list[Decision], tokens: list[np.ndarray],
               group: str) -> None:
    if not decisions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, schema_version=np.asarray(1), group=np.asarray(group),
            obs=np.stack([d.obs for d in decisions]),
            hidden=np.stack([d.hidden for d in decisions]),
            seat=np.asarray([d.seat for d in decisions], dtype=np.int64),
            prefix=np.asarray([d.prefix for d in decisions], dtype=np.int64),
            tokens=np.asarray(tokens, dtype=np.uint8).reshape(-1, TOKEN_DIM),
        )
    os.replace(temporary, path)
