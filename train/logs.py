"""Sample complete rounds for the supervised belief and behaviour experiments."""
import os
from pathlib import Path

import numpy as np

from train.buffer import Decision

TOKEN_DIM = 4 + 154 + 28
SCHEMA_VERSION = 2
# Schema 3 adds the candidate set of every decision and per-token metadata
# (abstract id, forced flag, phase) so that behaviour probes can be trained.
CANDIDATE_SCHEMA_VERSION = 3
# Per-decision driver codes: which player chose the action at that row.
DRIVER_POLICY = 0
DRIVER_BOT = 1


def public_token(event: object) -> np.ndarray:
    token = np.zeros(TOKEN_DIM, dtype=np.uint8)
    token[int(event.seat)] = 1
    token[4:158] = event.encoded_action
    # Tribute structure is private, even if an older engine emits those flags.
    token[4 + 146:4 + 154] = 0
    token[158 + int(event.cards_left)] = 1
    return token


def token_meta(event: object) -> tuple[int, int, int]:
    """Per-token metadata for schema 3: (abstract id, forced flag, phase).

    The abstract id is -1 for tribute and back-tribute actions, which have no
    entry in the play vocabulary (`gd.NUM_ABSTRACT`).
    """
    return int(event.action.abstract_id), int(bool(event.forced)), int(event.phase)


def save_round(path: Path, decisions: list[Decision], tokens: list[np.ndarray],
               group: str, meta: dict | None = None,
               token_metas: list[tuple[int, int, int]] | None = None) -> None:
    """Write one round. `meta` selects schema 2 with style and identity fields.

    Without `meta` the schema-1 payload is written unchanged, so self-play
    collections and archived datasets keep loading exactly as before. With
    `meta` the round additionally records `match_id`, `round_index`, `env_id`,
    the per-decision `driver`, the four seats' `styles` and `seat_driver`, the
    `style_region` label and the `styled` flag.

    `token_metas`, one `token_meta` tuple per token, selects schema 3 on top
    of schema 2. Every decision must then carry its candidate set: the round
    additionally records the ragged `cand` rows with `cand_offsets`, the
    per-candidate `cand_abstract` ids, the chosen index `choice`, and the
    per-token `token_abstract`, `token_forced` and `token_phase` arrays.
    """
    if not decisions:
        return
    if token_metas is not None and meta is None:
        raise ValueError("schema 3 candidate logs require the schema 2 metadata")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": np.asarray(1), "group": np.asarray(group),
        "obs": np.stack([d.obs for d in decisions]),
        "hidden": np.stack([d.hidden for d in decisions]),
        "seat": np.asarray([d.seat for d in decisions], dtype=np.int64),
        "prefix": np.asarray([d.prefix for d in decisions], dtype=np.int64),
        "tokens": np.asarray(tokens, dtype=np.uint8).reshape(-1, TOKEN_DIM),
    }
    if meta is not None:
        styles = np.asarray(meta["styles"], dtype=np.float32)
        seat_driver = np.asarray(meta["seat_driver"], dtype=np.int64)
        if styles.ndim != 2 or styles.shape[0] != 4 or seat_driver.shape != (4,):
            raise ValueError("schema 2 needs one style vector and driver per seat")
        payload.update(
            schema_version=np.asarray(SCHEMA_VERSION),
            match_id=np.asarray(int(meta["match_id"]), dtype=np.int64),
            round_index=np.asarray(int(meta["round_index"]), dtype=np.int64),
            env_id=np.asarray(int(meta["env_id"]), dtype=np.int64),
            driver=np.asarray([seat_driver[d.seat] for d in decisions], dtype=np.int64),
            styles=styles, seat_driver=seat_driver,
            style_region=np.asarray(str(meta["style_region"])),
            styled=np.asarray(bool(meta["styled"])),
        )
    if token_metas is not None:
        if len(token_metas) != len(tokens):
            raise ValueError("schema 3 needs one metadata tuple per token")
        if any(d.cand is None or d.cand_abstract is None or d.choice < 0 for d in decisions):
            raise ValueError("schema 3 needs the candidate set and choice of every decision")
        counts = [len(d.cand) for d in decisions]
        for d in decisions:
            if d.cand.shape[1:] != (154,) or len(d.cand_abstract) != len(d.cand):
                raise ValueError("candidate rows must be [k, 154] with one abstract id each")
            if not d.choice < len(d.cand) or not np.array_equal(d.cand[d.choice], d.action):
                raise ValueError("the chosen candidate must equal the decision's action")
        metas = np.asarray(token_metas, dtype=np.int64).reshape(-1, 3)
        payload.update(
            schema_version=np.asarray(CANDIDATE_SCHEMA_VERSION),
            cand=np.concatenate([np.asarray(d.cand, dtype=np.uint8) for d in decisions]),
            cand_offsets=np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
            cand_abstract=np.concatenate([np.asarray(d.cand_abstract, dtype=np.int64)
                                          for d in decisions]),
            choice=np.asarray([d.choice for d in decisions], dtype=np.int64),
            token_abstract=metas[:, 0], token_forced=metas[:, 1], token_phase=metas[:, 2],
        )
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)
