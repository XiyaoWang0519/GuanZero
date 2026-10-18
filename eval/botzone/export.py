"""Export a ``history_ppo`` checkpoint's actor to the Botzone bot's ``.npz``.

Only the parameters of the policy path are written (no critic, optimizer,
auxiliary or response heads), in FP32, with the architecture and the
checkpoint's identity under ``__config__``. Usage::

    python -m eval.botzone.export CHECKPOINT OUT.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POLICY_PREFIXES = ("bos", "public.", "round_embedding.", "phase_embedding.", "stream.",
                   "stream_norm.", "private.", "seat.", "q_proj.", "kv_proj.", "out_proj.",
                   "attention_norm.", "feed_forward.", "output_norm.", "action_tower.",
                   "fusion.")


def policy_weights(state_dict: dict) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().float().numpy()
            for key, value in state_dict.items() if key.startswith(POLICY_PREFIXES)}


def export(checkpoint: str | Path, out: str | Path) -> dict:
    from eval.history_policy import load_history_policy

    policy = load_history_policy(checkpoint, device="cpu")
    actor = policy.actor
    config = dict(actor.config.__dict__)
    config.update(checkpoint=str(checkpoint), checkpoint_id=policy.checkpoint_id,
                  heuristic_tribute=policy.heuristic_tribute)
    weights = policy_weights(actor.state_dict())
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, __config__=np.array(json.dumps(config)), **weights)
    return {"out": str(out), "bytes": out.stat().st_size, "tensors": len(weights),
            "parameters": int(sum(v.size for v in weights.values())),
            "checkpoint_id": policy.checkpoint_id}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.checkpoint, args.out), indent=2))


if __name__ == "__main__":
    main()
