"""Seconds per Transformer decision vs stream prefix length (1 CPU thread, evaluation path)."""
import random
import time

import gd
import numpy as np
import torch

torch.set_num_threads(1)
import ablate_lib as L
from train.history_model import PublicStream

z = np.load("out/donor_full_b11_0.donors.npz")
tokens, rounds, phases = z["tokens"], z["rounds"], z["phases"]
base, b11 = L.load_base(), L.load_b11()
engine = gd.Engine()
state = gd.MatchState()
engine.new_match(state, 7)
while int(state.phase) != int(gd.Phase.Play):
    engine.apply(state, engine.legal_actions(state)[0])
actions = engine.legal_actions(state)
rng = random.Random(0)
print("legal", len(actions))
for n in (0, 100, 300, 600, 1000, 1500, 2000):
    s = PublicStream()
    for j in range(n):
        s.append_token(tokens[j], int(rounds[j]), int(phases[j]))
    base.stream, base.started = s, True
    base.verify_stream = lambda st: None
    reps = 5
    t = time.perf_counter()
    for _ in range(reps):
        base.select(engine, state, actions, rng)
    print(f"prefix {n:5d}: {1000 * (time.perf_counter() - t) / reps:8.1f} ms per decision", flush=True)
t = time.perf_counter()
for _ in range(50):
    b11.select(engine, state, actions, rng) if hasattr(b11, "select") else None
print(f"B11: {1000 * (time.perf_counter() - t) / 50:.1f} ms per decision")
