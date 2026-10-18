"""Frozen diverse-opponent complete-match FULL/ROUND ablation; no optimizer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

import bench.qknorm_match_ablation as matches
from bench.public_history_diverse import load_actor, load_selected
from bench.public_history_probe import PublicCollector
from bench.qknorm_match_summary import ENDPOINTS, analyze


class PublicTrace:
    """Digest actual public actions only, including automatic passes; no forced bit."""
    def __init__(self, env, size):
        self.env = env
        self.hashes = [hashlib.sha256() for _ in range(size)]
        self.counts = np.zeros(size, np.int64)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def drain_public_actions(self):
        events = self.env.drain_public_actions()
        for event in events:
            if int(event.match_id) != 0:
                continue
            action = np.asarray(event.encoded_action, dtype=np.uint8)
            assert not action[146:].any()
            e = int(event.env_id)
            self.hashes[e].update(struct.pack("<4i",int(event.round_index),int(event.seat),
                                             int(event.phase),int(event.cards_left)))
            self.hashes[e].update(action[:146].tobytes())
            self.counts[e] += 1
        return events


class AblationCollector(matches.EvalCollector):
    last = None

    def __init__(self, env, *args, seeds, **kwargs):
        self.trace = PublicTrace(env,len(seeds))
        super().__init__(self.trace,*args,seeds=seeds,**kwargs)
        AblationCollector.last = self

    def _read_rows(self, batch):
        # Reuse the no-hidden-counts implementation, then select the learner's
        # current-round prefix. Opponents' prefixes and caches remain FULL.
        rows = PublicCollector._read_rows(self,batch)
        if self.round_streams:
            for i in np.flatnonzero(rows.identities==0):
                rows.prefix[i] = self.store.round_stream(rows.env_id[i],rows.match_id[i],
                                                        rows.round_index[i]).prefix
        return rows


def collect(args):
    opponents,opponent_meta = load_selected(args.selected_root)
    actors = {}
    for arm in ("control","norm16"):
        path = args.base/f"qknorm-short-ppo-2026-10-16/remote/runs/{arm}/endpoint.pt"
        actors[arm],record = load_actor(path)
        assert record["checkpoint_sha"] == ENDPOINTS[arm]
    folder = args.output/"results"/f"wave-{args.wave:02d}"
    folder.mkdir(parents=True,exist_ok=False)
    meta = dict(endpoints=ENDPOINTS,parent="a4bed2bfb4c3b8c1382a8ed5604e8e797307a37ea2088523dffb1243de6a58bd",
        parent_note="learner parent only; opponents are the selected old checkpoints",
        opponents=opponent_meta,opponent_selection_sha=matches.sha(args.selected_root/"selected.json"),
        waves=16,envs=32,seed=16102671,matches_per_condition=512,
        model_source_sha=matches.sha(__import__("train.history_model",fromlist=["x"]).__file__),
        runner_sha=matches.sha(__file__),torch=torch.__version__,device="cpu",
        sampling="T1 inverse CDF, independent per seed/seat streams; fixed tribute heuristic",
        reused_seed_note="same seeds as public predictability study; FULL trajectories must replay exactly")
    (folder/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    seed_table = np.random.default_rng(16102671).integers(1,2**63-1,size=(16,16))
    schedule = np.repeat(seed_table[args.wave],2).tolist()
    original = matches.EvalCollector
    matches.EvalCollector = AblationCollector
    first_round = {}
    try:
        with (folder/"waves.jsonl").open("w") as stream:
            for arm,view in (("control","full"),("control","round"),
                             ("norm16","full"),("norm16","round")):
                result = matches.run_wave(actors[arm],opponents,schedule,"cpu",view,20000)
                if view=="full":
                    reference = json.loads((args.selected_root/"raw"/f"wave-{args.wave:02d}"/
                                            f"{arm}-matches.json").read_text())
                    assert result["records"] == reference["records"],"FULL replay changed"
                    first_round[arm] = [r["first_round_sha"] for r in result["records"]]
                else:
                    assert first_round[arm] == [r["first_round_sha"] for r in result["records"]],"first-round FULL/ROUND mismatch"
                for record in result["records"]:
                    e = record["env"]
                    record["public_action_sha"] = AblationCollector.last.trace.hashes[e].hexdigest()
                    record["public_events"] = int(AblationCollector.last.trace.counts[e])
                result.update(wave=args.wave,arm=arm,view=view)
                stream.write(json.dumps(result)+"\n")
                stream.flush()
                print(json.dumps({k:v for k,v in result.items() if k!="records"}),flush=True)
    finally:
        matches.EvalCollector = original
    (folder/"complete.json").write_text(json.dumps(dict(complete=True,full_replay_parity=True))+"\n")


def summarize(args):
    selected = json.loads((args.selected_root/"selected.json").read_text())
    digest = matches.sha(args.selected_root/"selected.json")
    records = {arm:{view:{} for view in ("full","round")} for arm in ENDPOINTS}
    for wave in range(16):
        folder = args.output/"results"/f"wave-{wave:02d}"
        meta = json.loads((folder/"metadata.json").read_text())
        assert meta["opponents"] == selected["selected"] and meta["opponent_selection_sha"] == digest
        for line in (folder/"waves.jsonl").read_text().splitlines():
            row = json.loads(line)
            for record in row["records"]:
                records[row["arm"]][row["view"]][(record["seed"],record["team"])] = record
    result = analyze(args.output/"results")
    divergence = {}
    for arm,views in records.items():
        full,round_ = views["full"],views["round"]
        keys = sorted(full)
        assert len(keys)==512 and keys==sorted(round_)
        divergence[arm] = dict(matches=512,
            public_trajectory_different=sum(full[k]["public_action_sha"]!=round_[k]["public_action_sha"] for k in keys),
            public_trajectory_same=sum(full[k]["public_action_sha"]==round_[k]["public_action_sha"] for k in keys),
            full_replay_matches=512)
    result["trajectory_comparison"] = divergence
    result["opponent_alias_warning"] = "47241..47244 alias u2623/u20264/u35588/u86800, not the previous close snapshots"
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:result[key] for key in ("full_minus_round","interaction","trajectory_comparison")},indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--selected-root",type=Path,default=Path(".work/public-history-diverse-2026-10-16"))
    p.add_argument("--base",type=Path,default=Path(".work"))
    p.add_argument("--wave",type=int)
    p.add_argument("--summarize",action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.summarize:
        summarize(args)
    else:
        assert args.wave is not None and 0<=args.wave<16
        collect(args)


if __name__=="__main__":
    main()
