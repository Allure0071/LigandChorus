"""Matched plan-1 training, validation-only temperature selection and blinded E18 scoring."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
PEER = REPO / "reproduction/2026-11-27"
sys.path[:0] = [str(REPO), str(PEER / "scripts")]
from calibrate import digest, metrics, summarize
from e18_formal_benchmark_runner import SEED_BASE, rows_for_panel
from ligandchorus.benchmark import AA, score_panel
from ligandchorus.features import load_record, structure_to_feature_dict
from ligandchorus.model import build_model
from ligandchorus.quota_masking import masked_training_input, quota_trajectory
from ligandchorus.records import RecordStore, targeted_training_entries
from ligandchorus.sampling import regional_temperature, sample
from ligandchorus.train import save_weights, training_features
from ligandchorus.validation import load_validation

MODES = ("quota", "legacy")
DEVICE = torch.device("cuda:0")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def gpu(features):
    return {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in features.items()}


@torch.no_grad()
def blind_encode(model, f):
    return model.ligandmpnn.encode({**f, "S": torch.full_like(f["S"], 20)})


@torch.inference_mode()
def draw(model, f, encoded, mode, seed, inside, outside):
    # Match the archived inference context: generate across the assembly, score its assigned target.
    f = {**f, "S": torch.full_like(f["S"], 20)}
    if mode == "legacy":
        with patch.object(model.ligandmpnn, "encode", return_value=encoded):
            return sample(model, f, samples=10, steps=20, seed=seed, inside=inside, outside=outside)
    mask = f["mask"].expand(10, -1)
    enc = tuple(x.expand(10, *x.shape[1:]) for x in encoded)
    design = mask.bool() & f["chain_mask"].bool()
    traces = quota_trajectory(model, enc, mask, design, regional_temperature(f, inside, outside),
                              torch.Generator(device=DEVICE).manual_seed(seed))
    expected = (torch.arange(6, device=DEVICE)[:, None] * design.sum(-1) + 4) // 5
    if not torch.equal(traces.ne(20).sum(-1), expected):
        raise RuntimeError("Incomplete quota trajectory")
    return traces[-1]


def train(out, model, store, entries, labels, validation, cfg, smoke):
    steps = 2 if smoke else cfg["optimizer_steps"]
    every = 1 if smoke else cfg["validate_every"]
    pool = []
    for entry in entries:
        f = gpu(training_features(store, entry, labels[entry["key"]]))
        enc = blind_encode(model, f)
        pool.append((tuple(t.cpu() for t in enc), f["S"].cpu(), f["mask"].cpu(), f["loss_mask"].cpu()))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=0.01)
    rng = np.random.RandomState(cfg["training_seed"])
    best = {mode: {"score": -1, "step": 0} for mode in MODES}
    model.train()
    for step in range(1, steps + 1):
        losses = []
        for index in rng.choice(len(pool), min(len(pool), cfg["batch_size"]), replace=False):
            enc, native, mask, targets = pool[index]
            native = native.to(DEVICE)
            stage = int(torch.randint(1, 6, (), device=DEVICE))
            noisy, t, hidden = masked_training_input(native, targets.to(DEVICE), stage)
            loss = model.compute_substitution_loss({"mask": mask.to(DEVICE), "chain_mask": hidden},
                         native, noisy, t, encoded=tuple(x.to(DEVICE) for x in enc))
            losses.append(loss)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(json.dumps({"step": step, "ce": loss.item()}), flush=True)
        if step % every == 0 or step == steps:
            model.eval()
            for mode in MODES:
                values = []
                for i, (case, enc) in enumerate(validation):
                    f = gpu(case.features)
                    tokens = draw(model, f, tuple(x.to(DEVICE) for x in enc), mode,
                                  i + cfg["validation_seed"], 0.29, 0.24)
                    values.append((tokens[:, case.pocket_positions] == f["S"][:, case.pocket_positions]).float().mean().item())
                score = median(values)
                if score > best[mode]["score"]:
                    save_weights(model, out / f"{mode}_best.pt", replace=best[mode]["step"] > 0)
                    best[mode] = {"score": score, "step": step}
                with (out / "validation.jsonl").open("a") as stream:
                    stream.write(json.dumps({"mode": mode, "step": step, "median_pocket": score, "per_structure": values}) + "\n")
            dump(out / "selection.json", best)
            model.train()
    save_weights(model, out / "last.pt")


@torch.inference_mode()
def calibrate(out, model, validation, assets, smoke):
    protocol = read(PEER / "configs/calibration.json")
    ar_file = Path(assets["directories"]["calibration_run"]) / "run/structure_metrics.jsonl"
    reference = [json.loads(s)["ar_reference"] for s in ar_file.read_text().splitlines()]
    peer_ar = read(PEER / "results/calibration_summary.json")["official_ligandmpnn"]
    assert len(reference) == 128 and all(abs(mean(r[k] for r in reference) - v) < 1e-10 for k,v in peer_ar.items())
    grid = protocol["e18_temperature_grid"]
    configs = {f"{a:.2f}_{b:.2f}": (a,b) for a in grid for b in grid}
    if smoke:
        configs = {"0.29_0.24": (0.29, 0.24)}
    pairs = torch.triu_indices(10, 10, 1, device=DEVICE)
    locks = {}
    for mode in MODES:
        weights = out / f"{mode}_best.pt"
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval()
        scores = defaultdict(list)
        scores["ar_reference"] = reference[:len(validation)]
        for i, (case, enc) in enumerate(validation):
            f = gpu(case.features)
            enc = tuple(x.to(DEVICE) for x in enc)
            for name, (a,b) in configs.items():
                tokens = draw(model, f, enc, mode, protocol["base_seed"] + i, a, b)
                scores[name].append(metrics(tokens, f["S"], case.chain_mask[0].to(DEVICE),
                                             case.pocket_positions.to(DEVICE), pairs))
        result = summarize(scores, "0.29_0.24", configs)
        dump(out / f"{mode}_calibration.json", result)
        locks[mode] = {**result["selected"], "weight_sha256": digest(weights)}
    dump(out / "temperature_lock.json", locks)


@torch.inference_mode()
def score(out, model, smoke):
    config = read(PEER / "configs/resident_benchmark.json")
    locks = read(out / "temperature_lock.json")
    cases = list(rows_for_panel(config))
    if smoke:
        cases = cases[:2]
    inputs = []
    for row, folder, ref in cases:
        path = folder / row["npz_name"]
        mapping_path = folder / "alignment.json"
        if digest(path) != row["e18_input_sha256"] or digest(mapping_path) != row["alignment_sha256"]:
            raise ValueError("Frozen test input mismatch")
        mapping = read(mapping_path)
        if mapping["residue_ids"] != ref["residue_ids"]:
            raise ValueError("Target alignment differs")
        f = structure_to_feature_dict(load_record(path), DEVICE)
        assert f["S"].eq(AA.index("G")).all()
        chain_label = mapping["input_chain_order"].index(mapping["target_chain"])
        idx = (f["chain_labels"][0] == chain_label).nonzero().flatten()
        assert idx.numel() == len(mapping["input_residue_ids"]) == len(ref["residue_ids"])
        inputs.append((row, ref, idx, f, model.ligandmpnn.encode(f)))
    rows = {}
    pairs = np.triu_indices(10, k=1)
    for mode in (*MODES, "archived_e18_replay"):
        if mode == "archived_e18_replay":
            e18 = config["models"]["e18"]
            w = Path(e18["checkpoint"]["server_path"])
            lock = {"inside": e18["inside_temperature"], "outside": e18["outside_temperature"],
                    "weight_sha256": e18["checkpoint"]["sha256"]}
        else:
            w = out / f"{mode}_best.pt"
            lock = locks[mode]
        assert digest(w) == lock["weight_sha256"]
        model.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        model.eval()
        for i,(row, ref, idx, f, enc) in enumerate(inputs):
            decoder = "legacy" if mode == "archived_e18_replay" else mode
            tokens = draw(model, f, enc, decoder, SEED_BASE + i, lock["inside"], lock["outside"])[:,idx].cpu().numpy()
            if not np.all((tokens >= 0) & (tokens < 20)):
                raise ValueError(f"{row['key']}: unfilled target residue")
            seqs = ["".join(AA[t] for t in s) for s in tokens]
            key = row["key"]
            pred = {"key":key, "residue_ids":ref["residue_ids"], "sequences":seqs}
            values = score_panel({key:ref}, {mode:{key:pred}}, 10)["cases"][0]["models"][mode]
            values["chain_diversity"] = float((tokens[pairs[0]] != tokens[pairs[1]]).mean())
            rec = rows.setdefault(key, {"group":row["test_group"], "family":row["family_id"], "metrics": {}})
            rec["metrics"][mode] = values
    baseline = read(PEER / "results/scores.json")
    summary = {}
    for group in sorted({r["group"] for r in rows.values()}):
        comparisons = []
        for r in baseline[group]["cases"]:
            if r["key"] in rows:
                values = rows[r["key"]]["metrics"]
                values.update({k:r["models"][k] for k in ("e18","ligandmpnn")})
                comparisons.append(values)
        summary[group] = {m:{k:median(v[m][k] for v in comparisons) for k in comparisons[0][m]}
                          for m in comparisons[0]}
    dump(out / "scores.json", {"cases":rows, "medians":summary})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["train","calibrate","score"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.out
    cfg = read(PEER / "configs/training.json")
    assets = read(PEER / "assets.json")
    files = assets["files"]
    torch.manual_seed(cfg["training_seed"])
    model = build_model(DEVICE)
    encoder = Path(files["official_ligandmpnn"]["path"])
    assert digest(encoder) == files["official_ligandmpnn"]["sha256"]
    model.ligandmpnn.load_state_dict(torch.load(encoder, map_location="cpu", weights_only=True)["model_state_dict"])
    if args.phase == "score":
        score(out, model, args.smoke)
        return
    manifest,targets,panel = [Path(files[k]["path"]) for k in ("training_manifest","enzyme_training_targets","validation_panel")]
    for p,k in zip((manifest,targets,panel),("training_manifest","enzyme_training_targets","validation_panel")):
        assert digest(p)==files[k]["sha256"]
    entries, labels = targeted_training_entries(manifest, targets, cfg["training_assemblies"])
    assert len(entries)==5044
    store = RecordStore(manifest, Path(assets["directories"]["training_shard_root"]))
    cases = load_validation(panel, manifest, store, entries)
    assert len(cases)==128
    if args.smoke:
        cases, entries = cases[:2], entries[:4]
    validation = [(case, tuple(x.cpu() for x in blind_encode(model, gpu(case.features)))) for case in cases]
    if args.phase == "train":
        out.mkdir()
        train(out, model, store, entries, labels, validation, cfg, args.smoke)
    else:
        calibrate(out, model, validation, assets, args.smoke)


if __name__ == "__main__":
    main()
