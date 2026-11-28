"""Run the locked benchmark with per-case checkpoints and score the paired panels."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import mkdtemp

from ligandchorus.benchmark import read_rows, score_panel
from ligandchorus.benchmark_outputs import normalize


MODELS = ("e18", "ligandmpnn", "lasermpnn", "adflip")
BASELINE_ROOT = Path("/data/yxhuang/enzyme_design/建模/03_LigandMPNN/复现准备/官方代码")
DEPENDENCIES = Path("/data_raid/yxhuang/ligandmpnn_data_audit/deps")
ENV = Path("/data/yxhuang/enzyme_design/envs/lasermpnn/bin/python")
E18_CODE = Path("/data_raid/yxhuang/ligandchorus_e18_formal_20261127.HU2EG6/code")
SEED_BASE = 20261127


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def call(log, command, imports, cwd=None):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [*imports, DEPENDENCIES])))
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([str(x) for x in command], cwd=cwd, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def run_case(case, row, out, config, server, seed, device):
    from types import SimpleNamespace

    pdb = case / row["pdb_name"]
    if digest(pdb) != row["input_sha256"]:
        raise ValueError("Backbone hash differs from frozen input")
    if digest(case / "alignment.json") != row["alignment_sha256"]:
        raise ValueError("Residue mapping hash differs from frozen input")
    if digest(case / row["npz_name"]) != row["e18_input_sha256"]:
        raise ValueError("E18 NPZ hash differs from frozen input")
    samples = config["samples"]
    weights = {m: config["models"][m]["checkpoint"]["server_path"] for m in MODELS}
    e18 = config["models"]["e18"]
    ligandmpnn = config["models"]["ligandmpnn"]
    laser = config["models"]["lasermpnn"]
    adflip = config["models"]["adflip"]
    final = out
    if final.exists():
        if not all((final / "normalized" / f"{model}.jsonl").is_file() for model in MODELS):
            raise ValueError(f"Incomplete normalized records in {final}")
        return
    out = Path(mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    normalized = out / "normalized"
    npy = out / "e18.npy"
    call(out / "e18.log", [ENV, "-B", "-m", "ligandchorus.infer", "--input", case / row["npz_name"],
         "--weights", weights["e18"], "--output", npy, "--device", device, "--samples", samples,
         "--steps", e18["max_steps"], "--seed", seed, "--inside-temperature", e18["inside_temperature"],
         "--outside-temperature", e18["outside_temperature"]], [E18_CODE])
    lmp_dir = out / "ligandmpnn"
    lmp_root = BASELINE_ROOT / "LigandMPNN"
    call(out / "ligandmpnn.log", [ENV, "-B", lmp_root / "run.py", "--model_type", "ligand_mpnn",
         "--checkpoint_ligand_mpnn", weights["ligandmpnn"], "--pdb_path", pdb, "--out_folder", lmp_dir,
         "--temperature", ligandmpnn["temperature"], "--seed", seed,
         "--batch_size", samples, "--number_of_batches", 1, "--omit_AA", "X"], [lmp_root])
    laser_dir = out / "lasermpnn"
    laser_root = BASELINE_ROOT / "LASErMPNN"
    laser_flags = ["--noncanonical_aa_ligand"] if row["lasermpnn_noncanonical_ligand"] else []
    call(out / "lasermpnn.log", [ENV, "-B", Path(__file__).with_name("e18_formal_benchmark_worker.py"),
         "lasermpnn", seed, laser_root / "run_batch_inference.py", pdb, laser_dir, samples,
         "--designs_per_batch", laser["batch_size"], "--model_weights_path", weights["lasermpnn"],
         "--sequence_temp", laser["sequence_temperature"], "--first_shell_sequence_temp", laser["first_shell_temperature"],
         "--disabled_residues", "X", "--device", device, "--output_fasta", "--silent", *laser_flags],
         [BASELINE_ROOT, laser_root])
    adflip_path = out / "adflip.json"
    call(out / "adflip.log", [ENV, "-B", Path(__file__).with_name("e18_formal_benchmark_worker.py"),
         "adflip", seed, pdb, weights["adflip"], adflip_path, out / "adflip_work", device, samples,
         adflip["max_steps"], adflip["threshold"]],
         [server / "adflip", server, server / "adflip_deps"], cwd=server / "adflip")
    normalize(SimpleNamespace(
        alignment=case / "alignment.json", backbone=pdb, e18=npy,
        ligandmpnn=lmp_dir / "seqs" / f"{pdb.stem}.fa", lasermpnn=laser_dir, adflip=adflip_path,
        output=normalized, samples=samples, proteinmpnn=None, rfd3=None,
    ))
    out.rename(final)


def rows_for_development(server):
    for name in ("8d59", "food"):
        case = server / "fixtures" / name
        mapping = json.loads(read_json(case / "alignment.json"))
        reference = json.loads(read_json(case / "references.jsonl"))
        yield {
            "key": mapping["key"], "test_group": "enzyme" if name == "8d59" else "nonenzyme",
            "family_id": name, "pdb_name": "smoke_backbone_only.pdb",
            "npz_name": "blinded_e18_input.npz", "input_sha256": mapping["backbone_sha256"],
            "alignment_sha256": digest(case / "alignment.json"),
            "e18_input_sha256": digest(case / "blinded_e18_input.npz"),
            "lasermpnn_noncanonical_ligand": name == "8d59",
        }, case, reference


def rows_for_panel(config):
    panel = config["panel"]
    if digest(panel["path"]) != panel["sha256"]:
        raise ValueError("Panel hash differs from frozen manifest")
    refs = {}
    for group, artifact in config["references"].items():
        if digest(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"{group} reference hash differs")
        refs[group], _ = read_rows(Path(artifact["path"]))
        if len(refs[group]) != artifact["structures"]:
            raise ValueError(f"{group} reference count differs")
    rows = [json.loads(line) for line in Path(panel["path"]).read_text().splitlines()]
    by_group = {}
    for group, references in refs.items():
        keys = [r["key"] for r in rows if r["test_group"] == group]
        if len(keys) != len(references) or set(keys) != references.keys():
            raise ValueError(f"{group} case set differs")
        by_group[group] = len(keys)
    if sum(by_group.values()) != len(rows):
        raise ValueError("Panel includes an unknown group")
    for row in rows:
        pdb = Path(row["input_path"])
        yield dict(row, pdb_name=pdb.name, npz_name="e18_input.npz"), pdb.parent, refs[row["test_group"]][row["key"]]


def score_cases(cases, out, samples):
    from statistics import mean, median
    import numpy as np

    scores = {}
    for group in sorted({r["test_group"] for r, _, _ in cases}):
        rows = [(r, ref) for r, _, ref in cases if r["test_group"] == group]
        reference = {r["key"]: ref for r, ref in rows}
        predictions = {m: {} for m in MODELS}
        for row, _ in rows:
            for model in MODELS:
                pred = json.loads(read_json(out / row["key"] / "normalized" / (model + ".jsonl")))
                if pred["key"] != row["key"]:
                    raise ValueError("Prediction record key differs")
                predictions[model][row["key"]] = pred
        result = score_panel(reference, predictions, samples)
        pairs = np.triu_indices(samples, k=1)
        per_case = {r["key"]: r for r in result["cases"]}
        for model in MODELS:
            values = []
            for key, row in predictions[model].items():
                X = np.asarray([list(sequence) for sequence in row["sequences"]])
                diversity = float((X[pairs[0]] != X[pairs[1]]).mean())
                per_case[key]["models"][model]["chain_diversity"] = diversity
                values.append(diversity)
            result["summary"][model]["chain_diversity"] = {"mean": mean(values), "median": median(values)}
        families = {}
        for row, _ in rows:
            families.setdefault(row["family_id"], []).append(per_case[row["key"]]["models"])
        result["family_summary"] = {
            family: {"structures": len(items),
                     "models": {model: {metric: mean(x[model][metric] for x in items)
                                        for metric in items[0][model]} for model in MODELS}}
            for family, items in sorted(families.items())
        }
        result["structures"] = len(rows)
        scores[group] = result
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("development", "formal"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(read_json(args.config))
    for model, settings in config["models"].items():
        checkpoint = settings["checkpoint"]
        if digest(checkpoint["server_path"]) != checkpoint["sha256"]:
            raise ValueError(f"{model} checkpoint hash differs from the lock")
    for name, artifact in config["runtime_artifacts"].items():
        if digest(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"{name} runtime hash differs from the lock")
    server = Path(config["server_root"])
    report_path = args.output / "progress.json"
    config_sha = digest(args.config)
    if args.mode == "formal":
        pilot = json.loads(read_json(Path(config["development_report"])))
        if pilot["status"] != "complete" or pilot["mode"] != "development" or pilot["config_sha256"] != config_sha or pilot["completed_cases"] != 2:
            raise ValueError("Matching two-case development run must complete before formal sampling")
    previous = json.loads(read_json(report_path)) if report_path.exists() else None
    if previous and (previous["config_sha256"] != config_sha or previous["mode"] != args.mode):
        raise ValueError("Output already belongs to another configuration or run mode")
    cases = list(rows_for_development(server) if args.mode == "development" else rows_for_panel(config))
    progress = {
        "mode": args.mode, "config_sha256": config_sha, "status": "running", "completed_cases": 0,
        "total_cases": len(cases), "group_counts": dict(Counter(r["test_group"] for r, _, _ in cases)),
        "samples_per_model": config["samples"], "models": MODELS,
    }
    atomic_json(report_path, progress)
    for index, (row, case, _) in enumerate(cases):
        try:
            run_case(case, row, args.output / row["key"], config, server, SEED_BASE + index, config["device"])
        except Exception as error:
            atomic_json(report_path, dict(progress, status="failed", failed_case=row["key"], error=str(error)))
            raise
        progress["completed_cases"] += 1
        atomic_json(report_path, progress)
        print(f"{args.mode}: completed {index + 1}/{len(cases)}", flush=True)
    atomic_json(args.output / "scores.json", score_cases(cases, args.output, config["samples"]))
    atomic_json(report_path, dict(progress, status="complete"))


if __name__ == "__main__":
    main()
