"""Score paired fixed-backbone sequence outputs for one frozen test panel."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import numpy as np


SHARED_MODELS = ("e18", "ligandmpnn")
EXTERNAL_BASELINES = ("lasermpnn", "adflip")
HISTORICAL_BASELINES = ("proteinmpnn", "rfd3_direct")
AA = "ACDEFGHIKLMNPQRSTVWY"


def read_rows(path):
    payload = path.read_bytes()
    rows = {}
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key") if isinstance(row, dict) else None
        if not isinstance(key, str) or not key.strip() or key in rows:
            raise ValueError(f"{path}:{line_number}: expected a distinct case key")
        rows[key] = row
    if not rows:
        raise ValueError(f"{path}: no cases")
    return rows, hashlib.sha256(payload).hexdigest()


def score_panel(references, predictions, samples):
    if samples < 2:
        raise ValueError("At least two samples per case are needed for diversity")
    for model_name, rows in predictions.items():
        missing = references.keys() - rows.keys()
        unexpected = rows.keys() - references.keys()
        if missing or unexpected:
            raise ValueError(f"{model_name}: missing {sorted(missing)}, unexpected {sorted(unexpected)}")

    pairs = np.triu_indices(samples, k=1)
    cases = []
    for key, reference in references.items():
        native = reference.get("native_sequence")
        residue_ids = reference.get("residue_ids")
        pocket = reference.get("pocket_positions")
        if not isinstance(native, str) or not native or set(native) - set(AA):
            raise ValueError(f"{key}: expected a canonical native target sequence")
        if (not isinstance(residue_ids, list) or len(residue_ids) != len(native)
                or any(not isinstance(identifier, str) or not identifier for identifier in residue_ids)
                or len(set(residue_ids)) != len(residue_ids)):
            raise ValueError(f"{key}: expected a unique residue identifier per target position")
        if (not isinstance(pocket, list) or not pocket
                or any(type(index) is not int or not 0 <= index < len(native) for index in pocket)
                or len(set(pocket)) != len(pocket)):
            raise ValueError(f"{key}: expected distinct zero-based pocket positions on the target")

        native_tokens = np.frombuffer(native.encode("ascii"), dtype=np.uint8)
        scores = {}
        for model_name, rows in predictions.items():
            row = rows[key]
            if row.get("residue_ids") != residue_ids:
                raise ValueError(f"{key}/{model_name}: target residue order differs from the reference")
            sequences = row.get("sequences")
            if (not isinstance(sequences, list) or len(sequences) != samples
                    or any(not isinstance(seq, str) or len(seq) != len(native)
                           or set(seq) - set(AA) for seq in sequences)):
                raise ValueError(f"{key}/{model_name}: expected {samples} complete canonical sequences")
            tokens = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
            tokens = tokens.reshape(samples, len(native))
            matches = tokens == native_tokens
            pocket_tokens = tokens[:, pocket]
            scores[model_name] = {
                "chain_recovery": float(matches.mean()),
                "pocket_recovery": float(matches[:, pocket].mean()),
                "pocket_diversity": float((pocket_tokens[pairs[0]] != pocket_tokens[pairs[1]]).mean()),
            }
        cases.append({"key": key, "target_positions": len(native),
                      "pocket_positions": len(pocket), "models": scores})

    summary = {}
    for model_name in predictions:
        summary[model_name] = {}
        for metric in cases[0]["models"][model_name]:
            values = [case["models"][model_name][metric] for case in cases]
            summary[model_name][metric] = {"median": median(values), "mean": mean(values)}
    return {"cases": cases, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    for name in SHARED_MODELS:
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in EXTERNAL_BASELINES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    supplementary = parser.add_argument_group("supplementary comparisons")
    supplementary.add_argument("--rfd4-proteina", type=Path, help="add RFD4 to a separate supplementary comparison")
    historical = parser.add_argument_group("historical comparisons")
    for name in HISTORICAL_BASELINES:
        historical.add_argument(f"--{name.replace('_', '-')}", type=Path)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.proteinmpnn is not None:
        if args.rfd4_proteina is not None or any(getattr(args, name) is not None for name in EXTERNAL_BASELINES):
            parser.error("Historical and current baseline files must be scored in separate runs")
        model_names = (*SHARED_MODELS, *HISTORICAL_BASELINES)
    else:
        if args.rfd3_direct is not None:
            parser.error("--rfd3-direct is only available with --proteinmpnn for historical runs")
        missing = [f"--{name.replace('_', '-')}" for name in EXTERNAL_BASELINES if getattr(args, name) is None]
        if missing:
            parser.error(f"Current comparisons require {', '.join(missing)}")
        model_names = (*SHARED_MODELS, *EXTERNAL_BASELINES, "rfd4_proteina")
    references, reference_hash = read_rows(args.references)
    predictions, prediction_hashes = {}, {}
    for model_name in model_names:
        path = getattr(args, model_name)
        if path is not None:
            predictions[model_name], prediction_hashes[model_name] = read_rows(path)
    report = score_panel(references, predictions, args.samples)
    report.update({"structures": len(references), "samples_per_structure": args.samples,
                   "reference_sha256": reference_hash, "prediction_sha256": prediction_hashes})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
