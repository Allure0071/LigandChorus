"""Calibrate a locked E18 checkpoint against published LigandMPNN on validation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import time
from unittest.mock import patch

import torch

from ligandchorus.model import build_model
from ligandchorus.records import RecordStore, targeted_training_entries
from ligandchorus.sampling import sample
from ligandchorus.validation import load_validation


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def metrics(tokens, native, chain_mask, pocket, pairs):
    left, right = pairs
    differences = tokens[left] != tokens[right]
    matches = tokens == native
    return {
        "chain_diversity": differences[:, chain_mask].float().mean().item(),
        "pocket_diversity": differences[:, pocket].float().mean().item(),
        "chain_recovery": matches[:, chain_mask].float().mean().item(),
        "pocket_recovery": matches[:, pocket].float().mean().item(),
    }


def summarize(scores, old_key, configs):
    aggregates = {
        name: {metric: mean(row[metric] for row in rows) for metric in rows[0]}
        for name, rows in scores.items()
    }
    ar = aggregates["ar_reference"]
    diversity_keys = ("chain_diversity", "pocket_diversity")
    rows = []
    for name, (inside, outside) in configs.items():
        values = aggregates[name]
        gaps = {metric: 100 * (values[metric] - ar[metric]) for metric in diversity_keys}
        deviations = [abs(gap) for gap in gaps.values()]
        rows.append({
            "name": name,
            "inside": inside,
            "outside": outside,
            **values,
            "diversity_gaps_pp": gaps,
            "max_absolute_gap_pp": max(deviations),
            "sum_absolute_gaps_pp": sum(deviations),
            "within_one_pp": max(deviations) <= 1.0,
        })

    def rank(row):
        return (row["max_absolute_gap_pp"], row["sum_absolute_gaps_pp"], row["inside"], row["outside"])

    rows.sort(key=rank)
    return {
        "official_ligandmpnn": ar,
        "selected": rows[0],
        "previous": next(row for row in rows if row["name"] == old_key),
        "best_uniform": min((row for row in rows if row["inside"] == row["outside"]), key=rank),
        "ranking": rows,
    }


@torch.inference_mode()
def run(args):
    start = time.monotonic()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    root = args.formal
    manifest = root / "inputs/structures.jsonl"
    panel = root / "inputs/validation.jsonl"
    train_targets = root / "inputs/train_targets.jsonl"
    e18_weights = root / "formal/best.pt"
    ar_weights = Path((root / "inputs/encoder_weights.path").read_text().strip())
    checks = {
        "structure_manifest_sha256": manifest,
        "validation_manifest_sha256": panel,
        "e18_checkpoint_sha256": e18_weights,
        "ligandmpnn_checkpoint_sha256": ar_weights,
        "sampling_source_sha256": root / "code/ligandchorus/sampling.py",
    }
    verified = {key: digest(path) for key, path in checks.items()}
    if any(value != protocol[key] for key, value in verified.items()):
        raise ValueError("Recorded input hashes do not match the frozen protocol.")

    records, _ = targeted_training_entries(manifest, train_targets, None)
    cases = load_validation(panel, manifest, RecordStore(manifest, args.shard_root), records)
    if len(cases) != protocol["validation_structures"]:
        raise ValueError("Unexpected validation panel size.")

    def key(inside, outside):
        return f"e18_{inside:.2f}_{outside:.2f}"

    previous = tuple(protocol["previous_e18_temperatures"])
    grid = protocol["e18_temperature_grid"]
    configs = {key(inside, outside): (inside, outside) for inside in grid for outside in grid}
    configs[key(*previous)] = previous
    if args.smoke:
        cases = cases[:2]
        configs = {key(*previous): previous, key(0.24, 0.24): (0.24, 0.24)}

    device = torch.device("cuda:0")
    e18 = build_model(device).eval()
    e18.load_state_dict(torch.load(e18_weights, map_location="cpu", weights_only=True))
    ar = build_model(device).ligandmpnn.eval()
    ar.load_state_dict(torch.load(ar_weights, map_location="cpu", weights_only=True)["model_state_dict"])

    scores = defaultdict(list)
    sample_count = protocol["sample_count"]
    step_limit = protocol["e18_max_steps"]
    pairs = torch.triu_indices(sample_count, sample_count, offset=1, device=device)
    args.output.mkdir()
    print(f"Validation cases: {len(cases)}, E18 settings: {len(configs)}, GPU: {torch.cuda.get_device_name(device)}", flush=True)

    with (args.output / "structure_metrics.jsonl").open("x", encoding="utf-8") as output:
        for offset, case in enumerate(cases):
            features = {name: value.to(device) if torch.is_tensor(value) else value for name, value in case.features.items()}
            native = features["S"]
            features["S"] = torch.full_like(native, e18.mask_token)
            seed = protocol["base_seed"] + offset
            chain_mask = case.chain_mask[0].to(device)
            pocket = case.pocket_positions.to(device)
            order = torch.randn(
                (sample_count, native.shape[-1]), device=device,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            torch.manual_seed(seed)
            ar_input = dict(features, randn=order, batch_size=sample_count, temperature=protocol["ligandmpnn_temperature"])
            tokens = ar.sample(ar_input)["S"]
            row = {"structure": offset + 1, "ar_reference": metrics(tokens, native, chain_mask, pocket, pairs)}

            encoded = e18.ligandmpnn.encode(features)
            check = sample(e18, features, samples=sample_count, steps=step_limit, seed=seed,
                           inside=previous[0], outside=previous[1]) if offset == 0 else None
            with patch.object(e18.ligandmpnn, "encode", return_value=encoded):
                for name, (inside, outside) in configs.items():
                    tokens = sample(e18, features, samples=sample_count, steps=step_limit, seed=seed,
                                    inside=inside, outside=outside)
                    if offset == 0 and (inside, outside) == previous and not torch.equal(tokens, check):
                        raise AssertionError("Cached inference did not reproduce the production sampler.")
                    row[name] = metrics(tokens, native, chain_mask, pocket, pairs)
            for name, item in row.items():
                if name != "structure":
                    scores[name].append(item)
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
            output.flush()

            completed = offset + 1
            if completed == 1 or completed % 8 == 0 or completed == len(cases):
                print(f"{completed}/{len(cases)} structures scored in {time.monotonic()-start:.1f}s", flush=True)

    results = summarize(scores, key(*previous), configs)
    results.update({
        "validation_structures": len(cases),
        "temperature_candidates": len(configs),
        "elapsed_seconds": round(time.monotonic()-start, 1),
        "input_sha256": verified,
        "protocol_sha256": digest(args.protocol),
        "runner_sha256": digest(Path(__file__)),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "smoke": args.smoke,
    })
    write_json(args.output / "summary.json", results)
    print(json.dumps({name:results[name] for name in ("selected", "previous", "elapsed_seconds")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
