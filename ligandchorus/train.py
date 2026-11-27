"""Train the E18 decoder with a frozen LigandMPNN encoder."""

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from .features import structure_to_feature_dict
from .model import build_model
from .records import RecordStore, targeted_training_entries, training_entries
from .validation import evaluate, load_validation


def save_weights(model, path, *, replace=False):
    with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(model.state_dict(), stream)
        except BaseException:
            stream.close()
            temporary.unlink()
            raise
    try:
        if replace:
            temporary.replace(path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def training_features(source, entry, target_labels=None):
    features = structure_to_feature_dict(source.read(entry), "cpu")
    if target_labels is not None:
        labels = features["chain_labels"]
        sequence = features["S"]
        known = features["mask"].bool() & (sequence >= 0) & (sequence < 20)
        missing = set(target_labels) - set(labels[known].tolist())
        if missing:
            raise ValueError(f"{entry['key']}: no valid known residues on target chains {sorted(missing)}")
        features["loss_mask"] = known & torch.isin(labels, torch.tensor(target_labels))
    return features


def train(model, features, device, steps, batch_size, learning_rate, seed, *,
          validation=None, validate_every=1000, validation_seed=0, best_output=None):
    """Match the original optimizer and uniform sampling of training assemblies."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    rng = np.random.RandomState(seed)
    best_score = None
    for step in range(1, steps + 1):
        indices = rng.choice(len(features), size=min(batch_size, len(features)), replace=False)
        losses = []
        for index in indices:
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in features[index].items()}
            losses.append(model.compute_loss(batch, batch["S"], loss_mask=batch.get("loss_mask")))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(f"Step {step}/{steps}: loss = {loss.item():.4f}", flush=True)
        if validation is not None and (step % validate_every == 0 or step == steps):
            metrics = evaluate(model, validation, device, seed=validation_seed)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"Non-finite validation metrics at step {step}: {metrics}")
            score = metrics["pocket_recovery"]
            improved = best_score is None or score > best_score
            if improved:
                save_weights(model, best_output, replace=best_score is not None)
                best_score = score
            print("Validation " + json.dumps({"step": step, **metrics, "saved_best": improved}),
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL of NPZ paths or PyTorch shard entries")
    parser.add_argument("--shard-root", type=Path, help="Directory for shard paths (defaults to the manifest directory)")
    parser.add_argument("--encoder-weights", type=Path, required=True, help="Official LigandMPNN checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Path for the full E18 state dictionary")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--structures", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-targets", type=Path,
                        help="JSONL of selected split=train keys and their target chain_labels")
    parser.add_argument("--validation-manifest", type=Path,
                        help="JSONL of target chains and pockets on records with split=valid")
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--best-output", type=Path,
                        help="Best full E18 state dictionary (defaults to best.pt beside --output)")
    args = parser.parse_args()
    if min(args.structures, args.steps, args.batch_size) < 1:
        parser.error("structures, steps, and batch-size must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    best = None
    if args.validation_manifest is not None:
        if args.validate_every < 1:
            parser.error("validate-every must be positive")
        best = args.best_output or args.output.with_name("best.pt")
        if best.resolve() == args.output.resolve() or best.exists():
            parser.error(f"Best checkpoint must be a separate, unused path: {best}")
    elif args.best_output is not None:
        parser.error("--best-output requires --validation-manifest")
    labels_by_key = {}
    if args.train_targets is None:
        records = training_entries(args.manifest, args.structures)
    else:
        records, labels_by_key = targeted_training_entries(
            args.manifest, args.train_targets, args.structures,
        )
    if not records:
        parser.error("The manifest contains no training records")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_model(device)
    state = torch.load(args.encoder_weights, map_location="cpu", weights_only=True)
    model.ligandmpnn.load_state_dict(state["model_state_dict"])
    model.train()
    model.ligandmpnn.eval()
    source = RecordStore(args.manifest, args.shard_root)
    validation = None
    if best is not None:
        validation = load_validation(args.validation_manifest, args.manifest, source, records)
        print(f"Validating {len(validation)} structures every {args.validate_every} steps "
              f"and at the final step with seed {args.validation_seed}", flush=True)
    print(f"Loading {len(records)} training assemblies", flush=True)
    features = [training_features(source, item, labels_by_key.get(item.get("key"))) for item in records]
    if labels_by_key:
        target_count = sum(int(item["loss_mask"].sum()) for item in features)
        print(f"Target-chain supervision enabled for {target_count} known residues", flush=True)
    del source
    train(model, features, device, args.steps, args.batch_size, args.learning_rate, args.seed,
          validation=validation, validate_every=args.validate_every,
          validation_seed=args.validation_seed, best_output=best)
    save_weights(model, args.output)


if __name__ == "__main__":
    main()
