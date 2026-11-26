"""Train the E18 decoder with a frozen LigandMPNN encoder."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .features import load_record, structure_to_feature_dict
from .model import build_model


def train(model, features, device, steps, batch_size, learning_rate, seed):
    """Match the original optimizer and uniform sampling of training assemblies."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    rng = np.random.RandomState(seed)
    for step in range(1, steps + 1):
        indices = rng.choice(len(features), size=min(batch_size, len(features)), replace=False)
        losses = []
        for index in indices:
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in features[index].items()}
            losses.append(model.compute_loss(batch, batch["S"]))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(f"Step {step}/{steps}: loss = {loss.item():.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL of training NPZ paths")
    parser.add_argument("--encoder-weights", type=Path, required=True, help="Official LigandMPNN checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Path for the full E18 state dictionary")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--structures", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.structures, args.steps, args.batch_size) < 1:
        parser.error("structures, steps, and batch-size must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    with args.manifest.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()][:args.structures]
    if not records:
        parser.error("The training manifest is empty")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_model(device)
    state = torch.load(args.encoder_weights, map_location="cpu", weights_only=True)
    model.ligandmpnn.load_state_dict(state["model_state_dict"])
    model.train()
    model.ligandmpnn.eval()
    features = [
        structure_to_feature_dict(load_record(args.manifest.parent / item["path"]), "cpu")
        for item in records
    ]
    train(model, features, device, args.steps, args.batch_size, args.learning_rate, args.seed)
    with args.output.open("xb") as stream:
        torch.save(model.state_dict(), stream)


if __name__ == "__main__":
    main()
