"""Run E18 inference on one structural tensor record."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .features import load_record, structure_to_feature_dict
from .model import build_model
from .sampling import sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Structural record (.npz)")
    parser.add_argument("--weights", type=Path, required=True, help="Full E18 state dictionary")
    parser.add_argument("--output", type=Path, required=True, help="Path for sampled token array (.npy)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inside-temperature", type=float, default=0.29)
    parser.add_argument("--outside-temperature", type=float, default=0.24)
    args = parser.parse_args()
    device = torch.device(args.device)
    features = structure_to_feature_dict(load_record(args.input), device)
    model = build_model(device)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    tokens = sample(
        model, features, samples=args.samples, steps=args.steps, seed=args.seed,
        inside=args.inside_temperature, outside=args.outside_temperature,
    )
    with args.output.open("xb") as stream:
        np.save(stream, tokens.cpu().numpy(), allow_pickle=False)
    print(f"Saved {args.samples} token sequences to {args.output}")


if __name__ == "__main__":
    main()
