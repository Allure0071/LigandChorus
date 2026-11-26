"""Frozen-encoder, five-stage native identity reconstruction on the archived E18 panel."""

import argparse
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .features import structure_to_feature_dict
from .model import build_model


def verified_json(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Archived benchmark manifest changed: {path.name}")
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--encoder-weights", type=Path, required=True)
    parser.add_argument("--blosum", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rows = verified_json(
        args.selection,
        "1cfa56f24875d09d8ee26861e211d26568d8424178b4a214dc7ef751d30b4b41",
    )
    panel = verified_json(
        args.panel, "1e155c07c426e9a3a48003ce292b53b0c0f70790b94df608f159e1d86d7aad7a"
    )
    cases = {r["key"]: r for r in panel["cases"]}
    manifest = args.data / "manifest_strict.jsonl"
    check_rows = {}
    with manifest.open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["key"] in cases:
                check_rows[r["key"]] = r
    if set(check_rows) != set(cases) or set(check_rows).intersection(
        r["key"] for r in rows
    ):
        raise ValueError("Missing or overlapping benchmark structures")

    matrix = args.blosum.read_bytes()
    if (
        hashlib.sha256(matrix).hexdigest()
        != "85510d3846ee6d5f4778e425cf8daf6e0dbb889b306f2d13434e1254780efb40"
    ):
        raise ValueError("Expected Biopython BLOSUM62 (half-bit units)")
    lines = [
        line.split()
        for line in matrix.decode().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    table = {line[0]: line[1:] for line in lines[1:]}
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    scores = torch.tensor(
        [[int(table[a][lines[0].index(b)]) for b in alphabet] for a in alphabet],
        dtype=torch.float64,
    )
    exchange = torch.exp2(scores / 2)
    exchange /= exchange.sum(1, keepdim=True)
    identity = torch.eye(20, dtype=torch.float64)
    kernel = identity
    cumulative = []
    for beta in (0.10, 0.25, 0.45, 0.70):
        kernel = kernel @ ((1 - beta) * identity + beta * exchange)
        cumulative.append(kernel)
    cumulative.append(torch.full_like(kernel, 1 / 20))
    torch.testing.assert_close(
        torch.stack(cumulative).sum(-1), torch.ones(5, 20, dtype=torch.float64)
    )

    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    torch.manual_seed(18126)
    rng = np.random.default_rng(18126)
    torch.set_float32_matmul_precision("high")
    model = build_model(device)
    official = torch.load(args.encoder_weights, map_location="cpu", weights_only=True)[
        "model_state_dict"
    ]
    model.ligandmpnn.load_state_dict(official, strict=True)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01
    )
    assert not model.ligandmpnn.training and all(
        not p.requires_grad for p in model.ligandmpnn.parameters()
    )
    cumulative = torch.stack(cumulative).float().to(device)

    @lru_cache(maxsize=2)
    def shard(name):
        return torch.load(args.data / name, map_location="cpu", weights_only=False)

    def encode(row):
        record = shard(row["shard"])[row["index"]]
        features = structure_to_feature_dict(record, device)
        native = features["S"]
        features["S"] = torch.full_like(native, 20)
        with torch.no_grad():
            encoded = model.ligandmpnn.encode(features)
        return (
            tuple(v.cpu() for v in encoded),
            native.cpu(),
            features["mask"].cpu(),
            features["chain_mask"].cpu(),
        )

    steps = 8 if args.smoke else 25_000
    train_rows = rows[:16] if args.smoke else rows
    started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "cache_start",
                "structures": len(train_rows),
                "updates": steps,
                "device": torch.cuda.get_device_name(),
                "objective": "native-substitution-CE",
            }
        ),
        flush=True,
    )
    cache = []
    for i, row in enumerate(train_rows, 1):
        cache.append(encode(row))
        if i % 500 == 0:
            print(
                json.dumps(
                    {
                        "event": "cached",
                        "count": i,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )

    def inputs(entry, stage, generator=None):
        encoded, native, mask, chain = entry
        native = native.to(device)
        mask, chain = mask.to(device), chain.to(device)
        valid = mask.bool() & chain.bool() & native.ge(0) & native.lt(20)
        noisy = native.clone()
        noisy[valid] = torch.multinomial(
            cumulative[stage - 1, native[valid]], 1, generator=generator
        ).flatten()
        noisy[~mask.bool()] = 20
        t = torch.full((native.shape[0],), stage / 5, device=device)
        return (
            tuple(v.to(device) for v in encoded),
            native,
            noisy,
            t,
            mask,
            chain,
            valid,
        )

    with (args.out / "training.jsonl").open("x", buffering=1) as log:
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for index in rng.choice(len(cache), 4, replace=False):
                stage = int(torch.randint(1, 6, ()).item())
                enc, native, noisy, t, mask, chain, valid = inputs(cache[index], stage)
                loss = model.compute_substitution_loss(
                    {"mask": mask, "chain_mask": chain}, native, noisy, t, encoded=enc
                )
                (loss / 4).backward()
                batch_loss += float(loss.detach()) / 4
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == steps:
                log.write(
                    json.dumps(
                        {
                            "step": step,
                            "ce": batch_loss,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    + "\n"
                )
    for key, value in model.ligandmpnn.state_dict().items():
        torch.testing.assert_close(value.cpu(), official[key], rtol=0, atol=0)
    model.eval()
    evaluated = list(cases.values())
    if args.smoke:
        evaluated = evaluated[:2]
    records = []
    with torch.no_grad():
        for case in evaluated:
            entry = encode(check_rows[case["key"]])
            for seed in panel["seeds"]:
                generator = torch.Generator(device=device).manual_seed(seed)
                for stage in range(1, 6):
                    enc, native, noisy, t, mask, _, valid = inputs(
                        entry, stage, generator
                    )
                    logits = model.decode(enc, mask, noisy, t)[valid][:, :20].float()
                    pred, y, obs = logits.argmax(-1), native[valid], noisy[valid]
                    records.append(
                        {
                            "pdb_id": case["pdb_id"],
                            "partition": case["partition"],
                            "seed": seed,
                            "stage": stage,
                            "sites": y.numel(),
                            "ce": float(torch.nn.functional.cross_entropy(logits, y)),
                            "noisy_correct": int((obs == y).sum()),
                            "predicted_correct": int((pred == y).sum()),
                            "wrong_to_right": int(((obs != y) & (pred == y)).sum()),
                            "right_to_wrong": int(((obs == y) & (pred != y)).sum()),
                        }
                    )
    (args.out / "recovery_by_stage.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "steps": steps,
                "encoder_bitwise_unchanged": True,
                "evaluation_cases": len(evaluated),
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
