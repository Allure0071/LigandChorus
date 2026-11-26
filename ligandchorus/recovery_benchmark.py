"""Frozen-encoder, five-stage native identity reconstruction on the archived E18 panel."""

import argparse
import hashlib
import json
import os
import shutil
import time
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import torch

from .features import structure_to_feature_dict
from .model import build_model
from .sampling import regional_temperature


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore the last committed decoder and optimizer step",
    )
    parser.add_argument(
        "--generation",
        action="store_true",
        help="Run five reverse steps on the frozen E18 panel and save checkpoints",
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        help="Archive of the E18 quality metrics for the same panel",
    )
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
    transitions = []
    for beta in (0.10, 0.25, 0.45, 0.70):
        transitions.append((1 - beta) * identity + beta * exchange)
        kernel = kernel @ transitions[-1]
        cumulative.append(kernel)
    transitions.append(torch.full_like(kernel, 1 / 20))
    cumulative.append(kernel @ transitions[-1])
    torch.testing.assert_close(
        torch.stack(cumulative).sum(-1), torch.ones(5, 20, dtype=torch.float64)
    )

    if args.resume and not args.generation:
        parser.error("--resume requires --generation")
    args.out.mkdir(parents=True, exist_ok=args.resume)
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
    transitions = torch.stack(transitions).float().to(device)
    products = torch.cat((identity.float().to(device)[None], cumulative))

    @lru_cache(maxsize=24)
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
            regional_temperature(features).cpu(),
        )

    @cache
    def panel_entry(key):
        return encode(check_rows[key])

    @torch.inference_mode()
    def benchmark_trajectory(case_key, seed):
        """Generate on a panel structure without supplying its native labels to decode."""
        encoded, _, mask, chain, temperature = panel_entry(case_key)
        samples = panel["samples_per_seed"]
        device = model.W_out.weight.device
        encoded = tuple(v.to(device).expand(samples, *v.shape[1:]) for v in encoded)
        mask = mask.to(device).expand(samples, -1)
        design = mask.bool() & chain.to(device).bool()
        temperature = temperature.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        state = torch.randint(20, mask.shape, generator=generator, device=device)
        state = state.masked_fill(~design, model.mask_token)
        history = [state]
        for k in range(5, 0, -1):
            clock = torch.full((samples,), k / 5, device=device)
            logits = model.decode(encoded, mask, state, clock)[..., :20].float()
            clean = (logits / temperature).softmax(-1)
            # Mix q(x[k-1] | x[k], x[0]) using the decoder's distribution over x[0].
            j = state.clamp_max(19)
            denominator = products[k].T[j]
            previous = ((clean / denominator) @ products[k - 1]) * transitions[k - 1].T[
                j
            ]
            draw = torch.multinomial(previous.reshape(-1, 20), 1, generator=generator)
            state = draw.reshape_as(state).masked_fill(~design, model.mask_token)
            history.append(state)
        return torch.stack(history)

    steps = 8 if args.smoke else 25_000
    train_rows = rows[:16] if args.smoke else rows
    started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "train_start",
                "structures": len(train_rows),
                "updates": steps,
                "device": torch.cuda.get_device_name(),
                "objective": "native-substitution-CE",
            }
        ),
        flush=True,
    )

    @cache
    def train_entry(index):
        return encode(train_rows[index])

    def inputs(entry, stage, generator=None):
        encoded, native, mask, chain, _ = entry
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

    run_key = {
        "protocol": "e18-blosum5-v1",
        "structures": len(train_rows),
        "steps": steps,
        "encoder_sha256": hashlib.sha256(args.encoder_weights.read_bytes()).hexdigest(),
    }
    checkpoint = args.out / "decoder_optimizer.pt"
    head_names = {k for k in model.state_dict() if not k.startswith("ligandmpnn.")}

    def restore():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["run_key"] != run_key or state["head"].keys() != head_names:
            raise ValueError(
                "Checkpoint does not match the encoder and training protocol"
            )
        model.load_state_dict({**model.state_dict(), **state["head"]}, strict=True)
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device)
        return state["step"]

    start_step = restore() if args.resume else 0
    log_path = args.out / "training.jsonl"
    if args.resume:
        committed = []
        for line in log_path.read_text().splitlines():
            if not line.endswith("}"):
                break
            if json.loads(line)["step"] <= start_step:
                committed.append(line)
        temp = log_path.with_suffix(".tmp")
        temp.write_text("".join(line + "\n" for line in committed))
        os.replace(temp, log_path)
    with log_path.open("a" if args.resume else "x", buffering=1) as log:
        for step in range(start_step + 1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for index in rng.choice(len(train_rows), 4, replace=False):
                stage = int(torch.randint(1, 6, ()).item())
                enc, native, noisy, t, mask, chain, valid = inputs(
                    train_entry(index), stage
                )
                loss = model.compute_substitution_loss(
                    {"mask": mask, "chain_mask": chain}, native, noisy, t, encoded=enc
                )
                (loss / 4).backward()
                batch_loss += float(loss.detach()) / 4
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if args.generation and (
                step % (4 if args.smoke else 500) == 0 or step == steps
            ):
                temp = checkpoint.with_suffix(".tmp")
                torch.save(
                    {
                        "run_key": run_key,
                        "step": step,
                        "head": {
                            k: v.detach().cpu()
                            for k, v in model.state_dict().items()
                            if k in head_names
                        },
                        "optimizer": optimizer.state_dict(),
                        "numpy_rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(device),
                    },
                    temp,
                )
                os.replace(temp, checkpoint)
                if args.smoke and step == 4 and not args.resume:
                    shutil.copyfile(checkpoint, args.out / "smoke_step4.pt")
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
    if args.generation:
        # The same checkpoint used for the final report must reproduce a complete trajectory.
        first = evaluated[0]["key"]
        before = benchmark_trajectory(first, panel["seeds"][0])
        optimizer.state.clear()
        with torch.no_grad():
            model.W_out.weight.zero_()
        if restore() != steps or not optimizer.state:
            raise RuntimeError("Incomplete checkpoint restore")
        after = benchmark_trajectory(first, panel["seeds"][0])
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        counts = (after[2:] != after[1:-1]).sum(0)
        changed = (counts >= 2).nonzero()
        example = None
        if changed.numel():
            sample, position = changed[0].tolist()
            example = {
                "sample": sample,
                "position_zero_based": position,
                "states": after[:, sample, position].tolist(),
            }
        if args.smoke and example is None:
            raise RuntimeError("No same-site repeated revision observed in smoke test")
        audit = {
            "step": steps,
            "case_key": first,
            "reload_trajectory_identical": True,
            "optimizer_states": len(optimizer.state),
            "later_multiple_revisions": int((counts >= 2).sum()),
            "example": example,
        }
        (args.out / "reload_trace.json").write_text(json.dumps(audit, indent=2) + "\n")
        rows_out = []
        for case in evaluated:
            entry = panel_entry(case["key"])
            _, native, mask, chain, temperature = entry
            valid = mask.bool() & chain.bool() & native.ge(0) & native.lt(20)
            # regional_temperature uses the same C-alpha distance and <5 A boundary.
            pocket = valid & temperature.squeeze(-1).eq(0.29)
            native = native[0].to(device)
            for seed in panel["seeds"]:
                trajectory = benchmark_trajectory(case["key"], seed)
                candidates = trajectory[-1]
                i, j = torch.triu_indices(
                    len(candidates), len(candidates), offset=1, device=device
                )
                row = {
                    "key": case["key"],
                    "pdb_id": case["pdb_id"],
                    "cluster": case["cluster"],
                    "partition": case["partition"],
                    "seed": seed,
                    "samples": len(candidates),
                    "steps": 5,
                    "metrics": {},
                }
                for region, keep in (("global", valid), ("interface", pocket)):
                    keep = keep[0].to(device)
                    if not keep.any():
                        raise ValueError(f"Empty {region} mask on {case['key']}")
                    row["metrics"]["recovery_" + region] = float(
                        (candidates[:, keep] == native[keep]).float().mean()
                    )
                    row["metrics"]["diversity_" + region] = float(
                        (candidates[i][:, keep] != candidates[j][:, keep])
                        .float()
                        .mean()
                    )
                rows_out.append(row)
        (args.out / "generation_metrics.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows_out)
        )
        summaries = {}
        for split in ("calibration", "check"):
            group = [r["metrics"] for r in rows_out if r["partition"] == split]
            if group:
                summaries[split] = {
                    key: float(np.mean([r[key] for r in group])) for key in group[0]
                }
        (args.out / "table2_quality.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
        if args.reference_summary and not args.smoke:
            reference = json.loads(args.reference_summary.read_text())
            if (
                reference["check_pdbs"]
                != len({r["pdb_id"] for r in rows_out if r["partition"] == "check"})
                or reference["sampling_seeds"] != panel["seeds"]
                or reference["samples_per_seed"] != panel["samples_per_seed"]
            ):
                raise ValueError(
                    "Reference panel does not match the generation protocol"
                )
            quality = {
                **reference["metrics"],
                "blosum5_regional_0.29_0.24": summaries["check"],
            }
            differences = {
                name: {
                    k: 100 * (summaries["check"][k] - metrics[k])
                    for k in summaries["check"]
                }
                for name, metrics in reference["metrics"].items()
            }
            (args.out / "table2_comparison.json").write_text(
                json.dumps(
                    {
                        "check_pdbs": reference["check_pdbs"],
                        "metrics": quality,
                        "scheme2_minus_reference_pp": differences,
                    },
                    indent=2,
                )
                + "\n"
            )
    records = []
    with torch.no_grad():
        for case in evaluated:
            entry = panel_entry(case["key"])
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
