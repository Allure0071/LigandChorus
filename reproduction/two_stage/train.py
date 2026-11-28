"""Paired decoder training on the frozen general and enzyme splits, in that order."""

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import shutil
from statistics import median
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduction.plan1_formal.run import (
    DEVICE,
    MODES,
    PEER,
    blind_encode,
    draw,
    dump,
    gpu,
    read,
)
from ligandchorus.model import build_model
from ligandchorus.quota_masking import masked_training_input
from ligandchorus.records import (
    RecordStore,
    manifest_entries,
    targeted_training_entries,
)
from ligandchorus.train import save_weights, training_features
from ligandchorus.validation import load_validation
from calibrate import digest


def train(args):
    assets = read(PEER / "assets.json")
    cfg = read(PEER / "configs/training.json")
    files = assets["files"]
    general = args.split / "stage1_manifest.jsonl"
    split_audit = read(args.split / "selection_audit.json")
    if (
        split_audit["selected"] != 50000
        or digest(general) != split_audit["stage1_sha256"]
    ):
        raise ValueError(
            "Stage 1 requires the verified, locked 50,000-assembly manifest"
        )
    manifest, targets, panel, weights = [
        Path(files[k]["path"])
        for k in (
            "training_manifest",
            "enzyme_training_targets",
            "validation_panel",
            "official_ligandmpnn",
        )
    ]
    for p, k in zip(
        (manifest, targets, panel, weights),
        (
            "training_manifest",
            "enzyme_training_targets",
            "validation_panel",
            "official_ligandmpnn",
        ),
    ):
        if digest(p) != files[k]["sha256"]:
            raise ValueError(f"Input hash changed: {k}")
    enzyme, labels = targeted_training_entries(manifest, targets, 5044)
    groups = [list(manifest_entries(general)), enzyme]
    assert len(groups[0]) == 50000 and len(enzyme) == 5044
    assert {r["pdb_id"] for r in groups[0]}.isdisjoint(r["pdb_id"] for r in enzyme)
    budgets = (2, 2) if args.smoke else (100000, 25000)
    interval = 1 if args.smoke else 1000
    batch_size = cfg["batch_size"]
    out = args.out
    out.mkdir(exist_ok=args.resume)
    base = Path(assets["directories"]["training_shard_root"])
    official = torch.load(weights, map_location="cpu", weights_only=True)[
        "model_state_dict"
    ]
    models, optimizers, generators = {}, {}, {}
    for mode in MODES:
        torch.manual_seed(cfg["training_seed"])
        model = build_model(DEVICE)
        model.ligandmpnn.load_state_dict(official, strict=True)
        models[mode] = model.train()
        optimizers[mode] = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["learning_rate"],
            weight_decay=0.01,
        )
        generators[mode] = torch.Generator(device=DEVICE).manual_seed(
            cfg["training_seed"]
        )
    rng = np.random.RandomState(cfg["training_seed"])
    best = {mode: {"score": -1, "step": 0} for mode in MODES}
    protocol = {
        "stage1_sha256": split_audit["stage1_sha256"],
        "enzyme_sha256": files["training_manifest"]["sha256"],
        "validation_sha256": files["validation_panel"]["sha256"],
        "encoder_sha256": files["official_ligandmpnn"]["sha256"],
        "budgets": list(budgets),
        "seed": cfg["training_seed"],
        "batch": batch_size,
    }
    saved = 0
    ckpt = out / "resume.pt"
    if args.resume:
        state = torch.load(
            ckpt, map_location="cpu", weights_only=False
        )  # own local atomic checkpoint
        if state["protocol"] != protocol:
            raise ValueError("Resume protocol mismatch")
        saved = state["global_step"]
        best = state["best"]
        rng.set_state(state["rng"])
        for mode in MODES:
            models[mode].load_state_dict(
                {**models[mode].state_dict(), **state["heads"][mode]}
            )
            optimizers[mode].load_state_dict(state["optimizers"][mode])
            generators[mode].set_state(state["generators"][mode])
        del state
    dump(out / "protocol.json", protocol)
    for stage, original in enumerate(groups, 1):
        offset = 0 if stage == 1 else budgets[0]
        if saved >= offset + budgets[stage - 1]:
            continue
        store = RecordStore(general if stage == 1 else manifest, base)
        entries = original
        if args.smoke:
            # Include the longest general assembly to exercise the real GPU memory path.
            longest = max(original, key=lambda r: r.get("residues", 0))
            entries = [longest] + [r for r in original if r["key"] != longest["key"]][
                : batch_size - 1
            ]
        print(
            json.dumps(
                {"event": "input_cache", "stage": stage, "structures": len(entries)}
            ),
            flush=True,
        )
        store_entries = sorted(
            enumerate(entries),
            key=lambda ir: (ir[1].get("shard", ""), ir[1].get("index", 0)),
        )
        pool = [None] * len(entries)
        for loaded, (index, row) in enumerate(store_entries, 1):
            f = training_features(
                store, row, labels[row["key"]] if stage == 2 else None
            )
            if stage == 1:
                f["loss_mask"] = f["mask"].bool() & f["S"].ge(0) & f["S"].lt(20)
            pool[index] = f
            if loaded % 5000 == 0:
                print(
                    json.dumps(
                        {"event": "inputs_loaded", "stage": stage, "count": loaded}
                    ),
                    flush=True,
                )
        validation = []
        if stage == 2:
            cases = load_validation(panel, manifest, store, enzyme)
            assert len(cases) == 128
            for case in cases[:2] if args.smoke else cases:
                validation.append(
                    (
                        case,
                        tuple(
                            t.cpu()
                            for t in blind_encode(models["quota"], gpu(case.features))
                        ),
                    )
                )
        del store
        cache = OrderedDict()
        cache_bytes = 0
        cache_limit = 16 * 1024**3
        start = time.monotonic()
        for step in range(max(saved - offset, 0) + 1, budgets[stage - 1] + 1):
            totals = dict.fromkeys(MODES, 0.0)
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            for idx in rng.choice(len(pool), batch_size, replace=False):
                cpu = pool[idx]
                mask, native, valid = (
                    cpu[k].to(DEVICE) for k in ("mask", "S", "loss_mask")
                )
                if idx in cache:
                    record, size = cache.pop(idx)
                    cache[idx] = record, size
                    enc = tuple(t.to(DEVICE) for t in record)
                else:
                    enc = blind_encode(models["quota"], gpu(cpu))
                    record = tuple(t.cpu() for t in enc)
                    size = sum(t.numel() * t.element_size() for t in record)
                    while cache and cache_bytes + size > cache_limit:
                        cache_bytes -= cache.popitem(last=False)[1][1]
                    if size <= cache_limit:
                        cache[idx] = record, size
                        cache_bytes += size
                for mode in MODES:
                    model = models[mode]
                    if mode == "quota":
                        k = int(
                            torch.randint(
                                1, 6, (), device=DEVICE, generator=generators[mode]
                            )
                        )
                        noisy, t, hidden = masked_training_input(
                            native, valid, k, generators[mode]
                        )
                        loss = model.compute_substitution_loss(
                            {"mask": mask, "chain_mask": hidden},
                            native,
                            noisy,
                            t,
                            encoded=enc,
                        )
                    else:
                        with patch.object(model.ligandmpnn, "encode", return_value=enc):
                            loss = model.compute_loss(
                                {"S": native, "mask": mask},
                                native,
                                loss_mask=valid,
                                generator=generators[mode],
                            )
                    (loss / batch_size).backward()
                    totals[mode] += loss.detach().item() / batch_size
            for mode in MODES:
                torch.nn.utils.clip_grad_norm_(
                    models[mode].parameters(), 1.0, error_if_nonfinite=True
                )
                optimizers[mode].step()
            if step == 1 or step % 100 == 0 or step == budgets[stage - 1]:
                print(
                    json.dumps(
                        {
                            "event": "update",
                            "stage": stage,
                            "step": step,
                            "global_step": offset + step,
                            "loss": totals,
                            "seconds": time.monotonic() - start,
                        }
                    ),
                    flush=True,
                )
            if step % interval != 0 and step != budgets[stage - 1]:
                continue
            if stage == 2:
                for mode in MODES:
                    model = models[mode].eval()
                    scores = []
                    for i, (case, stored) in enumerate(validation):
                        f = gpu(case.features)
                        pred = draw(
                            model,
                            f,
                            tuple(t.to(DEVICE) for t in stored),
                            mode,
                            i + cfg["validation_seed"],
                            0.29,
                            0.24,
                        )
                        scores.append(
                            (
                                pred[:, case.pocket_positions]
                                == f["S"][:, case.pocket_positions]
                            )
                            .float()
                            .mean()
                            .item()
                        )
                    val = median(scores)
                    if val > best[mode]["score"]:
                        best[mode] = {"score": val, "step": step}
                        save_weights(
                            model,
                            out / f"{mode}_best.pt",
                            replace=(out / f"{mode}_best.pt").exists(),
                        )
                    dump(
                        out / f"validation_{step}_{mode}.json",
                        {"median_pocket": val, "per_structure": scores},
                    )
                    model.train()
                dump(out / "selection.json", best)
            state = {
                "protocol": protocol,
                "global_step": offset + step,
                "best": best,
                "rng": rng.get_state(),
                "heads": {
                    m: {
                        k: t.cpu()
                        for k, t in models[m].state_dict().items()
                        if not k.startswith("ligandmpnn.")
                    }
                    for m in MODES
                },
                "optimizers": {m: o.state_dict() for m, o in optimizers.items()},
                "generators": {m: g.get_state() for m, g in generators.items()},
            }
            tmp = ckpt.with_suffix(".tmp")
            torch.save(state, tmp)
            tmp.replace(ckpt)
            del state
            if stage == 1 and step == budgets[0]:
                transfer = out / "stage1_transfer.pt"
                shutil.copyfile(ckpt, transfer.with_suffix(".tmp"))
                transfer.with_suffix(".tmp").replace(transfer)
        for mode in MODES:
            for k, t in models[mode].ligandmpnn.state_dict().items():
                torch.testing.assert_close(t.cpu(), official[k], rtol=0, atol=0)
            p = out / f"{mode}_stage{stage}_last.pt"
            save_weights(models[mode], p, replace=p.exists())
        dump(
            out / f"stage{stage}_complete.json",
            {
                "stage": stage,
                "global_step": offset + budgets[stage - 1],
                "encoder_bitwise_unchanged": True,
                "optimizer_steps": {
                    m: int(next(iter(o.state.values()))["step"].item())
                    for m, o in optimizers.items()
                },
            },
        )
        del pool, cache, validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())
