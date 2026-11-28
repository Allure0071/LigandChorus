"""Evaluate a fixed set of target chains and structure-derived pockets."""

import statistics
from dataclasses import dataclass

import torch

from .features import structure_to_feature_dict
from .records import manifest_entries
from .sampling import sample


@dataclass(frozen=True)
class ValidationCase:
    features: dict
    chain_mask: torch.Tensor
    pocket_positions: torch.Tensor


def load_validation(panel, manifest, store, training_records):
    selections = {}
    for selection in manifest_entries(panel):
        key = selection.get("key")
        if not isinstance(key, str) or not key or key in selections:
            raise ValueError(f"Validation requires distinct nonempty record keys: {key!r}")
        selections[key] = selection
    if not selections:
        raise ValueError("The validation panel is empty")

    entries = {}
    for entry in manifest_entries(manifest):
        key = entry.get("key")
        if key in selections:
            if key in entries or entry.get("split") != "valid":
                raise ValueError(f"Expected one explicit valid record for {key!r}")
            entries[key] = entry
    missing = selections.keys() - entries.keys()
    if missing:
        raise ValueError(f"Validation keys absent from the structural manifest: {sorted(missing)}")
    train_pdbs = {row["pdb_id"] for row in training_records if row.get("pdb_id")}
    shared_pdbs = train_pdbs & {row["pdb_id"] for row in entries.values() if row.get("pdb_id")}
    if shared_pdbs:
        raise ValueError(f"PDB IDs found in both training and validation: {sorted(shared_pdbs)}")

    cases = []
    for key, selection in selections.items():
        chain = selection.get("chain_label")
        positions = selection.get("pocket_positions")
        if type(chain) is not int or not isinstance(positions, list) or not positions:
            raise ValueError(f"{key}: specify an integer chain_label and nonempty pocket_positions")
        if any(type(index) is not int or index < 0 for index in positions):
            raise ValueError(f"{key}: pocket positions must be nonnegative integer offsets")
        if len(set(positions)) != len(positions):
            raise ValueError(f"{key}: duplicate pocket positions")

        features = structure_to_feature_dict(store.read(entries[key]), "cpu")
        native = features["S"]
        score_mask = features["mask"].bool() & (features["chain_labels"] == chain)
        score_mask = score_mask & (native >= 0) & (native < 20)
        if max(positions) >= native.shape[1] or not score_mask[0, positions].all():
            raise ValueError(f"{key}: every pocket position must be scorable on the target chain")
        cases.append(ValidationCase(features, score_mask, torch.tensor(positions)))
    return cases


@torch.inference_mode()
def evaluate(model, cases, device, *, seed, samples=10, steps=20):
    """Use fixed corruption and sampling, then take structure-level medians."""
    was_training = model.training
    model.eval()
    losses, chain_recoveries, pocket_recoveries, diversities = [], [], [], []
    try:
        left, right = torch.triu_indices(samples, samples, offset=1, device=device)
        for offset, case in enumerate(cases):
            features = {key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in case.features.items()}
            native = features["S"]
            chain = case.chain_mask.to(device)
            pocket = case.pocket_positions.to(device)
            case_seed = seed + offset
            generator = torch.Generator(device=device).manual_seed(case_seed)
            losses.append(model.compute_loss(
                features, native, loss_mask=chain, generator=generator,
            ).item())
            tokens = sample(model, features, samples=samples, steps=steps, seed=case_seed)
            matches = tokens == native
            chain_recoveries.append(matches[:, chain[0]].float().mean().item())
            pocket_recoveries.append(matches[:, pocket].float().mean().item())
            pocket_tokens = tokens[:, pocket]
            differences = pocket_tokens[left] != pocket_tokens[right]
            diversities.append(differences.float().mean().item())
    finally:
        if was_training:
            model.train()
            model.ligandmpnn.eval()
    return {
        "structures": len(cases),
        "samples_per_structure": samples,
        "masked_ce": statistics.mean(losses),
        "chain_recovery": statistics.median(chain_recoveries),
        "pocket_recovery": statistics.median(pocket_recoveries),
        "pocket_diversity": statistics.median(diversities),
    }
