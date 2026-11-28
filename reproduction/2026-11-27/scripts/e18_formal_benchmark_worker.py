"""Seed the upstream baseline entry points and audit ADFLIP residue order."""

import hashlib
import json
from pathlib import Path
import random
import runpy
import sys
from time import perf_counter

import numpy as np
import torch


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def point_set_difference(actual, target):
    if actual.shape != target.shape or not np.isfinite(actual).all():
        return float("inf")
    distances = np.linalg.norm(actual[:, None] - target[None], axis=2)
    return float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))


def load_adflip(weights, device):
    from design import Config as CheckpointConfig, load_flow_model

    # The release checkpoint pickles this upstream class as __main__.Config.
    setattr(sys.modules["__main__"], "Config", CheckpointConfig)
    ema, _ = load_flow_model(str(weights), torch.device(device))
    return ema.ema_model.eval()


def run_adflip(seed, arguments, model=None):
    import gemmi
    from data import all_atom_parse
    from model import discrete_flow_aa

    pdb, weights, output, work = map(Path, arguments[:4])
    device = torch.device(arguments[4])
    samples, steps = map(int, arguments[5:7])
    threshold = float(arguments[7])
    if model is None:
        model = load_adflip(weights, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    structure = gemmi.read_structure(str(pdb))[0]
    protein = [(chain.name, res) for chain in structure for res in chain if res.het_flag == "A"]
    if not protein or any(res.name != "GLY" for _, res in protein):
        raise ValueError("Expected a blinded glycine backbone")
    residue_ids = [f"{chain}:{res.seqid}" for chain, res in protein]
    ca = np.asarray([list(res["CA"][0].pos) for _, res in protein])
    ligand = np.asarray([list(atom.pos) for chain in structure for res in chain if res.het_flag == "H"
                         for atom in res if atom.element.atomic_number > 1])
    if not len(ligand):
        raise ValueError("Expected a designated ligand")
    parser = discrete_flow_aa.pdb2data
    checks = 0

    def audited(path, device="cpu", *args, **kwargs):
        nonlocal checks
        data = parser(path, device, *args, **kwargs)
        protein_mask = data["is_protein"].bool()
        design_mask = protein_mask & data["is_center"].bool() & data["backbone_mask"].bool()
        observed_ca = data["position"][design_mask].cpu().numpy()
        observed_ligand = data["position"][~protein_mask].cpu().numpy()
        if observed_ca.shape != ca.shape or not np.allclose(observed_ca, ca, atol=0.003, rtol=0):
            raise ValueError(f"{Path(path).name}: ADFLIP changed designable Cα order or coordinates")
        if point_set_difference(observed_ligand, ligand) > 0.003:
            raise ValueError(f"{Path(path).name}: ADFLIP changed the shared ligand coordinates")
        checks += 1
        return data

    discrete_flow_aa.pdb2data = audited
    model.sample_save_path = str(work)
    sequences = []
    try:
        with torch.inference_mode():
            for sample in range(samples):
                seed_rng(seed * 100 + sample)
                tokens, _ = model.adaptive_sample(str(pdb), num_step=steps, threshold=threshold, argmax_final=True)
                seq = "".join(all_atom_parse.restype_3to1[all_atom_parse.index_to_token[i]]
                              for i in tokens.flatten().tolist())
                if len(seq) != len(ca) or set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
                    raise ValueError("ADFLIP returned incomplete or noncanonical tokens")
                sequences.append(seq)
    finally:
        discrete_flow_aa.pdb2data = parser
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    output.write_text(json.dumps({"backbone_sha256": hashlib.sha256(pdb.read_bytes()).hexdigest(),
                                  "input_residue_ids": residue_ids, "sequences": sequences,
                                  "coordinate_checks": checks}), encoding="utf-8")
    print(f"Verified {checks} coordinate parses for {samples} samples", flush=True)
    return elapsed


if __name__ == "__main__":
    model, raw_seed, *args = sys.argv[1:]
    seed = int(raw_seed)
    seed_rng(seed)
    if model == "lasermpnn":
        program, *flags = args
        sys.argv = [program, *flags]
        runpy.run_path(program, run_name="__main__")
    elif model == "adflip":
        run_adflip(seed, args)
    else:
        raise ValueError(f"Unknown worker model: {model}")
