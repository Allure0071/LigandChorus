"""Build LigandMPNN features from a structural tensor record."""

import numpy as np
import torch


def structure_to_feature_dict(structure, device, num_ligand_atoms=25):
    """将 shard 结构转换为 feature_dict"""
    X = torch.from_numpy(structure['X']).float().to(device)
    S = torch.from_numpy(structure['S']).long().to(device)
    Y_global = torch.from_numpy(structure['Y']).float().to(device)
    Y_t_global = torch.from_numpy(structure['Y_t']).long().to(device)
    residue_mask = torch.from_numpy(structure['residue_mask']).float().to(device)
    chain_labels = torch.from_numpy(structure['chain_labels']).long().to(device)

    L = X.shape[0]
    B = 1

    Ca = X[:, 1, :]
    distances = torch.cdist(Ca, Y_global)
    num_atoms = min(num_ligand_atoms, Y_global.shape[0])
    _, nearest_indices = torch.topk(distances, num_atoms, dim=1, largest=False)

    Y = torch.zeros(L, num_ligand_atoms, 3, device=device)
    Y_t = torch.zeros(L, num_ligand_atoms, dtype=torch.long, device=device)
    Y_m = torch.zeros(L, num_ligand_atoms, device=device)

    for i in range(L):
        Y[i, :num_atoms] = Y_global[nearest_indices[i]]
        Y_t[i, :num_atoms] = Y_t_global[nearest_indices[i]]
        Y_m[i, :num_atoms] = 1.0

    X = X.unsqueeze(0)
    S = S.unsqueeze(0)
    Y = Y.unsqueeze(0)
    Y_t = Y_t.unsqueeze(0)
    Y_m = Y_m.unsqueeze(0)
    residue_mask = residue_mask.unsqueeze(0)
    chain_labels = chain_labels.unsqueeze(0)

    mask = residue_mask
    chain_mask = torch.ones_like(mask)
    R_idx = torch.arange(L, device=device).unsqueeze(0)
    bias = torch.zeros(B, L, 21, device=device)

    xyz_37 = torch.zeros(B, L, 37, 3, device=device)
    xyz_37_m = torch.zeros(B, L, 37, device=device)
    xyz_37[:, :, :4, :] = X
    xyz_37_m[:, :, :4] = 1.0

    Ca = X[:, :, 1, :]
    N = X[:, :, 0, :]
    C = X[:, :, 2, :]
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
    xyz_37[:, :, 4, :] = Cb
    xyz_37_m[:, :, 4] = 1.0

    feature_dict = {
        'X': X, 'S': S, 'Y': Y, 'Y_t': Y_t, 'Y_m': Y_m,
        'mask': mask, 'chain_mask': chain_mask, 'R_idx': R_idx,
        'chain_labels': chain_labels, 'bias': bias,
        'temperature': 0.1, 'batch_size': B,
        'symmetry_residues': [[]], 'symmetry_weights': [[]],
        'xyz_37': xyz_37, 'xyz_37_m': xyz_37_m,
    }

    return feature_dict


def load_record(path):
    """Read a structural NPZ without loading pickled objects."""
    keys = ("X", "S", "Y", "Y_t", "residue_mask", "chain_labels")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(keys) - set(archive.files)
        if missing:
            raise ValueError(f"Missing structural arrays in {path}: {', '.join(sorted(missing))}")
        record = {key: archive[key] for key in keys}
    backbone, atoms = record["X"], record["Y"]
    if backbone.ndim != 3 or backbone.shape[1:] != (4, 3) or not backbone.shape[0]:
        raise ValueError("X must describe at least one residue with coordinates shaped [L, 4, 3]")
    if atoms.ndim != 2 or atoms.shape[1] != 3:
        raise ValueError("Y must have shape [A, 3]")
    for key in ("S", "residue_mask", "chain_labels"):
        if record[key].shape != (len(backbone),):
            raise ValueError(f"{key} must have shape [L]")
    if record["Y_t"].shape != (len(atoms),):
        raise ValueError("Y_t must have shape [A]")
    return record
