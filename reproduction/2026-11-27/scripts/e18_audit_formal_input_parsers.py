"""Check what each model can read from the shared inputs, without sampling."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from ligandchorus.benchmark_outputs import BACKBONE, context_and_ligand, heavy_atoms, read_model


def read_expected(row):
    path = Path(row["input_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["input_sha256"]:
        raise ValueError("shared_pdb_hash_mismatch")
    context, ligand = context_and_ligand(read_model(path))
    labels, ids, backbone = [], [], []
    for index, (chain, residues) in enumerate(context.items()):
        for res in residues:
            atoms = heavy_atoms(res)
            backbone.append([atoms[name] for name in BACKBONE])
            labels.append(index)
            ids.append(f"{chain}:{res.seqid}")
    points = heavy_atoms(ligand[1])
    atoms = {a.name: a for a in ligand[1] if a.name in points}
    return path, ids, np.asarray(backbone), np.asarray(labels), [
        (atoms[name].element.name, pos) for name, pos in points.items()
    ]


def difference(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"coordinate_shape_or_finiteness_mismatch: {actual.shape} versus {expected.shape}")
    return float(np.linalg.norm(actual - expected, axis=-1).max())


def ligand_difference(actual, types, expected, encode):
    if len(actual) != len(expected) or not np.isfinite(actual).all():
        raise ValueError(f"ligand_atom_count_or_finiteness_mismatch: {len(actual)} versus {len(expected)}")
    expected_types = np.asarray([encode(name) for name, _ in expected])
    if Counter(types.tolist()) != Counter(expected_types.tolist()):
        raise ValueError("ligand_element_counts_differ")
    reference = np.asarray([position for _, position in expected])
    distances = np.linalg.norm(actual[:, None] - reference[None], axis=2)
    distances[types[:, None] != expected_types[None]] = np.inf
    return float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))


def verify_e18(path, row, ids, backbone, labels, ligand):
    import gemmi
    from ligandchorus.features import load_record, structure_to_feature_dict

    npz = path.with_name("e18_input.npz")
    if hashlib.sha256(npz.read_bytes()).hexdigest() != row["e18_input_sha256"]:
        raise ValueError("e18_record_hash_mismatch")
    record = load_record(npz)
    features = structure_to_feature_dict(record, "cpu")
    if not np.all(record["S"] == "ACDEFGHIKLMNPQRSTVWY".index("G")):
        raise ValueError("e18_input_sequence_not_blinded")
    if not np.array_equal(features["chain_labels"].numpy()[0], labels) or not features["mask"].all():
        raise ValueError("e18_chain_labels_or_mask_mismatch")
    if not features["Y_m"].sum(dim=-1).eq(min(25, len(ligand))).all():
        raise ValueError("e18_context_atom_count_mismatch")
    return difference(features["X"][0].numpy(), backbone), ligand_difference(
        record["Y"], record["Y_t"], ligand, lambda name: gemmi.Element(name).atomic_number
    ), 0


def verify_ligandmpnn(path, row, ids, backbone, labels, ligand):
    import gemmi
    from data_utils import parse_PDB

    data, _, _, insertion, _ = parse_PDB(str(path))
    actual_ids = [f"{chain}:{number}{icode.strip()}" for chain, number, icode in zip(
        data["chain_letters"], data["R_idx"].tolist(), insertion, strict=True)]
    if actual_ids != ids or not np.array_equal(data["chain_labels"].numpy(), labels) or not data["mask"].all():
        raise ValueError("ligandmpnn_residue_ids_chain_labels_or_mask_mismatch")
    mask = data["Y_m"].bool()
    return difference(data["X"].numpy(), backbone), ligand_difference(
        data["Y"][mask].numpy(), data["Y_t"][mask].numpy(), ligand,
        lambda name: gemmi.Element(name).atomic_number
    ), 0


def verify_lasermpnn(path, row, ids, backbone, labels, ligand):
    import gemmi
    from run_inference import ProteinComplexData, get_protein_hierview
    from LASErMPNN.utils.constants import dataset_atom_order

    data = ProteinComplexData(get_protein_hierview(str(path)), str(path),
                              treat_noncanonical_as_ligand=row["lasermpnn_noncanonical_ligand"],
                              verbose=False)
    actual_ids = [f"{r.chain_id}:{r.resnum}{r.icode.strip()}" for r in data.residue_identifiers]
    if actual_ids != ids:
        raise ValueError("lasermpnn_residue_id_order_mismatch")
    lig = data.ligand_info
    original = [i for i, r in enumerate(lig.residue_identifiers)
                if r.resname == row["ligand_component"] and r.chain_id == "L"]
    if len(original) != 1:
        raise ValueError("lasermpnn_designated_ligand_not_recognized")
    extra = [r for i, r in enumerate(lig.residue_identifiers) if i != original[0]]
    if extra and (not row["lasermpnn_noncanonical_ligand"] or any(r.resname != "CAP" for r in extra)):
        raise ValueError("lasermpnn_unexpected_extra_ligand")
    raw_lig = lig.to_unprocessed_ligand_data()
    keep = raw_lig.lig_subbatch_indices == original[0]
    order = [dataset_atom_order["G"].index(a) for a in BACKBONE]
    return difference(data.heavy_atom_coords[:, order].numpy(), backbone), ligand_difference(
        raw_lig.lig_coords[keep].numpy(), raw_lig.lig_atomic_numbers[keep].numpy(), ligand,
        lambda name: gemmi.Element(name).atomic_number
    ), int((~keep).sum().item())


def verify_adflip(path, row, ids, backbone, labels, ligand):
    from data.all_atom_parse import get_element_index, pdb2data

    data = pdb2data(str(path))
    protein = data["is_protein"].bool()
    bb = protein & data["is_backbone"].bool()
    design = protein & data["is_center"].bool() & data["backbone_mask"].bool()
    if not np.array_equal(data["chain_id"][design].numpy(), labels):
        raise ValueError("adflip_designable_chain_order_mismatch")
    if difference(data["position"][design].numpy(), backbone[:, 1]) > 0.003:
        raise ValueError("adflip_designable_ca_order_mismatch")
    return difference(data["position"][bb].numpy(), backbone.reshape(-1, 3)), ligand_difference(
        data["position"][~protein].numpy(), data["element_index"][~protein].numpy(), ligand,
        lambda name: get_element_index(name.upper())
    ), 0


CHECKS = {"e18": verify_e18, "ligandmpnn": verify_ligandmpnn,
          "lasermpnn": verify_lasermpnn, "adflip": verify_adflip}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model", choices=CHECKS)
    cli.add_argument("inputs", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    results = Counter()
    with args.output.open("x", encoding="utf-8") as output:
        for line in args.inputs.read_text().splitlines():
            row = json.loads(line)
            record = {"key": row["key"], "test_group": row["test_group"]}
            if row["status"] != "materialized":
                record.update(status="input_unresolved")
            else:
                try:
                    path, ids, backbone, labels, ligand_atoms = read_expected(row)
                    bb, ligand, caps = CHECKS[args.model](path, row, ids, backbone, labels, ligand_atoms)
                    if max(bb, ligand) > 0.003:
                        raise ValueError(f"coordinate_delta_exceeds_tolerance: {bb:g}, {ligand:g}")
                    record.update(status="pass", max_backbone_delta_A=bb,
                                  max_ligand_delta_A=ligand, internal_ligand_cap_atoms=caps)
                except (RuntimeError, ValueError, OSError, IndexError, KeyError, AssertionError) as error:
                    record.update(status="failed", reason=str(error) or type(error).__name__)
            output.write(json.dumps(record) + "\n")
            output.flush()
            results[record["status"]] += 1
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
