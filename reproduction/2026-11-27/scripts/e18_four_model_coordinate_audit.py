"""Read-only coordinate checks for the pinned four-model test candidates."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import string

import gemmi
import numpy as np
from prody import flagDefinition


BACKBONE = {"N", "CA", "C", "O"}
AA = set("ACDEFGHIKLMNPQRSTVWY")
PDB_PROTEIN_CHAIN_IDS = (string.ascii_uppercase + string.ascii_lowercase).replace("L", "")


def heavy_atoms(residue):
    atoms = {}
    for atom in sorted(residue, key=lambda a: (a.name, -a.occ, a.altloc)):
        if atom.element.atomic_number > 1 and atom.occ > 0:
            atoms.setdefault(atom.name, atom)
    return atoms


def coordinates(atoms):
    return np.asarray([[a.pos.x, a.pos.y, a.pos.z] for a in atoms]).reshape(-1, 3)


def protein_positions(chain):
    positions = {}
    for residue in chain.get_polymer():
        aa = gemmi.find_tabulated_residue(residue.name)
        if aa.one_letter_code.upper() not in AA:
            continue
        atoms = heavy_atoms(residue)
        identifier = str(residue.seqid)
        if identifier not in positions or len(atoms) > len(positions[identifier][1]):
            positions[identifier] = (residue, atoms)
    return positions


def pocket_ids(target_positions, ligand_atoms):
    pocket = []
    for identifier, (_, atoms) in target_positions.items():
        side = coordinates(a for name, a in atoms.items() if name not in BACKBONE | {"OXT"})
        if len(side) and np.any(np.sum((side[:, None] - ligand_atoms[None]) ** 2, axis=-1) < 25):
            pocket.append(identifier)
    return pocket


def ligand_conformation_check(ligand, prior_atoms, target, original_pocket):
    unlabeled = {" ", "\x00"}
    codes = sorted({atom.altloc for atom in ligand} - unlabeled)
    results = []
    for code in codes:
        atoms = {
            a.name: a for a in ligand
            if a.element.atomic_number > 1 and a.occ > 0 and a.altloc in (unlabeled | {code})
        }
        same_atoms = atoms.keys() == prior_atoms.keys()
        same_pocket = set(pocket_ids(target, coordinates(atoms.values()))) == set(original_pocket)
        results.append({"alternate_id": code, "same_heavy_atom_names": same_atoms,
                        "same_pocket": same_pocket,
                        "matches_pinned_coordinates": same_atoms and all(
                            atoms[name].pos.dist(a.pos) <= 0.001 for name, a in prior_atoms.items())})
    return results


def audit(row, removed_ligands, amino_acids, nonstandard_amino_acids):
    path = Path(row["coordinate_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["coordinate_sha256"]:
        raise ValueError("coordinate_sha256_mismatch")
    structure = gemmi.read_structure(str(path))
    if structure.name.lower() != row["pdb_id"].lower() or len(structure) == 0:
        raise ValueError("coordinate_structure_mismatch")
    structure.setup_entities()
    assemblies = [assembly for assembly in structure.assemblies if assembly.name == row["assembly_id"]]
    if len(assemblies) != 1:
        raise ValueError("assembly_not_unique")
    assembly = gemmi.make_assembly(assemblies[0], structure[0], gemmi.HowToNameCopiedChain.Dup)
    target = assembly[row["target_copy_index"]]
    if target.name != row["target_author_chain"]:
        raise ValueError("target_copy_mismatch")
    target_entity = row["target_entity"].split("_", 1)[1]
    all_positions = [protein_positions(chain) for chain in assembly]
    target_positions = {
        identifier: (residue, atoms)
        for identifier, (residue, atoms) in all_positions[row["target_copy_index"]].items()
        if residue.entity_id == target_entity and BACKBONE <= atoms.keys()
    }
    if len(target_positions) != row["target_backbone_positions"]:
        raise ValueError("target_backbone_count_mismatch")
    ligand_index, ligand_author, ligand_number, ligand_subchain = row["ligand_instance"]
    ligand_chain = assembly[ligand_index]
    ligands = [
        res for res in ligand_chain
        if ligand_chain.name == ligand_author and str(res.seqid) == ligand_number
        and res.subchain == ligand_subchain and res.name == row["ligand_component"]
        and res.entity_type == gemmi.EntityType.NonPolymer
    ]
    if len(ligands) != 1:
        raise ValueError("ligand_instance_mismatch")
    ligand = ligands[0]
    ligand_atom_map = heavy_atoms(ligand)
    ligand_atoms = coordinates(ligand_atom_map.values())
    if len(ligand_atoms) < 2 or not np.isfinite(ligand_atoms).all():
        raise ValueError("invalid_ligand_coordinates")
    pocket = pocket_ids(target_positions, ligand_atoms)
    if len(set(row["pocket_residue_ids"])) != len(pocket) or set(pocket) != set(row["pocket_residue_ids"]):
        raise ValueError("pocket_residue_set_mismatch")

    contexts = [
        [res for res, atoms in positions.values() if BACKBONE <= atoms.keys()]
        for positions in all_positions
    ]
    context = [res for chain in contexts for res in chain]
    protein_copies = [(index, chain.name) for index, (chain, positions) in enumerate(zip(assembly, contexts))
                     if positions]
    author_chains = [name for _, name in protein_copies]
    if len(protein_copies) > len(PDB_PROTEIN_CHAIN_IDS):
        raise ValueError("protein_chain_count_exceeds_chain_id_capacity")
    alias_map = {
        index: alias for (index, _), alias in zip(protein_copies, PDB_PROTEIN_CHAIN_IDS)
    }
    conformation_check = ligand_conformation_check(ligand, ligand_atom_map, target_positions, pocket)
    flags = {
        "multiple_protein_chains": len(author_chains) > 1,
        "repeated_author_chain": len(set(author_chains)) < len(author_chains),
        "insertion_code_target": any(r.seqid.icode.strip() for r, _ in target_positions.values()),
        "insertion_code_in_pocket": any(target_positions[i][0].seqid.icode.strip() for i in pocket),
        "insertion_code_any_chain": any(r.seqid.icode.strip() for r in context),
        "author_chain_id_exceeds_pdb_width": any(len(name) != 1 for name in author_chains),
        "ligand_component_exceeds_pdb_width": len(ligand.name) > 3,
        "alternate_ligand_conformations": bool(conformation_check),
        "adflip_crystallization_aid": ligand.name in removed_ligands,
        "lasermpnn_peptide_like_atom_names": BACKBONE <= ligand_atom_map.keys(),
        "adflip_amino_acid_name": ligand.name in amino_acids,
        "prody_nonstandard_amino_acid_name": ligand.name in nonstandard_amino_acids,
    }
    return {
        "coordinate_status": "matched",
        "protein_chains": len(author_chains),
        "target_positions": len(target_positions),
        "pocket_positions": len(pocket),
        "ligand_heavy_atoms": len(ligand_atoms),
        "flags": [name for name, value in flags.items() if value],
        "proposed_protein_chain_aliases": [
            {"assembly_chain_index": index, "author_chain": name, "input_chain": alias_map[index]}
            for index, name in protein_copies
        ],
        "proposed_target_input_chain": alias_map[row["target_copy_index"]],
        "proposed_ligand_chain": "L",
        "ligand_conformation_check": conformation_check,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("expected_sha256")
    parser.add_argument("adflip", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != args.expected_sha256:
        raise ValueError("candidate_manifest_sha256_mismatch")
    stage = args.adflip / "data" / "misc" / "af3_crystallization_aids.txt"
    removed = set(stage.read_text().splitlines())
    nonstandard = set(flagDefinition("nonstdaa"))
    amino_acids = set(flagDefinition("stdaa"))
    decisions = []
    groups = defaultdict(Counter)
    for line in manifest_bytes.splitlines():
        row = json.loads(line)
        group = groups[row["test_group"]]
        group["candidates"] += 1
        try:
            result = audit(row, removed, amino_acids, nonstandard)
        except (ValueError, OSError, IndexError, RuntimeError) as error:
            result = {"coordinate_status": "unresolved", "reason": str(error)}
        group[result["coordinate_status"]] += 1
        group.update(result.get("flags", []))
        decisions.append({**{k: row[k] for k in ("pdb_id", "test_group", "family_id", "ligand_component")},
                          **result})
    report = {
        "manifest_sha256": args.expected_sha256,
        "candidate_count": len(decisions),
        "group_counts": groups,
        "case_counts_by_ligand": dict(Counter(row["ligand_component"] for row in decisions)),
        "formal_prediction_records_generated": 0,
        "decisions": decisions,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["group_counts"], indent=2))


if __name__ == "__main__":
    main()
