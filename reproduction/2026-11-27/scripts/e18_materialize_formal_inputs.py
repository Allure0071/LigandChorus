"""Materialize the shared inputs and separate scoring references for the pinned benchmark."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np

from e18_four_model_coordinate_audit import heavy_atoms, protein_positions, pocket_ids, coordinates
from ligandchorus.benchmark_outputs import BACKBONE, context_and_ligand, read_model


def digest(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream)
        stream.write("\n")


def residue_record(name, identifier, het_flag, atoms):
    residue = gemmi.Residue()
    residue.name = name
    residue.seqid = identifier
    residue.het_flag = het_flag
    for source in atoms:
        atom = source.clone()
        atom.altloc = "\x00"
        residue.add_atom(atom)
    return residue


def select_ligand(assembly, row, audit):
    index, author, number, subchain = row["ligand_instance"]
    chain = assembly[index]
    matches = [res for res in chain
               if chain.name == author and str(res.seqid) == number and res.subchain == subchain
               and res.name == row["ligand_component"] and res.entity_type == gemmi.EntityType.NonPolymer]
    if len(matches) != 1:
        raise ValueError("ligand_instance_mismatch")
    ligand = matches[0]
    atoms = heavy_atoms(ligand)
    if audit["ligand_conformation_check"]:
        compatible = [x["alternate_id"] for x in audit["ligand_conformation_check"]
                      if x["matches_pinned_coordinates"] and x["same_pocket"]]
        if len(compatible) != 1:
            raise ValueError("ligand_conformation_not_resolved")
        selected = compatible[0]
        atoms = heavy_atoms(residue_record(ligand.name, ligand.seqid, "H", [
            a for a in ligand if a.altloc in {" ", "\x00", selected}
        ]))
    else:
        selected = None
    if len(atoms) != audit["ligand_heavy_atoms"]:
        raise ValueError("ligand_atom_count_mismatch")
    return ligand.name, atoms, selected


def materialize(row, audit, cases):
    pid = row["pdb_id"]
    path = Path(row["coordinate_path"])
    if digest(path.read_bytes()) != row["coordinate_sha256"]:
        raise ValueError("source_coordinate_hash_mismatch")
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    spec = next(a for a in structure.assemblies if a.name == row["assembly_id"])
    assembly = gemmi.make_assembly(spec, structure[0], gemmi.HowToNameCopiedChain.Dup)
    model = gemmi.Model("1")
    original_target = {}
    expected = {}
    target_alias = audit["proposed_target_input_chain"]
    for alias in audit["proposed_protein_chain_aliases"]:
        source = assembly[alias["assembly_chain_index"]]
        if source.name != alias["author_chain"]:
            raise ValueError("protein_chain_alias_source_mismatch")
        chain = gemmi.Chain(alias["input_chain"])
        expected[chain.name] = {}
        for number, (res, atoms) in protein_positions(source).items():
            if not set(BACKBONE) <= atoms.keys():
                continue
            if chain.name == target_alias:
                if res.entity_id != row["target_entity"].split("_", 1)[1]:
                    raise ValueError("target_chain_contains_multiple_entities")
                original_target[number] = (res, atoms)
            backbone = [atoms[atom] for atom in BACKBONE]
            expected[chain.name][number] = coordinates(backbone)
            chain.add_residue(residue_record("GLY", res.seqid, "A", backbone))
        model.add_chain(chain)
    if len(original_target) != row["target_backbone_positions"]:
        raise ValueError("target_length_mismatch")

    ligand_name, ligand_atoms, alternate = select_ligand(assembly, row, audit)
    if set(pocket_ids(original_target, coordinates(ligand_atoms.values()))) != set(row["pocket_residue_ids"]):
        raise ValueError("pocket_mapping_mismatch")
    ligand_chain = gemmi.Chain(audit["proposed_ligand_chain"])
    ligand_chain.add_residue(residue_record(ligand_name, gemmi.SeqId(1, " "), "H", ligand_atoms.values()))
    model.add_chain(ligand_chain)
    output = gemmi.Structure()
    output.add_model(model)
    dest = cases / pid
    dest.mkdir()
    pdb = dest / "backbone.pdb"
    pdb.write_text(output.make_pdb_string(), encoding="ascii")
    parsed_context, parsed_ligand = context_and_ligand(read_model(pdb))
    if list(parsed_context) != sorted(expected) or parsed_ligand[1].name != ligand_name:
        raise ValueError("PDB_roundtrip_chain_or_ligand_mismatch")
    parsed_ligand_atoms = {atom.name: atom for atom in parsed_ligand[1]}
    if (parsed_ligand_atoms.keys() != ligand_atoms.keys() or any(
        atom.element.atomic_number != ligand_atoms[name].element.atomic_number
        or atom.pos.dist(ligand_atoms[name].pos) > 0.001 for name, atom in parsed_ligand_atoms.items()
    )):
        raise ValueError("PDB_roundtrip_ligand_coordinate_or_element_mismatch")
    X, labels = [], []
    for label, (chain, residues) in enumerate(parsed_context.items()):
        if list(expected[chain]) != [str(res.seqid) for res in residues]:
            raise ValueError("PDB_roundtrip_target_or_context_numbering_mismatch")
        for res in residues:
            atoms = heavy_atoms(res)
            if res.name != "GLY" or set(atoms) != set(BACKBONE):
                raise ValueError("PDB_roundtrip_unmasked_or_incomplete_backbone")
            point = coordinates(atoms[name] for name in BACKBONE)
            if np.linalg.norm(point - expected[chain][str(res.seqid)], axis=1).max() > 0.001:
                raise ValueError("PDB_roundtrip_backbone_coordinates_mismatch")
            X.append(point)
            labels.append(label)
    target_ids = [f"{target_alias}:{seqid}" for seqid in original_target]
    author_ids = [f"{row['target_author_chain']}:{seqid}" for seqid in original_target]
    mapping = {"key": pid, "backbone_sha256": digest(pdb.read_bytes()),
               "input_chain_order": list(parsed_context), "target_chain": target_alias,
               "input_residue_ids": target_ids, "residue_ids": author_ids}
    write_json(dest / "alignment.json", mapping)
    np.savez_compressed(
        dest / "e18_input.npz", X=np.asarray(X, dtype=np.float32),
        Y=coordinates(parsed_ligand_atoms.values()).astype(np.float32),
        Y_t=np.asarray([a.element.atomic_number for a in parsed_ligand_atoms.values()], dtype=np.int64),
        chain_labels=np.asarray(labels, dtype=np.int64),
        S=np.full(len(X), "ACDEFGHIKLMNPQRSTVWY".index("G"), dtype=np.int64),
        residue_mask=np.ones(len(X), dtype=np.float32),
    )
    sequence = "".join(gemmi.find_tabulated_residue(res.name).one_letter_code.upper()
                       for res, _ in original_target.values())
    pocket = set(row["pocket_residue_ids"])
    reference = {"key": pid, "residue_ids": author_ids, "native_sequence": sequence,
                 "pocket_positions": [i for i, rid in enumerate(original_target) if rid in pocket]}
    return {
        "key": pid, "test_group": row["test_group"], "family_id": row["family_id"],
        "source_coordinate_sha256": row["coordinate_sha256"],
        "input_path": str(pdb), "input_sha256": mapping["backbone_sha256"],
        "e18_input_sha256": digest((dest / "e18_input.npz").read_bytes()),
        "alignment_sha256": digest((dest / "alignment.json").read_bytes()),
        "chain_lengths": {chain: len(res) for chain, res in parsed_context.items()},
        "target_chain": target_alias, "target_positions": len(target_ids),
        "pocket_positions": len(reference["pocket_positions"]), "ligand_component": ligand_name,
        "ligand_heavy_atoms": len(ligand_atoms), "ligand_alternate_id": alternate,
        "lasermpnn_noncanonical_ligand": "lasermpnn_peptide_like_atom_names" in audit["flags"],
        "status": "materialized",
    }, reference


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("manifest", type=Path)
    cli.add_argument("audit", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    audit = json.loads(args.audit.read_text())
    manifest = args.manifest.read_bytes()
    if digest(manifest) != audit["manifest_sha256"]:
        raise ValueError("candidate_manifest_hash_mismatch")
    by_pdb = {row["pdb_id"]: row for row in audit["decisions"]}
    rows = [json.loads(line) for line in manifest.splitlines()]
    if len(by_pdb) != len(rows) or set(by_pdb) != {row["pdb_id"] for row in rows}:
        raise ValueError("coordinate_audit_case_set_mismatch")
    cases = args.out.resolve() / "cases"
    cases.mkdir(parents=True)
    summary = Counter()
    with (args.out / "inputs.jsonl").open("x", encoding="utf-8") as inputs, (
        args.out / "scoring_references.jsonl"
    ).open("x", encoding="utf-8") as references:
        for row in rows:
            decision = by_pdb[row["pdb_id"]]
            try:
                if decision["coordinate_status"] != "matched":
                    raise ValueError("prior_coordinate_check_unresolved")
                record, reference = materialize(row, decision, cases)
            except (ValueError, RuntimeError, OSError, StopIteration, IndexError) as error:
                record = {"key": row["pdb_id"], "test_group": row["test_group"],
                          "family_id": row["family_id"], "status": "unresolved", "reason": str(error)}
            else:
                references.write(json.dumps(reference) + "\n")
            inputs.write(json.dumps(record) + "\n")
            inputs.flush()
            summary[record["status"]] += 1
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
