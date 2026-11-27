"""Normalize benchmark outputs for one sequence-blinded complex."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .benchmark import AA

BACKBONE = ("N", "CA", "C", "O")


def read_model(path):
    import gemmi

    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError(f"{path}: expected one coordinate model")
    structure.setup_entities()
    return structure[0]


def heavy_atoms(residue):
    atoms = {}
    for atom in residue:
        if atom.element.atomic_number <= 1:
            continue
        if atom.name in atoms:
            raise ValueError(f"{residue.seqid}: ambiguous heavy atom {atom.name}")
        atoms[atom.name] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    return atoms


def context_and_ligand(model):
    chains = {}
    for chain in model:
        residues = list(chain.get_polymer())
        if residues:
            if chain.name in chains:
                raise ValueError(f"Duplicate chain {chain.name}")
            chains[chain.name] = residues
    if not chains:
        raise ValueError("Expected at least one protein chain")
    ligand = [
        (other.name, residue) for other in model for residue in other
        if residue.het_flag == "H" and len(heavy_atoms(residue)) > 1
    ]
    if len(ligand) != 1:
        raise ValueError("Expected one heavy-atom ligand instance")
    return chains, ligand[0]


def target_and_ligand(model):
    chains, ligand = context_and_ligand(model)
    if len(chains) != 1:
        raise ValueError("Expected one protein chain")
    name = next(iter(chains))
    return model[name], chains[name], ligand


def read_fasta(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            records.append([line[1:], ""])
        elif line.strip():
            if not records:
                raise ValueError(f"{path}: FASTA sequence before its header")
            records[-1][1] += line.strip()
    return records


def checked_sequences(sequences, size, samples, name):
    if len(sequences) != samples or any(len(seq) != size or set(seq) - set(AA) for seq in sequences):
        raise ValueError(f"{name}: expected {samples} complete canonical sequences of length {size}")
    return sequences


def mpnn_samples(path, name, basename, placeholder, samples, *, chains=None, target_index=0):
    records = read_fasta(path)
    if chains and name == "proteinmpnn":
        header = records[0][0] if records else ""
        match = re.search(r"designed_chains=(\[[^\]]*\])", header)
        if not match or ast.literal_eval(match[1]) != list(chains):
            raise ValueError(f"{path}: designed chains do not match the input order")
    if not isinstance(placeholder, str):
        sep = ":" if name == "ligandmpnn" else "/"
        sizes = [len(part) for part in placeholder]
        if records and records[0][1] != sep.join(placeholder):
            raise ValueError(f"{path}: FASTA input does not match the full blinded complex")
        if any([len(part) for part in seq.split(sep)] != sizes for _, seq in records):
            raise ValueError(f"{path}: FASTA chain segment lengths differ from the blinded input")
        records = [(header, seq.split(sep)[target_index]) for header, seq in records]
        placeholder = placeholder[target_index]
    if not records or records[0][0].split(",", 1)[0] != basename or records[0][1] != placeholder:
        raise ValueError(f"{path}: FASTA input record does not match the blinded backbone")
    id_field = "id" if name == "ligandmpnn" else "sample"
    indices = [re.search(rf"(?:^|, ){id_field}=(\d+)(?:,|$)", header)
               for header, _ in records[1:]]
    if not all(indices) or [int(match[1]) for match in indices] != list(range(1, samples + 1)):
        raise ValueError(f"{path}: missing, reordered, or duplicate sampled FASTA records")
    return checked_sequences([seq for _, seq in records[1:]], len(placeholder), samples, name)


def max_aligned_deltas(actual, reference, actual_ligand, reference_ligand):
    p, q = actual.mean(axis=0), reference.mean(axis=0)
    u, _, vh = np.linalg.svd((actual - p).T @ (reference - q))
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(u @ vh)
    rotation = u @ sign @ vh
    shift = q - p @ rotation
    return (
        float(np.linalg.norm(actual @ rotation + shift - reference, axis=1).max()),
        float(np.linalg.norm(actual_ligand @ rotation + shift - reference_ligand, axis=1).max()),
    )


def lasermpnn_samples(directory, basename, target, residues, ligand, samples, ideal_ca=None):
    import gemmi

    pattern = re.compile(rf"{re.escape(basename)}_design_(\d+)_segment_.*_chain_{re.escape(target)}(?: |$)")
    by_sample = {}
    for header, sequence in read_fasta(directory / "designs.fasta"):
        match = pattern.match(header)
        if match is None:
            continue
        index = int(match[1])
        if index in by_sample:
            raise ValueError(f"{directory}: duplicate design {index} for chain {target}")
        by_sample[index] = sequence
    if set(by_sample) != set(range(samples)):
        raise ValueError(f"{directory}: expected design indices 0 through {samples - 1} for chain {target}")
    identifiers = [str(res.seqid) for res in residues]
    reference = np.stack([heavy_atoms(res)["CA"] for res in residues])
    tolerance = 0.1
    if ideal_ca is not None:
        reference = ideal_ca
        tolerance = 0.003
    reference_ligand = sorted(map(tuple, heavy_atoms(ligand[1]).values()))
    designs = directory / basename if (directory / basename).is_dir() else directory
    for index, sequence in sorted(by_sample.items()):
        path = designs / f"design_{index}.pdb"
        chains = read_model(path)
        out = [res for res in chains[target] if res.het_flag == "A"]
        if [str(res.seqid) for res in out] != identifiers:
            raise ValueError(f"{path}: target residue identifiers differ from the shared input")
        positions = np.stack([heavy_atoms(res)["CA"] for res in out])
        if not np.isfinite(positions).all() or np.linalg.norm(positions - reference, axis=1).max() > tolerance:
            raise ValueError(f"{path}: target Cα coordinates differ from the expected backbone")
        ligands = [res for chain in chains for res in chain if res.het_flag == "H" and res.name == ligand[1].name]
        if len(ligands) != 1:
            raise ValueError(f"{path}: expected the original ligand in the designed structure")
        ligand_points = sorted(map(tuple, heavy_atoms(ligands[0]).values()))
        if len(ligand_points) != len(reference_ligand) or not np.allclose(
                ligand_points, reference_ligand, atol=0.003, rtol=0):
            raise ValueError(f"{path}: ligand coordinates differ from the shared input")
        output_seq = "".join(gemmi.find_tabulated_residue(res.name).one_letter_code for res in out)
        if sequence != output_seq:
            raise ValueError(f"{path}: sequence differs from the corresponding FASTA record")
    return checked_sequences([by_sample[i] for i in range(samples)], len(residues), samples, "lasermpnn")


def adflip_samples(path, digest, context, target, samples):
    output = json.loads(path.read_text(encoding="utf-8"))
    identifiers = [f"{chain}:{res.seqid}" for chain, residues in context.items() for res in residues]
    if output["backbone_sha256"] != digest or output["input_residue_ids"] != identifiers:
        raise ValueError(f"{path}: ADFLIP residue order or backbone hash differs from the shared input")
    sequence_list = checked_sequences(output["sequences"], len(identifiers), samples, "adflip")
    target_positions = [i for i, identifier in enumerate(identifiers) if identifier.split(":", 1)[0] == target]
    return ["".join(sequence[i] for i in target_positions) for sequence in sequence_list]


def lasermpnn_ca_reference(path, digest, context, target):
    record = json.loads(path.read_text(encoding="utf-8"))
    flat = [(chain, res) for chain, residues in context.items() for res in residues]
    ids = [f"{chain}:{res.seqid}" for chain, res in flat]
    if digest != record["backbone_sha256"] or ids != record["input_residue_ids"]:
        raise ValueError(f"{path}: LASErMPNN preprocessing residue IDs or input hash differ")
    expected_raw = np.stack([heavy_atoms(res)["CA"] for _, res in flat])
    raw, ideal = (np.asarray(record[key]) for key in ("raw_ca", "idealized_ca"))
    if raw.shape != expected_raw.shape or ideal.shape != expected_raw.shape or not np.isfinite(ideal).all():
        raise ValueError(f"{path}: invalid LASErMPNN coordinate arrays")
    if not np.allclose(raw, expected_raw, atol=0.003, rtol=0):
        raise ValueError(f"{path}: LASErMPNN original Cα locations differ")
    return ideal[[chain == target for chain, _ in flat]], float(np.linalg.norm(ideal - raw, axis=1).max())


def rfd3_sample(path, input_chain, input_residues, input_ligand, tolerance, *, context=None):
    import gemmi

    context = {input_chain: input_residues} if context is None else context
    returned, ligand = context_and_ligand(read_model(path))
    sidecar = path.with_name(path.name.removesuffix(".cif.gz").removesuffix(".cif") + ".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    input_keys = [f"{input_chain}{r.seqid}" for r in input_residues]
    before = [(f"{chain}{r.seqid}", r) for chain, rs in context.items() for r in rs]
    after = [(f"{chain}{r.seqid}", r) for chain, rs in returned.items() for r in rs]
    forward = metadata["diffused_index_map"]
    output_residues = dict(after)
    if (len(set(key for key, _ in before)) != len(before) or len(output_residues) != len(after)
            or set(forward) != {key for key, _ in before}
            or len(set(forward.values())) != len(forward) or set(forward.values()) != output_residues.keys()):
        raise ValueError(f"{path}: RFD3 residue mapping differs from the shared input")
    block = gemmi.cif.read_file(str(path)).sole_block()
    annotations = block.find("_atom_site.", [
        "auth_asym_id", "auth_atom_id", "auth_seq_id", "pdbx_PDB_ins_code", "is_motif_atom_with_fixed_seq",
    ])
    flags = []
    for chain, atom, residue, insertion, fixed in annotations:
        key = chain + residue + ("" if insertion in {"?", "."} else insertion)
        if atom == "CA" and key in output_residues:
            flags.append((key, fixed.lower()))
    if (len(flags) != len(after) or dict(flags).keys() != output_residues.keys()
            or any(flag not in {"0", "false"} for _, flag in flags)):
        raise ValueError(f"{path}: expected an explicit unfixed-sequence flag for every protein position")
    if ligand[1].name != input_ligand[1].name:
        raise ValueError(f"{path}: RFD3 ligand component differs")
    ref_lig = heavy_atoms(input_ligand[1])
    out_lig = heavy_atoms(ligand[1])
    if ref_lig.keys() != out_lig.keys():
        raise ValueError(f"{path}: RFD3 ligand atom names differ")

    ref_bb = np.stack([heavy_atoms(res)[atom] for _, res in before for atom in BACKBONE])
    out_bb = np.stack([heavy_atoms(output_residues[forward[key]])[atom] for key, _ in before for atom in BACKBONE])
    atom_names = list(ref_lig)
    bb_delta, ligand_delta = max_aligned_deltas(
        out_bb, ref_bb, np.stack([out_lig[a] for a in atom_names]),
        np.stack([ref_lig[a] for a in atom_names]),
    )
    if not np.isfinite([bb_delta, ligand_delta]).all() or max(bb_delta, ligand_delta) > tolerance:
        raise ValueError(f"{path}: RFD3 geometry exceeds the shared alignment tolerance")
    sequence = "".join(gemmi.find_tabulated_residue(output_residues[forward[key]].name).one_letter_code
                       for key in input_keys)
    return sequence, {"file": path.name, "unfixed_sequence_positions": len(flags),
                      "target_positions": len(input_keys),
                      "backbone_max_delta_A": bb_delta, "ligand_max_delta_A": ligand_delta}


def normalize(args):
    mapping = json.loads(args.alignment.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.backbone.read_bytes()).hexdigest()
    if digest != mapping["backbone_sha256"]:
        raise ValueError("Backbone hash differs from the residue alignment")
    context, ligand = context_and_ligand(read_model(args.backbone))
    order = mapping.get("input_chain_order", list(context) if len(context) == 1 else None)
    target = mapping.get("target_chain", next(iter(context)) if len(context) == 1 else None)
    if order != list(context) or order != sorted(context) or target not in context:
        raise ValueError("Expected an explicit target chain and the sorted input chain order")
    residues = context[target]
    input_ids = [f"{target}:{res.seqid}" for res in residues]
    native_ids = mapping["residue_ids"]
    if input_ids != mapping["input_residue_ids"] or len(native_ids) != len(input_ids):
        raise ValueError("Backbone residue mapping does not match")
    if any(res.name != "GLY" or set(heavy_atoms(res)) != set(BACKBONE)
           for chain_residues in context.values() for res in chain_residues):
        raise ValueError("Expected sequence-blinded GLY residues with only backbone atoms")
    size = len(residues)
    placeholders = ["G" * len(chain_residues) for chain_residues in context.values()]
    target_index = order.index(target)
    start = sum(len(part) for part in placeholders[:target_index])
    total = sum(len(part) for part in placeholders)

    e18 = np.load(args.e18, allow_pickle=False)
    if (e18.shape != (args.samples, total) or not np.issubdtype(e18.dtype, np.integer)
            or np.any(e18 < 0) or np.any(e18 >= len(AA))):
        raise ValueError("E18 tokens must encode complete canonical sequences in backbone order")
    sequences = {"e18": ["".join(AA[token] for token in sequence) for sequence in e18[:, start:start + size]]}
    mpnn_models = ("ligandmpnn", "proteinmpnn") if args.proteinmpnn else ("ligandmpnn",)
    for model in mpnn_models:
        sequences[model] = mpnn_samples(
            getattr(args, model), model, args.backbone.stem,
            placeholders if len(order) > 1 else placeholders[0], args.samples,
            chains=order if len(order) > 1 else None, target_index=target_index,
        )
    if args.lasermpnn:
        ideal, max_shift = (None, None)
        ideal_path = getattr(args, "lasermpnn_ca_reference", None)
        if ideal_path is not None:
            ideal, max_shift = lasermpnn_ca_reference(ideal_path, digest, context, target)
        sequences["lasermpnn"] = lasermpnn_samples(
            args.lasermpnn, args.backbone.stem, target, residues, ligand, args.samples, ideal_ca=ideal,
        )
        sequences["adflip"] = adflip_samples(args.adflip, digest, context, target, args.samples)
    rfd3_audit = {}
    if args.rfd3:
        if len(args.rfd3) != args.samples or len({path.resolve() for path in args.rfd3}) != args.samples:
            raise ValueError("Expected one distinct RFD3 structure file per sample")
        direct = [rfd3_sample(path, target, residues, ligand, args.geometry_tolerance, context=context)
                  for path in args.rfd3]
        sequences["rfd3_direct"] = checked_sequences(
            [sequence for sequence, _ in direct], size, args.samples, "rfd3_direct",
        )
        rfd3_audit = {
            "geometry_tolerance_A": args.geometry_tolerance,
            "rfd3_geometry": [geometry for _, geometry in direct],
        }
    args.output.mkdir()
    for model, sampled in sequences.items():
        row = {"key": mapping["key"], "residue_ids": native_ids, "sequences": sampled}
        (args.output / f"{model}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    audit = {
        "key": mapping["key"],
        "backbone_sha256": digest,
        "target_positions": size,
        "input_chain_lengths": {name: len(rs) for name, rs in context.items()},
        "target_chain": target,
        "samples_per_model": args.samples,
        "unique_sequences": {model: len(set(seq)) for model, seq in sequences.items()},
        **rfd3_audit,
    }
    if args.lasermpnn and max_shift is not None:
        audit["lasermpnn_max_backbone_idealization_shift_A"] = max_shift
    (args.output / "alignment_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("backbone", "alignment", "e18", "ligandmpnn", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--lasermpnn", type=Path, help="LASErMPNN output directory with FASTA and designed PDB files")
    parser.add_argument("--adflip", type=Path, help="ADFLIP JSON with the input SHA-256, residue IDs and sequences")
    parser.add_argument("--proteinmpnn", type=Path, help="historical comparison FASTA")
    parser.add_argument("--rfd3", type=Path, nargs="+")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--geometry-tolerance", type=float)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("Expected at least two samples")
    if args.proteinmpnn:
        if args.lasermpnn or args.adflip:
            parser.error("Choose either the historical or the current four-model comparison")
    elif not (args.lasermpnn and args.adflip) or args.rfd3:
        parser.error("The current comparison requires both --lasermpnn and --adflip")
    if args.rfd3 and (args.geometry_tolerance is None
                      or not np.isfinite(args.geometry_tolerance) or args.geometry_tolerance <= 0):
        parser.error("--rfd3 requires a positive, finite --geometry-tolerance")
    normalize(args)


if __name__ == "__main__":
    main()
