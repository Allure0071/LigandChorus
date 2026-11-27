"""Keep one benchmark model resident and record synchronized per-case timings."""

import argparse
import ast
import functools
import json
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import numpy as np
import torch

from e18_formal_benchmark_runner import atomic_json, digest
from e18_formal_benchmark_worker import load_adflip, run_adflip, seed_rng


def now(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter()


def e18_engine(device, config, samples):
    from ligandchorus.features import load_record, structure_to_feature_dict
    from ligandchorus.model import build_model
    from ligandchorus.sampling import sample

    model = build_model(device)
    model.load_state_dict(torch.load(config["checkpoint"]["server_path"], map_location="cpu", weights_only=True))
    model.eval()

    def run(row, pdb, npz, target, seed):
        start = now(device)
        features = structure_to_feature_dict(load_record(npz), device)
        tokens = sample(model, features, samples=samples, steps=config["max_steps"], seed=seed,
                        inside=config["inside_temperature"], outside=config["outside_temperature"])
        tokens = tokens.cpu().numpy()
        elapsed = now(device) - start
        np.save(target / "e18.npy", tokens, allow_pickle=False)
        return elapsed

    return run


def ligandmpnn_engine(device, config, samples, source):
    # Reuse the pinned upstream parser, model construction, main loop, and output formatter.
    # Remove only its per-call model initialization block.
    program = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    cli = program.body.pop()
    assert isinstance(cli, ast.If) and isinstance(cli.body[-1], ast.Expr)
    assert isinstance(cli.body[-2], ast.Assign) and cli.body[-2].targets[0].id == "args"
    main = next(node for node in program.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    start = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                 and any(isinstance(n, ast.Name) and n.id == "checkpoint" for n in node.targets))
    end = next(i for i, node in enumerate(main.body) if i > start and isinstance(node, ast.If)
               and ast.unparse(node.test) == "args.pack_side_chains")
    initialize = ast.Module(body=main.body[start:end], type_ignores=[])
    main.body[start:end] = ast.parse("model, atom_context_num = resident_model, resident_atom_context_num").body
    scope = {"__name__": "ligandmpnn_resident", "__file__": str(source)}
    exec(compile(ast.fix_missing_locations(program), str(source), "exec"), scope)
    parser_body = ast.Module(body=cli.body[:-2], type_ignores=[])
    exec(compile(ast.fix_missing_locations(parser_body), str(source), "exec"), scope)
    init_args = scope["argparser"].parse_args([
        "--model_type", "ligand_mpnn", "--checkpoint_ligand_mpnn", config["checkpoint"]["server_path"],
        "--temperature", str(config["temperature"]), "--batch_size", str(samples), "--number_of_batches", "1",
        "--omit_AA", "".join(config["omit"]), "--verbose", "0",
    ])
    assert not init_args.pack_side_chains
    scope.update(args=init_args, checkpoint_path=init_args.checkpoint_ligand_mpnn, device=torch.device(device))
    exec(compile(ast.fix_missing_locations(initialize), str(source), "exec"), scope)
    scope.update(resident_model=scope.pop("model"), resident_atom_context_num=scope.pop("atom_context_num"))
    scope.pop("checkpoint")
    original_parse = scope["parse_PDB"]

    def run(row, pdb, npz, target, seed):
        events = {}

        def parse(*args, **kwargs):
            events["begin"] = now(device)
            return original_parse(*args, **kwargs)

        def record_output(name, *args, **kwargs):
            if str(name).endswith(".fa"):
                events["end"] = now(device)
            return open(name, *args, **kwargs)

        scope.update(parse_PDB=parse, open=record_output)
        args = argparse.Namespace(**vars(init_args))
        args.seed = seed
        args.pdb_path = str(pdb)
        args.out_folder = str(target)
        scope["main"](args)
        return events["end"] - events["begin"]

    return run


def lasermpnn_engine(device, config, samples):
    import run_batch_inference as official
    from run_inference import ProteinComplexData, get_protein_hierview

    checkpoint = config["checkpoint"]["server_path"]
    loaded = official.load_model_from_parameter_dict(checkpoint, device)
    original_infer = official._run_inference
    official.load_model_from_parameter_dict = lambda *args, **kwargs: loaded

    def run(row, pdb, npz, target, seed):
        seed_rng(seed)
        timings = []

        @functools.wraps(original_infer)
        def timed_infer(*args, **kwargs):
            started = now(device)
            result = original_infer(*args, **kwargs)
            timings.append(now(device) - started)
            return result

        official._run_inference = timed_infer
        flag = row["lasermpnn_noncanonical_ligand"]
        official.run_inference(pdb, target, checkpoint, config["sequence_temperature"], None, device,
                               samples, config["batch_size"], 1, False, True, verbose=False,
                               first_shell_sequence_temp=config["first_shell_temperature"],
                               disabled_residues=",".join(config["disabled_residues"]),
                               noncanonical_aa_ligand=flag, output_fasta=True, ala_budget=4, gly_budget=0)
        # The reference here is the upstream deterministic input transformation.
        data = ProteinComplexData(get_protein_hierview(str(pdb)), str(pdb),
                                  treat_noncanonical_as_ligand=flag, verbose=False)
        original = data.heavy_atom_coords[:, 1].numpy()
        ideal = data.compute_backbone_coords()[0][:, 1].numpy()
        ids = [f"{r.chain_id}:{r.resnum}{r.icode.strip()}" for r in data.residue_identifiers]
        atomic_json(target / "idealized_ca.json", {
            "backbone_sha256": row["input_sha256"],
            "input_residue_ids": ids,
            "raw_ca": original.tolist(),
            "idealized_ca": ideal.tolist(),
        })
        return sum(timings)

    return run


def adflip_engine(device, config, samples):
    weights = config["checkpoint"]["server_path"]
    model = load_adflip(weights, device)

    def run(row, pdb, npz, target, seed):
        return run_adflip(seed, [pdb, weights, target / "adflip.json", target / "adflip_work",
                                 device, str(samples), str(config["max_steps"]), str(config["threshold"])],
                          model=model)

    return run


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model", choices=("e18", "ligandmpnn", "lasermpnn", "adflip"))
    cli.add_argument("config", type=Path)
    cli.add_argument("cases", type=Path)
    cli.add_argument("out", type=Path)
    cli.add_argument("warmup", type=Path)
    args = cli.parse_args()
    run_config = json.loads(args.config.read_text())
    cases = json.loads(args.cases.read_text())
    warmup = json.loads(args.warmup.read_text())
    model = args.model
    params = run_config["models"][model]
    samples, device = run_config["samples"], run_config["device"]
    engine = {"e18": e18_engine, "ligandmpnn": ligandmpnn_engine, "lasermpnn": lasermpnn_engine,
              "adflip": adflip_engine}[model]
    extra = [Path(run_config["runtime_artifacts"]["ligandmpnn_cli"]["path"])] if model == "ligandmpnn" else []
    started = perf_counter()
    infer = engine(device, params, samples, *extra)
    load_seconds = now(device) - started
    args.out.mkdir(parents=True, exist_ok=True)

    def run(row, out):
        pdb = Path(row["pdb"])
        npz = Path(row["npz"])
        if digest(pdb) != row["input_sha256"] or digest(npz) != row["e18_input_sha256"]:
            raise ValueError("Input differs from the locked case")
        started = now(device)
        seconds = infer(row, pdb, npz, out, row["seed"])
        return seconds, now(device) - started

    run(warmup, Path(mkdtemp(prefix=".warmup.", dir=args.out)))
    atomic_json(args.out / "runtime.json", {"load_seconds": load_seconds, "device": torch.cuda.get_device_name(0)
                                          if torch.cuda.is_available() else device, "warmup_cases": 1})
    for index, row in enumerate(cases, start=1):
        final = args.out / row["key"]
        if not final.exists():
            temp = Path(mkdtemp(prefix=f".{row['key']}.", dir=args.out))
            seconds, total = run(row, temp)
            atomic_json(temp / "timing.json", {"key": row["key"], "samples": samples, "warm_seconds": seconds,
                                              "case_wall_seconds": total, "seed": row["seed"]})
            temp.rename(final)
        print(f"{model}: completed {index}/{len(cases)}", flush=True)


if __name__ == "__main__":
    main()
