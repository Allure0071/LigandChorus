"""Replay the 2026-11-27 E18 experiments on the project server."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from statistics import median, quantiles
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = HERE / "scripts"
CONFIGS = HERE / "configs"
RESULTS = HERE / "results"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same(path, expected):
    if sha(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def dump(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def invoke(command, gpu, imports, cwd=None, python=sys.executable):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, imports)), CUDA_VISIBLE_DEVICES=gpu,
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([python, "-B", "-u", *map(str, command)], env=env, cwd=cwd, check=True)


def local_runtime_artifact(name, artifact):
    # The resident worker is distinct from its earlier serial counterpart.
    if name in {"worker", "adflip_upstream_entry"}:
        return None
    candidate = SCRIPTS / Path(artifact["path"]).name
    return candidate if candidate.is_file() else None


def verify(assets, server=False):
    cfg = read(CONFIGS / "resident_benchmark.json")
    artifacts = cfg["runtime_artifacts"]
    for name, artifact in artifacts.items():
        script = local_runtime_artifact(name, artifact)
        if script is not None:
            same(script, artifact["sha256"])
    for name, path in (("scoring", REPO / "ligandchorus/benchmark.py"),
                       ("e18_sampler", REPO / "ligandchorus/sampling.py"),
                       ("e18_cli", REPO / "ligandchorus/infer.py")):
        same(path, artifacts[name]["sha256"])
    for entry in (HERE / "audit/training_code_sha256.txt").read_text().splitlines():
        digest, path = entry.split(maxsplit=1)
        same(REPO / path.removeprefix("code/"), digest)
    for script in [Path(__file__), *SCRIPTS.glob("*.py")]:
        ast.parse(script.read_text(encoding="utf-8-sig"), filename=str(script))

    selection = read(RESULTS / "calibration_selection.json")
    same(SCRIPTS / "calibrate.py", read(RESULTS / "calibration_summary.json")["runner_sha256"])
    lock = read(CONFIGS / "temperature_lock.json")
    same(CONFIGS / "calibration.json", selection["protocol_sha256"])
    same(RESULTS / "calibration_summary.json", selection["summary_sha256"])
    same(RESULTS / "calibration_selection.json", lock["calibration"]["selection_sha256"])
    same(CONFIGS / "temperature_lock.json", cfg["calibration_lock"]["sha256"])
    receipt = read(RESULTS / "benchmark_receipt.json")
    same(CONFIGS / "resident_benchmark.json", receipt["config_sha256"])
    assert receipt["status"] == "complete" and receipt["total_cases"] == 140
    for name in ("scores", "timings"):
        same(RESULTS / f"{name}.json", assets["files"][f"formal_{'score' if name == 'scores' else 'timing'}_report"]["sha256"])
    same(RESULTS / "training.log", assets["files"]["training_log"]["sha256"])
    summaries, timings = (read(RESULTS / f"{name}.json") for name in ("scores", "timings"))
    assert {group: row["structures"] for group, row in summaries.items()} == {"enzyme": 80, "nonenzyme": 60}
    assert len(timings["cases"]) == 140 and timings["samples"] == cfg["samples"] == 10
    for group, report in summaries.items():
        assert set(report["summary"]) == set(cfg["models"])
        group_timings = [row for row in timings["cases"] if row["test_group"] == group]
        assert len(group_timings) == report["structures"]
        assert {row["key"] for row in report["cases"]} == {row["key"] for row in group_timings}
        for model in cfg["models"]:
            for metric in ("warm_seconds", "case_wall_seconds"):
                values = [row["models"][model][metric] for row in group_timings]
                summary = timings["groups"][group][model][metric]
                assert summary["median"] == median(values)
                assert summary["p90"] == quantiles(values, n=10, method="inclusive")[8]
    if server:
        for artifact in assets["files"].values():
            path = Path(artifact["path"])
            if path.stat().st_size != artifact["bytes"]:
                raise ValueError(f"Size mismatch: {path}")
            same(path, artifact["sha256"])
        for path in map(Path, assets["directories"].values()):
            if not path.is_dir():
                raise FileNotFoundError(path)
        for artifact in artifacts.values():
            same(Path(artifact["path"]), artifact["sha256"])
        shards = [p for p in Path(assets["directories"]["training_shards"]).iterdir() if p.is_file()]
        assert len(shards) == assets["training_shard_inventory"]["file_count"]
        assert sum(p.stat().st_size for p in shards) == assets["training_shard_inventory"]["bytes"]
    print("Local source, locks and result checks passed." if not server else
          "Local source, locks, result checks and server asset checks passed.", flush=True)


def training(out, gpu, assets, smoke):
    info = read(CONFIGS / "training.json")
    files = assets["files"]
    for key in ("training_manifest", "enzyme_training_targets", "validation_panel", "official_ligandmpnn"):
        same(Path(files[key]["path"]), files[key]["sha256"])
    out.mkdir(parents=True)
    invoke([
        "-m", "ligandchorus.train", "--manifest", files["training_manifest"]["path"],
        "--shard-root", assets["directories"]["training_shard_root"],
        "--train-targets", files["enzyme_training_targets"]["path"],
        "--validation-manifest", files["validation_panel"]["path"],
        "--encoder-weights", files["official_ligandmpnn"]["path"],
        "--structures", 2 if smoke else info["training_assemblies"],
        "--steps", 1 if smoke else info["optimizer_steps"],
        "--batch-size", info["batch_size"], "--learning-rate", info["learning_rate"],
        "--validate-every", info["validate_every"], "--seed", info["training_seed"],
        "--validation-seed", info["validation_seed"], "--device", "cuda:0",
        "--output", out / "last.pt", "--best-output", out / "best.pt",
    ], gpu, [REPO])


def inference(out, gpu, assets, fixture):
    cfg = read(CONFIGS / "temperature_lock.json")["e18"]
    artifact = cfg["checkpoint"]
    same(Path(artifact["server_path"]), artifact["sha256"])
    record = fixture or Path(assets["directories"]["development_fixtures"]) / "food/blinded_e18_input.npz"
    if out.exists():
        raise FileExistsError(out)
    invoke(["-m", "ligandchorus.infer", "--input", record, "--weights", artifact["server_path"],
            "--device", "cuda:0", "--samples", "10", "--steps", cfg["max_steps"], "--seed", "1",
            "--inside-temperature", cfg["inside_temperature"], "--outside-temperature", cfg["outside_temperature"],
            "--output", out], gpu, [REPO])


def calibrate(out, gpu, assets, smoke):
    protocol = CONFIGS / "calibration.json"
    invoke([SCRIPTS / "calibrate.py", "--formal", assets["directories"]["enzyme_validation_run"],
            "--shard-root", assets["directories"]["training_shard_root"],
            "--protocol", protocol, "--output", out, *(["--smoke"] if smoke else [])], gpu, [REPO])


def prepare_inputs(out, assets):
    """Build coordinates from the frozen candidates and check all four parsers on CPU."""
    files, dirs = assets["files"], assets["directories"]
    manifest = files["coordinate_candidate_manifest"]
    panel = files["formal_input_manifest"]
    same(Path(manifest["path"]), manifest["sha256"])
    same(Path(panel["path"]), panel["sha256"])
    out.mkdir(parents=True)
    common = [REPO, assets["runtime"]["input_parser_dependencies"]]
    adflip = Path(dirs["adflip_source"])
    adflip_imports = [adflip, assets["runtime"]["adflip_dependencies"], *common]
    audited = out / "coordinate_audit.json"
    invoke([SCRIPTS / "e18_four_model_coordinate_audit.py", manifest["path"], manifest["sha256"],
            adflip, audited], "", adflip_imports)
    rebuilt = out / "inputs"
    invoke([SCRIPTS / "e18_materialize_formal_inputs.py", manifest["path"], audited, rebuilt], "", adflip_imports)
    staged = [json.loads(line) for line in (rebuilt / "inputs.jsonl").read_text().splitlines()]
    published = [json.loads(line) for line in Path(panel["path"]).read_text().splitlines()]
    fields = ("key", "input_sha256", "e18_input_sha256", "alignment_sha256", "test_group", "family_id")
    if len(staged) != len(published) or any(any(a[key] != b[key] for key in fields)
                                           for a, b in zip(staged, published)):
        raise ValueError("Regenerated inputs differ from the locked formal inputs")
    ligandmpnn = Path(dirs["ligandmpnn_source"])
    laser = Path(dirs["lasermpnn_source"])
    imports = {"e18": common, "ligandmpnn": [ligandmpnn, *common],
               "lasermpnn": [laser.parent, laser, *common], "adflip": adflip_imports}
    for name, paths in imports.items():
        report = out / f"{name}_parser.jsonl"
        python = assets["runtime"]["benchmark_python"] if name in {"lasermpnn", "adflip"} else sys.executable
        invoke([SCRIPTS / "e18_audit_formal_input_parsers.py", name, rebuilt / "inputs.jsonl", report],
               "", paths, cwd=adflip if name == "adflip" else None, python=python)
        audits = [json.loads(line) for line in report.read_text().splitlines()]
        if [x["key"] for x in audits] != [x["key"] for x in published] or any(x["status"] != "pass" for x in audits):
            raise ValueError(f"{name}: regenerated input parser checks differ")
    print("Rebuilt 140 shared inputs and reproduced the four-model parser checks.", flush=True)


def benchmark(out, gpu, assets, smoke):
    if gpu != "0":
        raise ValueError("The locked timing protocol calls for physical RTX 4090 D GPU 0")
    out.mkdir(parents=True)
    config = read(CONFIGS / "resident_benchmark.json")
    config["development_report"] = str(out / "development/progress.json")
    for name, artifact in config["runtime_artifacts"].items():
        local = local_runtime_artifact(name, artifact)
        if local is not None:
            same(local, artifact["sha256"])
            artifact["path"] = str(local)
    runtime = out / "runtime/ligandchorus"
    runtime.mkdir(parents=True)
    for item in (REPO / "ligandchorus").iterdir():
        if item.name == "benchmark_outputs.py":
            continue
        if item.name != "__pycache__":
            (runtime / item.name).symlink_to(item, target_is_directory=item.is_dir())
    (runtime / "benchmark_outputs.py").symlink_to(SCRIPTS / "resident_benchmark_outputs.py")
    config_path = out / "resident_config.json"
    dump(config_path, config)
    imports = [runtime.parent, assets["runtime"]["input_parser_dependencies"]]
    runner = SCRIPTS / "e18_resident_benchmark.py"
    invoke([runner, config_path, out / "development", "development"], gpu, imports)
    if not smoke:
        invoke([runner, config_path, out / "formal", "formal"], gpu, imports)


def rescore(assets):
    sys.path[:0] = [str(SCRIPTS), str(REPO)]
    from e18_formal_benchmark_runner import rows_for_panel, score_cases

    config = read(CONFIGS / "resident_benchmark.json")
    cases = list(rows_for_panel(config))
    original = read(RESULTS / "scores.json")
    repeated = score_cases(cases, Path(assets["directories"]["normalized_predictions"]), config["samples"])
    if repeated != original:
        raise ValueError("Re-scoring differs from the published result")
    print("Re-scoring reproduced all numeric results for 80 enzyme and 60 nonenzyme structures.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify").add_argument("--server", action="store_true")
    commands.add_parser("rescore")
    commands.add_parser("prepare-inputs").add_argument("--output", type=Path, required=True)
    for name in ("train", "infer", "calibrate", "benchmark"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--gpu", default="0")
        if name == "infer":
            cmd.add_argument("--input", type=Path)
        else:
            cmd.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assets = read(HERE / "assets.json")
    if args.command == "verify":
        verify(assets, args.server)
    elif args.command == "rescore":
        rescore(assets)
    elif args.command == "prepare-inputs":
        prepare_inputs(args.output.resolve(), assets)
    else:
        handler = {"train": training, "infer": inference, "calibrate": calibrate, "benchmark": benchmark}[args.command]
        handler(args.output.resolve(), args.gpu, assets, args.input if args.command == "infer" else args.smoke)


if __name__ == "__main__":
    main()
