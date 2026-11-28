"""Run the agreed four-model comparison in resident, same-device model phases."""

import argparse
import json
from pathlib import Path
from statistics import median, quantiles
from tempfile import mkdtemp
from types import SimpleNamespace

from e18_formal_benchmark_runner import (
    BASELINE_ROOT, ENV, MODELS, SEED_BASE, atomic_json, call, digest,
    rows_for_development, rows_for_panel, score_cases,
)
from ligandchorus.benchmark_outputs import normalize


def make_tasks(cases):
    for index, (row, case, _) in enumerate(cases):
        if digest(case / "alignment.json") != row["alignment_sha256"]:
            raise ValueError(f"{row['key']}: locked input mapping differs")
        yield dict(row, pdb=str(case / row["pdb_name"]), npz=str(case / row["npz_name"]),
                   seed=SEED_BASE + index)


def run(config_path, out, mode):
    config = json.loads(config_path.read_text())
    samples, server = config["samples"], Path(config["server_root"])
    for model, settings in config["models"].items():
        artifact = settings["checkpoint"]
        if digest(artifact["server_path"]) != artifact["sha256"]:
            raise ValueError(f"{model}: checkpoint differs from the lock")
    for name, artifact in config["runtime_artifacts"].items():
        if digest(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"{name}: runtime differs from the lock")
    config_sha = digest(config_path)
    if mode == "formal":
        receipt = json.loads(Path(config["development_report"]).read_text())
        if receipt["config_sha256"] != config_sha or receipt["status"] != "complete":
            raise ValueError("The matching end-to-end development run must pass first")
    out.mkdir(exist_ok=True)
    status_file = out / "progress.json"
    if status_file.exists():
        previous = json.loads(status_file.read_text())
        if previous["config_sha256"] != config_sha or previous["mode"] != mode:
            raise ValueError("The target directory belongs to another benchmark")
    cases = list(rows_for_development(server) if mode == "development" else rows_for_panel(config))
    tasks = list(make_tasks(cases))
    warmup = next(make_tasks(rows_for_development(server)))
    warmup["seed"] = SEED_BASE - 1
    atomic_json(out / "case_tasks.json", tasks)
    atomic_json(out / "warmup_task.json", warmup)
    phase_dir, normalized_dir = out / "models", out / "cases"
    phase_dir.mkdir(exist_ok=True)
    normalized_dir.mkdir(exist_ok=True)
    status = {"status": "running", "mode": mode, "config_sha256": config_sha,
              "total_cases": len(cases), "samples_per_model": samples, "completed_models": []}
    overlay = config_path.parent / "runtime"
    worker = Path(__file__).with_name("e18_resident_benchmark_worker.py")
    imports = {
        "e18": [overlay], "ligandmpnn": [overlay, BASELINE_ROOT / "LigandMPNN"],
        "lasermpnn": [overlay, BASELINE_ROOT, BASELINE_ROOT / "LASErMPNN"],
        "adflip": [overlay, server / "adflip", server, server / "adflip_deps"],
    }
    try:
        for model in MODELS:
            status["phase"] = model
            atomic_json(status_file, status)
            log = out / f"{model}.log"
            phase = phase_dir / model
            call(log, [ENV, "-B", "-u", worker, model, config_path, out / "case_tasks.json",
                       phase, out / "warmup_task.json"], imports[model],
                 cwd=server / "adflip" if model == "adflip" else None)
            if any(not (phase / row["key"] / "timing.json").is_file() for row in tasks):
                raise ValueError(f"{model}: missing completed case records")
            status["completed_models"].append(model)
        status["phase"] = "normalization_and_scoring"
        atomic_json(status_file, status)
        for row, case, _ in cases:
            key = row["key"]
            model_dirs = {m: phase_dir / m / key for m in MODELS}
            pdb = case / row["pdb_name"]
            case_out = normalized_dir / key
            if case_out.exists():
                continue
            temp = Path(mkdtemp(prefix=f".{key}.", dir=normalized_dir))
            normalize(SimpleNamespace(
                alignment=case / "alignment.json", backbone=pdb, e18=model_dirs["e18"] / "e18.npy",
                ligandmpnn=model_dirs["ligandmpnn"] / "seqs" / f"{pdb.stem}.fa",
                lasermpnn=model_dirs["lasermpnn"], adflip=model_dirs["adflip"] / "adflip.json",
                lasermpnn_ca_reference=model_dirs["lasermpnn"] / "idealized_ca.json",
                output=temp / "normalized", samples=samples, proteinmpnn=None, rfd3=None,
            ))
            temp.rename(case_out)
        report = score_cases(cases, normalized_dir, samples)
        atomic_json(out / "scores.json", report)
        time_report = {"device": config["hardware"], "samples": samples, "method": config["timing_method"],
                       "groups": {}, "model_load": {}, "cases": []}
        for m in MODELS:
            time_report["model_load"][m] = json.loads((phase_dir / m / "runtime.json").read_text())
        for row in tasks:
            t = {}
            for m in MODELS:
                v = json.loads((phase_dir / m / row["key"] / "timing.json").read_text())
                if v["seed"] != row["seed"] or v["samples"] != samples:
                    raise ValueError("Timing does not belong to this locked case")
                if not 0 < v["warm_seconds"] <= v["case_wall_seconds"]:
                    raise ValueError("Invalid timing")
                t[m] = v
            time_report["cases"].append({"key": row["key"], "test_group": row["test_group"], "models": t})
        for group in sorted(report):
            rows = [row for row in time_report["cases"] if row["test_group"] == group]
            time_report["groups"][group] = {}
            for model in MODELS:
                time_report["groups"][group][model] = {}
                for field in ("warm_seconds", "case_wall_seconds"):
                    values = [row["models"][model][field] for row in rows]
                    time_report["groups"][group][model][field] = {
                        "median": median(values), "p90": quantiles(values, n=10, method="inclusive")[8]
                        if len(values) > 1 else None,
                    }
        atomic_json(out / "timings.json", time_report)
        atomic_json(status_file, dict(status, status="complete", phase="done"))
    except Exception as error:
        atomic_json(status_file, dict(status, status="failed", error=str(error)))
        raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("config", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("mode", choices=("development", "formal"))
    args = cli.parse_args()
    run(args.config.resolve(), args.output.resolve(), args.mode)
