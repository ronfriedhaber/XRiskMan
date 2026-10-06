#!/usr/bin/env python3
"""Launch one Modal run and its localhost watcher."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys
import uuid

from config import ExperimentConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run = uuid.uuid4().hex
    data = asdict(ExperimentConfig.read(args.config))
    for section, updates in (("training", {"steps": args.steps, "seed": args.seed}),
                             ("reward", {"alpha": args.alpha, "beta": args.beta})):
        data[section].update({k: v for k, v in updates.items() if v is not None})
    experiment = ExperimentConfig.from_dict(data)
    experiment = replace(experiment, training=replace(experiment.training, out=f"runs/{run}"))
    config_path = Path("/tmp") / f"pace-{run}.json"
    config_path.write_text(json.dumps(asdict(experiment)))
    watcher = subprocess.Popen(["uv", "run", "--group", "train", "--group", "modal",
                                "python", "watch.py", "--follow-run", run,
                                "--port", str(args.port)])
    try:
        command = ["uv", "run", "--group", "modal", "modal", "run", "modal_app.py",
                   "--config", str(config_path)]
        print(f"Run: {run}\nWatcher: http://127.0.0.1:{args.port}", flush=True)
        raise SystemExit(subprocess.call(command))
    finally:
        if watcher.poll() is None:
            watcher.terminate()


if __name__ == "__main__":
    main()
