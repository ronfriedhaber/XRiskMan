"""Launch explicitly: uv run --group modal modal run modal_app.py --steps 1000000."""
import json
from dataclasses import asdict, replace
from typing import Optional
import os
from pathlib import Path
import tomllib
import uuid

import modal

from config import ExperimentConfig

app = modal.App("pace-rl")
volume = modal.Volume.from_name("pace-rl-runs", create_if_missing=True)
# Training dependencies install inside Modal, not on the launching machine.
project = tomllib.loads(Path(__file__).with_name("pyproject.toml").read_text())
dependencies = project["project"]["dependencies"] + project["dependency-groups"]["train"]
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install(*dependencies)
         .add_local_file(Path(__file__).with_name("pyproject.toml"), "/root/pyproject.toml"))
for module in ("config", "pace", "train"):
    image = image.add_local_file(Path(__file__).with_name(f"{module}.py"), f"/root/{module}.py")


@app.function(image=image, gpu="T4", cpu=4, memory=8192, timeout=4 * 60 * 60,
              max_containers=1, retries=0, volumes={"/runs": volume},
              secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])])
def train_remote(experiment: dict):
    from train import main
    import wandb

    run = Path(experiment["training"]["out"]).name
    try:
        # W&B reads WANDB_API_KEY (and optional WANDB_ENTITY) from Modal's Secret.
        with wandb.init(project=os.environ.get("WANDB_PROJECT", "pace-rl"),
                        id=run, name=run, dir="/runs") as tracker:
            result = main(["--out", f"/runs/{run}"], experiment=ExperimentConfig.from_dict(experiment),
                          on_checkpoint=volume.commit, tracker=tracker)
            wandb_url = tracker.url
    finally:
        volume.commit()
    return {"run": run, "wandb_url": wandb_url, "evaluation": result,
            "download": f"modal volume get pace-rl-runs {run}/checkpoint.pt ./checkpoint.pt"}


@app.local_entrypoint()
def launch(config: str = "", steps: Optional[int] = None, seed: Optional[int] = None,
           alpha: Optional[float] = None, beta: Optional[float] = None):
    settings = asdict(ExperimentConfig.read(config or None))
    run = Path(settings["training"]["out"]).name if settings["training"]["out"].startswith("runs/") else uuid.uuid4().hex
    for section, changes in (("training", {"steps": steps, "seed": seed}), ("reward", {"alpha": alpha, "beta": beta})):
        settings[section].update({k: v for k, v in changes.items() if v is not None})
    experiment = ExperimentConfig.from_dict(settings)  # Validate before starting GPU work.
    experiment = ExperimentConfig(experiment.environment, experiment.model,
                                  replace(experiment.training, out=f"runs/{run}"), experiment.reward)
    print(json.dumps(train_remote.remote(asdict(experiment)), indent=2))
