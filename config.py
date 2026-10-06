"""Experiment defaults. JSON sections: environment, model, training, reward."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentConfig:
    dt: float = 1 / 60
    decision_dt: float = 0.1
    duration: float = 90.0
    delay: float = 2.0
    acceleration: float = 0.62
    deceleration: float = 0.62
    initial_safety: float = 12.0
    safety_flat_until: float = 18.0
    safety_plateaus: bool = True
    max_hazard: float = 0.24

    def __post_init__(self):
        if not isinstance(self.safety_plateaus, bool):
            raise ValueError("safety_plateaus must be boolean")
        for name, value in vars(self).items():
            if name != "safety_plateaus" and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.dt, self.decision_dt, self.duration, self.acceleration, self.deceleration) <= 0:
            raise ValueError("time steps, duration, and acceleration/deceleration must be positive")
        if self.dt > 1:
            raise ValueError("dt must be <= 1 second")
        for name in ("decision_dt", "duration", "delay"):
            ratio = getattr(self, name) / self.dt
            if not math.isclose(ratio, round(ratio), abs_tol=1e-8, rel_tol=0):
                raise ValueError(f"{name} must be an integer multiple of dt")


@dataclass(frozen=True)
class ModelConfig:
    context: int = 32
    width: int = 64
    heads: int = 4
    layers: int = 2
    input_dim: int = 16

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.context, self.width, self.heads, self.layers)) or self.width % self.heads:
            raise ValueError("positive dimensions required; width must be divisible by heads")
        if type(self.input_dim) is not int or self.input_dim not in (16, 23):
            raise ValueError("input_dim must be 16, or 23 for legacy checkpoints")


@dataclass(frozen=True)
class TrainingConfig:
    # Run / runtime
    steps: int = 8_388_608
    seed: int = 0
    out: str = "runs/pace"
    device: str = "auto"
    threads: int = 4
    envs: int = 32
    # Imitation
    imitation_updates: int = 5000
    imitation_lr: float = 3e-4
    # PPO
    rollout: int = 1024
    epochs: int = 4
    batch: int = 256
    lr: float = 3e-4
    adam_eps: float = 1e-5
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    # Evaluation / checkpoints
    eval_episodes: int = 128
    eval_every: int = 5

    def __post_init__(self):
        for name in ("steps", "threads", "envs", "rollout", "epochs", "batch", "eval_episodes", "eval_every"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        if type(self.imitation_updates) is not int or self.imitation_updates < 0:
            raise ValueError("imitation_updates must be a nonnegative integer")
        for name in ("lr", "imitation_lr", "adam_eps", "max_grad_norm", "target_kl"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("gamma", "gae_lambda", "clip_ratio"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("value_coef", "entropy_coef"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if not isinstance(self.out, str) or not self.out:
            raise ValueError("out must be a nonempty path string")


@dataclass(frozen=True)
class RewardConfig:
    alpha: float = 0.0
    beta: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in (self.alpha, self.beta)):
            raise ValueError("alpha and beta must be finite and nonnegative")


@dataclass(frozen=True)
class ExperimentConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def read(cls, path=None):
        return cls.from_dict(json.loads(Path(path).read_text()) if path else {})

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("configuration must be a JSON object")
        sections = dict(environment=EnvironmentConfig, model=ModelConfig,
                        training=TrainingConfig, reward=RewardConfig)
        if data.keys() - sections.keys():
            raise ValueError(f"Unknown configuration sections: {data.keys() - sections.keys()}")
        return cls(**{name: kind(**data.get(name, {})) for name, kind in sections.items()})

    def arguments(self):
        return {**asdict(self.training), **asdict(self.model), **asdict(self.reward)}
