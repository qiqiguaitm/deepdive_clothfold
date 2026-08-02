import dataclasses
import enum
import logging
import os
import pathlib
import socket
import sys

import numpy as np

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


class _OnlineRobotwinHintPolicy:
    def __init__(self, policy: _policy.BasePolicy, encoder: str):
        self._policy = policy
        self._encoder = encoder
        self._hint_computer = None
        self._intervention = os.getenv("ROBOTWIN_HINT_INTERVENTION", "correct").strip().lower()
        self._override_hint = None
        if self._intervention == "override":
            path = pathlib.Path(os.environ["ROBOTWIN_HINT_OVERRIDE_PATH"])
            hint = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
            expected_dim = 1152 if encoder == "so400m" else 768
            if hint.shape != (expected_dim,) or not np.all(np.isfinite(hint)):
                raise ValueError(
                    f"Invalid RoboTwin hint override {path}: shape={hint.shape}, "
                    f"expected=({expected_dim},), finite={np.all(np.isfinite(hint))}"
                )
            self._override_hint = hint
            logging.info("Loaded fixed RoboTwin hint override from %s", path)

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        if "lmwm_hint" not in obs:
            if self._intervention == "zero":
                hint_dim = 1152 if self._encoder == "so400m" else 768
                obs["lmwm_hint"] = np.zeros((1, hint_dim), dtype=np.float32)
            elif self._intervention == "override":
                obs["lmwm_hint"] = self._override_hint[None]
            else:
                if self._hint_computer is None:
                    repo = pathlib.Path(__file__).resolve().parents[2]
                    sys.path.insert(0, str(repo / "lmvla" / "lawam"))
                    from examples.Robotwin.eval_files.hint_online_robotwin import RobotwinHintComputer

                    self._hint_computer = RobotwinHintComputer(self._encoder, "cuda")
                head = np.asarray(obs["images"]["cam_high"])
                if head.ndim == 3 and head.shape[0] in (1, 3, 4):
                    head = np.moveaxis(head, 0, -1)
                obs["lmwm_hint"] = self._hint_computer.compute(head)[None].astype(np.float32)
        return self._policy.infer(obs)


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if hint_encoder := os.getenv("OPENPI_SERVER_HINT_ENCODER"):
        logging.info("Enabling server-side RoboTwin hint encoder: %s", hint_encoder)
        policy = _OnlineRobotwinHintPolicy(policy, hint_encoder)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
