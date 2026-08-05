"""FastWAM isolated deployment server.

This module wraps the original ``serve_fastwam_ws.py`` without changing it.
Only the two gripper dimensions are post-processed. Arm joint predictions are
returned bit-identically.

The online failure this addresses is receding-horizon gripper procrastination:
the model repeatedly predicts a closed gripper at h0 but an open gripper around
h24. Since the controller replans before h24 is executed, the open event never
reaches hardware.

The first isolated implementation used a rolling future maximum for every
inference step. That fixed opening, but it also kept overwriting later closing
intent and left both grippers open for the whole task. This revision performs
one bootstrap per server session: while a gripper has never actually opened,
copy one future opening target into only the controller's executed prefix.
Once proprioception confirms opening, that gripper is permanently passed
through bit-identically, including all later closing commands.
"""

import argparse
import glob
import os
import time

import numpy as np

from serve_fastwam_ws import FastwamPolicy, WebsocketPolicyServer


GRIPPER_DIMS = (6, 13)


def apply_initial_open_bootstrap(
    actions: np.ndarray,
    state: np.ndarray,
    bootstrap_done: np.ndarray,
    lookahead: int,
    trigger: float,
    prefix_steps: int,
    open_confirm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap initial opening, then permanently restore raw model commands.

    ``bootstrap_done`` contains one flag per gripper. A flag becomes true only
    after real gripper proprioception reaches ``open_confirm``. Until then, a
    future opening target may be copied into the first ``prefix_steps`` actions,
    which is the portion consumed before the next replan. No arm dimension and
    no action after the prefix is changed.

    Returns ``(fixed, next_done, boosted)``. The function is pure so the state
    transitions can be tested without loading the model.
    """
    raw = np.asarray(actions, dtype=np.float32)
    proprio = np.asarray(state, dtype=np.float32).reshape(-1)
    done = np.asarray(bootstrap_done, dtype=bool).reshape(-1).copy()
    if raw.ndim != 2 or raw.shape[1] < 14:
        raise ValueError(f"expected [H,14+] actions, got {raw.shape}")
    if proprio.size < 14:
        raise ValueError(f"expected 14D state, got {proprio.shape}")
    if done.size != len(GRIPPER_DIMS):
        raise ValueError(f"expected {len(GRIPPER_DIMS)} bootstrap flags, got {done.shape}")

    fixed = raw.copy()
    horizon = raw.shape[0]
    window = min(max(int(lookahead), 0), horizon - 1)
    prefix = min(max(int(prefix_steps), 0), horizon)
    threshold = float(trigger)
    confirm = float(open_confirm)
    boosted = np.zeros(len(GRIPPER_DIMS), dtype=bool)

    for side, dim in enumerate(GRIPPER_DIMS):
        if done[side]:
            continue
        if float(proprio[dim]) >= confirm:
            done[side] = True
            continue
        if window <= 0 or prefix <= 0:
            continue
        future_open = float(np.max(raw[:window + 1, dim]))
        if future_open >= threshold:
            fixed[:prefix, dim] = np.maximum(fixed[:prefix, dim], future_open)
            boosted[side] = True
    return fixed, done, boosted


class IsolatedFastwamPolicy(FastwamPolicy):
    def __init__(self, args):
        super().__init__(args)
        self.reset_gripper_bootstrap()
        self._gripper_no_close_chunks = np.zeros(len(GRIPPER_DIMS), dtype=np.int64)

    def reset_gripper_bootstrap(self) -> None:
        self._gripper_bootstrap_done = np.zeros(len(GRIPPER_DIMS), dtype=bool)

    def reset_session(self) -> None:
        """Reset per-trial state while retaining weights and compiled graphs."""
        self.reset_gripper_bootstrap()
        self._gripper_no_close_chunks.fill(0)
        print("[serve_fastwam_isolated] new client session: bootstrap reset", flush=True)

    def infer(self, obs: dict) -> dict:
        result = super().infer(obs)
        raw = np.asarray(result["actions"], dtype=np.float32)
        previous_done = self._gripper_bootstrap_done.copy()
        if self.args.disable_gripper_bootstrap:
            fixed = raw.copy()
            next_done = previous_done
            boosted = np.zeros(len(GRIPPER_DIMS), dtype=bool)
        else:
            fixed, next_done, boosted = apply_initial_open_bootstrap(
                raw,
                state=np.asarray(obs["state"], dtype=np.float32),
                bootstrap_done=previous_done,
                lookahead=self.args.gripper_open_lookahead,
                trigger=self.args.gripper_open_trigger,
                prefix_steps=self.args.gripper_bootstrap_prefix,
                open_confirm=self.args.gripper_open_confirm,
            )
        self._gripper_bootstrap_done = next_done
        result["actions"] = fixed

        n = getattr(self, "_n", 0)
        state_grip = np.asarray(obs["state"], dtype=np.float32)[list(GRIPPER_DIMS)] * 1000.0
        for side, dim in enumerate(GRIPPER_DIMS):
            if not next_done[side]:
                self._gripper_no_close_chunks[side] = 0
                continue
            has_future_close = bool(
                np.min(raw[:, dim]) <= self.args.gripper_close_trigger
            )
            if has_future_close:
                self._gripper_no_close_chunks[side] = 0
            else:
                self._gripper_no_close_chunks[side] += 1
                warn_after = self.args.gripper_no_close_warn_after
                count = int(self._gripper_no_close_chunks[side])
                if warn_after > 0 and (
                    count == warn_after or count % (warn_after * 2) == 0
                ):
                    label = "L" if side == 0 else "R"
                    print(
                        f"[gripper-diagnostic #{n}] WARN {label}: "
                        f"{count} consecutive chunks contain no raw close target "
                        f"<= {self.args.gripper_close_trigger * 1000:.1f}mm; "
                        "post-processing is already raw passthrough",
                        flush=True,
                    )
        if np.any(next_done != previous_done):
            print(
                f"[gripper-bootstrap #{n}] confirmed-open L/R="
                f"{next_done.tolist()} proprio_mm={state_grip.round(1).tolist()} "
                "-> permanent raw passthrough",
                flush=True,
            )
        if n <= 5 or n % 20 == 0:
            before = raw[0, list(GRIPPER_DIMS)] * 1000.0
            after = fixed[0, list(GRIPPER_DIMS)] * 1000.0
            print(
                f"[gripper-bootstrap #{n}] h0 mm "
                f"L/R {before.round(1).tolist()} -> {after.round(1).tolist()} "
                f"boosted={boosted.tolist()} done={next_done.tolist()} "
                f"(initial-only, lookahead={self.args.gripper_open_lookahead}, "
                f"prefix={self.args.gripper_bootstrap_prefix})",
                flush=True,
            )

        if self.args.debug_dump_dir and n <= self.args.debug_dump_n:
            os.makedirs(self.args.debug_dump_dir, exist_ok=True)
            np.savez(
                os.path.join(self.args.debug_dump_dir, f"post_{n:03d}.npz"),
                action_raw=raw,
                action_fixed=fixed,
                gripper_state=np.asarray(obs["state"], dtype=np.float32)[list(GRIPPER_DIMS)],
                bootstrap_done=next_done,
                bootstrap_boosted=boosted,
                no_close_chunks=self._gripper_no_close_chunks.copy(),
            )
        return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8016)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--stats", default="data/visrobot01_fold/dataset_stats.json")
    ap.add_argument("--t5_cache", default="data/text_embeds_cache/visrobot01_fold/*.pt")
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--opt_tier", default="exact", choices=["eager", "exact", "fp8"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--gripper_proprio_neutral", type=float, default=None)
    ap.add_argument("--gripper_open_lookahead", type=int, default=24)
    ap.add_argument("--gripper_open_trigger", type=float, default=0.01)
    ap.add_argument("--gripper_bootstrap_prefix", type=int, default=8)
    ap.add_argument("--gripper_open_confirm", type=float, default=0.015)
    ap.add_argument("--gripper_close_trigger", type=float, default=0.005)
    ap.add_argument("--gripper_no_close_warn_after", type=int, default=120)
    ap.add_argument("--disable_gripper_bootstrap", action="store_true")
    ap.add_argument("--debug_dump_dir", default="")
    ap.add_argument("--debug_dump_n", type=int, default=200)
    args = ap.parse_args()
    if not 1 <= args.gripper_open_lookahead <= 47:
        ap.error("--gripper_open_lookahead must be in [1, 47]")
    if not 1 <= args.gripper_bootstrap_prefix <= 48:
        ap.error("--gripper_bootstrap_prefix must be in [1, 48]")
    if args.gripper_bootstrap_prefix > args.gripper_open_lookahead + 1:
        ap.error(
            "--gripper_bootstrap_prefix cannot exceed "
            "--gripper_open_lookahead + 1"
        )
    if not 0.0 <= args.gripper_open_trigger <= 0.08:
        ap.error("--gripper_open_trigger must be in [0.0, 0.08] meters")
    if not 0.0 <= args.gripper_open_confirm <= 0.08:
        ap.error("--gripper_open_confirm must be in [0.0, 0.08] meters")
    if not 0.0 <= args.gripper_close_trigger <= 0.08:
        ap.error("--gripper_close_trigger must be in [0.0, 0.08] meters")
    if args.gripper_no_close_warn_after < 0:
        ap.error("--gripper_no_close_warn_after must be >= 0")
    if args.debug_dump_n < 0:
        ap.error("--debug_dump_n must be >= 0")

    policy = IsolatedFastwamPolicy(args)
    if args.warmup:
        dummy = {
            "state": np.zeros(14, np.float32),
            "images": {
                key: np.zeros((3, 240, 320), np.uint8)
                for key in ("top_head", "hand_left", "hand_right")
            },
        }
        for i in range(int(args.warmup)):
            start = time.monotonic()
            response = policy.infer(dummy)
            print(
                f"[serve_fastwam_isolated] warmup {i}: "
                f"{response['actions'].shape} {(time.monotonic() - start) * 1e3:.0f}ms",
                flush=True,
            )
        # Dummy warmup observations must never consume the real-robot one-shot.
        policy.reset_gripper_bootstrap()

    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={
            "model": "fastwam-isolated",
            "action_dim": 14,
            "action_horizon": 48,
            "gripper_open_lookahead": args.gripper_open_lookahead,
            "gripper_open_trigger": args.gripper_open_trigger,
            "gripper_bootstrap_prefix": args.gripper_bootstrap_prefix,
            "gripper_open_confirm": args.gripper_open_confirm,
            "gripper_close_trigger": args.gripper_close_trigger,
            "gripper_no_close_warn_after": args.gripper_no_close_warn_after,
            "gripper_bootstrap_enabled": not args.disable_gripper_bootstrap,
        },
    ).serve_forever()


if __name__ == "__main__":
    main()
