#!/usr/bin/env python
"""本机 in-process LIBERO eval, 支持 A0(无hint)/A1(dino)/A2(so400m) 在线 hint 注入.
每帧: live agentview(flip=both 正立) → HintComputer → hint → obs["lmwm_hint"] → policy.infer → env.step(inv_pm1 gripper).
用法: local_eval_hint.py --config pi05_libero_a1_prefix_eval --ckpt <dir> --encoder dinov3-base --tasks -1 --trials 5 --seed 0 --gpu 0 --out <dir>
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.abspath("kai0/src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", default="none", help="none(A0) | dinov3-base(A1) | so400m(A2)")
    ap.add_argument("--tasks", type=int, default=-1)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ.setdefault("LIBERO_CONFIG_PATH", os.environ["LIBERO_HOME"] + "/libero")

    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from examples.LIBERO.eval_files.libero_benchmark_adapters import get_benchmark_adapter, quat2axisangle
    from libero.libero import benchmark as bm

    policy = _pc.create_trained_policy(_config.get_config(args.config), args.ckpt)
    print(f"[eval] policy {args.config} loaded", flush=True)
    hc = None
    if args.encoder != "none":
        from examples.LIBERO.eval_files.hint_online import HintComputer
        hc = HintComputer(args.encoder, "cuda")

    adapter = get_benchmark_adapter("libero")
    ts = bm.get_benchmark_dict()[args.suite]()
    n_tasks = ts.n_tasks if args.tasks < 0 else min(args.tasks, ts.n_tasks)
    max_steps = adapter.get_max_steps(args.suite)

    def flip(a): return a[::-1, ::-1]
    def build_obs(obs, prompt):
        img = np.ascontiguousarray(flip(obs["agentview_image"]))
        e = {"observation/image": img,
             "observation/wrist_image": np.ascontiguousarray(flip(obs["robot0_eye_in_hand_image"])),
             "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                 np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1))).astype(np.float32),
             "prompt": prompt}
        if hc is not None:
            e["lmwm_hint"] = hc.compute(img)[None].astype(np.float32)  # [1, DIN]
        return e

    out_dir = os.path.join(args.out, "suites", args.suite); os.makedirs(out_dir, exist_ok=True)
    jf = open(os.path.join(out_dir, "episodes.jsonl"), "w")
    per_task = {}
    for task_id in range(n_tasks):
        task = ts.get_task(task_id); init = ts.get_task_init_states(task_id); prompt = task.language
        succ = 0
        for trial in range(args.trials):
            env, _ = adapter.build_env(task, 512, args.seed)
            try:
                env.reset(); obs = env.set_init_state(init[trial % len(init)])
                chunk, done, t = None, False, 0
                for step in range(max_steps + 10):
                    if t < 10:
                        obs, _, done, _ = env.step([0.]*6 + [-1.]); t += 1; continue
                    if chunk is None or (t - 10) % args.replan == 0:
                        chunk = np.asarray(policy.infer(build_obs(obs, prompt))["actions"])
                    a = chunk[(t - 10) % args.replan][:7].copy()
                    a[6] = 1.0 - 2.0 * (a[6] > 0.5)  # inv_pm1
                    obs, _, done, _ = env.step(a.tolist()); t += 1
                    if done: break
            finally:
                try: env.close()
                except Exception: pass
            jf.write(json.dumps({"task_id": task_id, "trial": trial, "success": bool(done)}) + "\n"); jf.flush()
            succ += int(done)
        per_task[task_id] = succ / args.trials
        print(f"[eval] task {task_id}: SR={per_task[task_id]:.3f} ({succ}/{args.trials})", flush=True)
    jf.close()
    agg = float(np.mean(list(per_task.values())))
    json.dump({"per_task_SR": per_task, "aggregate_SR": agg, "t6": per_task.get(6), "seed": args.seed},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(f"[eval] DONE agg_SR={agg:.4f} t6={per_task.get(6)}", flush=True)


if __name__ == "__main__":
    main()
