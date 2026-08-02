#!/usr/bin/env python
"""openpi(pi05) LIBERO rollout 客户端 —— 对接 openpi websocket policy server(serve_policy.py)。

为**最大可比性**(承 libero_eval_wp577 的 starVLA eval): 复用同一 env adapter(build_env/get_max_steps)、
同样的 obs 提取(agentview/wrist flip + eef_pos+quat2axisangle+gripper)、同样的 seed 处理、同样的
episodes.jsonl schema({"task_id","success"})。**唯一变量 = policy 从 starVLA ModelClient 换成 openpi
WebsocketClientPolicy.infer(obs)** —— env/trials/seed/判据全不变, 与之前 LMWM/lawam LIBERO 结果可比。

pi05 侧: server 端做 Normalize + LiberoInputs/Outputs transform(config pi05_libero_*), client 只发原始
obs(observation/image|wrist_image|state + prompt), 收 {"actions": [horizon,7]}。

用法(在装了 LIBERO 的 client env, 且能 import openpi_client + eval_files.libero_benchmark_adapters):
  python eval_libero_openpi.py --host 127.0.0.1 --port 8000 --task-suite-name libero_10 \
      --num-trials-per-task 50 --seed 0 --replan-steps 5 --output-dir <dir>
"""
import argparse, json, logging, os, sys
import numpy as np

# 复用 wp577/starVLA eval 的 env adapter + obs helpers(model-agnostic 部分)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # lmvla/lawam
from examples.LIBERO.eval_files.libero_benchmark_adapters import (  # noqa: E402
    get_benchmark_adapter, quat2axisangle,
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
# ⭐ 必须匹配训练数据渲染分辨率: LIBERO_fastwam observation.images.image 实测 512×512
# (server 端 ResizeImages→224, 但 512→224 vs 256→224 下采样不同 → 256 渲染会视觉域偏移致 SR 崩).
LIBERO_ENV_RESOLUTION = 512


def _openpi_client(host, port):
    """WebsocketClientPolicy(host,port).infer(obs)->{'actions':...}. openpi_client 已装 kai0/.venv;
    client env 若缺则 pip install openpi-client(见 eval yaml)。"""
    from openpi_client import websocket_client_policy
    return websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port))


_FLIP = os.environ.get("EVAL_IMG_FLIP", "both")  # both([::-1,::-1]) | vert([::-1]) | none


def _flip(a):
    if _FLIP == "both": return a[::-1, ::-1]
    if _FLIP == "vert": return a[::-1]
    return a


def _extract_obs(obs, task_description):
    """与 libero_eval_core 逐字段一致的 obs 提取 → openpi LiberoInputs 期望键。"""
    img = np.ascontiguousarray(_flip(obs["agentview_image"]))
    element = {"observation/image": img, "prompt": str(task_description)}
    if "robot0_eye_in_hand_image" in obs:
        element["observation/wrist_image"] = np.ascontiguousarray(_flip(obs["robot0_eye_in_hand_image"]))
    # ⭐ state 必须 8 维匹配训练(LIBERO_fastwam observation.state = eef_pos3 + axisangle3 + gripper_qpos2).
    # (libero_eval_core 的 starVLA 版用 gripper[-1:]=7维, openpi pi05 是 8维 → 必须用完整 gripper_qpos 2维)
    element["observation/state"] = np.concatenate((
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
    )).astype(np.float32)
    return element


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8000")
    ap.add_argument("--task-suite-name", default="libero_10")
    ap.add_argument("--num-trials-per-task", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replan-steps", type=int, default=5, help="每 N 步 replan 一次(开环执行 chunk 前 N 个)")
    ap.add_argument("--num-steps-wait", type=int, default=10, help="episode 起始沉降步(dummy action)")
    ap.add_argument("--num-tasks", type=int, default=-1, help="-1=全部任务")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    np.random.seed(args.seed)
    adapter = get_benchmark_adapter("libero")  # env adapter: build_env / get_max_steps
    # task_suite 走标准 LIBERO benchmark API(非 adapter)
    from libero.libero import benchmark as _bm
    task_suite = _bm.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = task_suite.n_tasks if args.num_tasks < 0 else min(args.num_tasks, task_suite.n_tasks)
    max_steps = adapter.get_max_steps(args.task_suite_name)
    client = _openpi_client(args.host, args.port)
    logging.info(f"[eval] suite={args.task_suite_name} tasks={n_tasks} trials={args.num_trials_per_task} "
                 f"seed={args.seed} replan={args.replan_steps} max_steps={max_steps} server={args.host}:{args.port}")

    out_dir = os.path.join(args.output_dir, "suites", args.task_suite_name)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "episodes.jsonl")
    jf = open(jsonl_path, "w")

    per_task = {}
    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        succ = 0
        for trial in range(args.num_trials_per_task):
            env, task_description = adapter.build_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            try:
                env.reset()
                obs = env.set_init_state(init_states[trial % len(init_states)])
                action_chunk, done, t = None, False, 0
                for step in range(max_steps + args.num_steps_wait):
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION); t += 1
                        continue
                    if action_chunk is None or (t - args.num_steps_wait) % args.replan_steps == 0:
                        _el = _extract_obs(obs, task_description)
                        action_chunk = np.asarray(client.infer(_el)["actions"])
                        if task_id == 0 and trial == 0 and (t - args.num_steps_wait) < 15:
                            logging.info(f"[dbg] FLIP={_FLIP} GRIPPER={os.environ.get('EVAL_GRIPPER','pm1')} prompt={_el['prompt']!r}")
                            logging.info(f"[dbg] t={t} state={np.round(_el['observation/state'],3)} "
                                         f"img={_el['observation/image'].shape}{_el['observation/image'].dtype}"
                                         f" range[{_el['observation/image'].min()},{_el['observation/image'].max()}]"
                                         f" act_chunk={action_chunk.shape} act0={np.round(action_chunk[0],3)}")
                    action = action_chunk[(t - args.num_steps_wait) % args.replan_steps][:7].copy()
                    # ⭐ gripper 值域转换. 实测训练数据 action gripper=binary{0,1} 且 **1=开, 0=闭**
                    # (ep188: action==1↔qpos 0.037开, action==0↔qpos 0.026闭). env(robosuite)实测 +1=闭/-1=开.
                    # 故正确映射 = inv_pm1: 1(开)→env -1(开), 0(闭)→env +1(闭) = 1-2*g. (pm1 是反的, 会致夹爪起步即闭合空抓)
                    #   inv_pm1: 1-2*g  [默认, 正确]   pm1: 2*g-1 [反, 已证错]   none: 原样
                    _gm = os.environ.get("EVAL_GRIPPER", "inv_pm1")
                    if _gm == "inv_pm1":
                        action[6] = 1.0 - 2.0 * float(action[6] > 0.5)
                    elif _gm == "pm1":
                        action[6] = 2.0 * float(action[6] > 0.5) - 1.0
                    # 每 40 步打一次进度(定位 abort 位置 + 确认机器人在动 = gripper 生效)
                    if task_id == 0 and trial == 0 and (t - args.num_steps_wait) % 40 == 0:
                        logging.info(f"[prog] t={t}/{max_steps} eef={np.round(obs['robot0_eef_pos'],3)} "
                                     f"grip_raw={action_chunk[(t-args.num_steps_wait)%args.replan_steps][6]:.2f}"
                                     f"->env={action[6]:.1f} done={done}")
                    obs, _, done, _ = env.step(action.tolist())
                    t += 1
                    if done:
                        break
            finally:
                try: env.close()
                except Exception: pass
            jf.write(json.dumps({"task_id": int(task_id), "trial": int(trial), "success": bool(done)}) + "\n")
            jf.flush()
            succ += int(bool(done))
            logging.info(f"  [trial] task={task_id} trial={trial} success={bool(done)} steps={t}")
        per_task[task_id] = succ / args.num_trials_per_task
        logging.info(f"  task {task_id}: SR={per_task[task_id]:.3f} ({succ}/{args.num_trials_per_task})")

    jf.close()
    agg = float(np.mean(list(per_task.values()))) if per_task else 0.0
    summary = {"task_suite_name": args.task_suite_name, "seed": args.seed,
               "num_trials_per_task": args.num_trials_per_task, "per_task_SR": per_task,
               "aggregate_SR": agg, "t6": per_task.get(6)}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    logging.info(f"[done] agg_SR={agg:.4f} t6={per_task.get(6)} → {jsonl_path}")


if __name__ == "__main__":
    main()
