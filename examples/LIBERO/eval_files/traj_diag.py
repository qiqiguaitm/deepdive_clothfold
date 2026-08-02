#!/usr/bin/env python
"""单 trial 逐步轨迹日志: state/action/eef, 看机器人怎么动 + state 是否跳变。"""
import os, sys, numpy as np
sys.path.insert(0, os.path.abspath("kai0/src"))
os.environ.setdefault('LIBERO_CONFIG_PATH', os.environ['LIBERO_HOME']+'/libero')
from openpi.training import config as _config
from openpi.policies import policy_config as _pc
from examples.LIBERO.eval_files.libero_benchmark_adapters import get_benchmark_adapter, quat2axisangle
from libero.libero import benchmark as bm

CKPT=os.environ['CKPT']
policy=_pc.create_trained_policy(_config.get_config("pi05_libero_a0"), CKPT)
print("[traj] policy loaded")
adapter=get_benchmark_adapter('libero'); ts=bm.get_benchmark_dict()['libero_10']()
task=ts.get_task(0); init=ts.get_task_init_states(0)
prompt=task.language
def flip(a): return a[::-1,::-1]
def build_obs(obs):
    e={"observation/image":np.ascontiguousarray(flip(obs['agentview_image'])),
       "observation/wrist_image":np.ascontiguousarray(flip(obs['robot0_eye_in_hand_image'])),
       "observation/state":np.concatenate((obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),
           np.asarray(obs['robot0_gripper_qpos'],dtype=np.float32).reshape(-1))).astype(np.float32),
       "prompt":prompt}
    return e
env,desc=adapter.build_env(task,512,0); env.reset(); obs=env.set_init_state(init[0])
for _ in range(10): obs,_,_,_=env.step([0.]*6+[-1.])
print(f"[traj] task={desc}")
chunk=None; done=False; t=0
for t in range(200):
    if t%5==0:
        el=build_obs(obs)
        chunk=np.asarray(policy.infer(el)["actions"])
        if t%20==0:
            st=el['observation/state']
            print(f"t={t:3d} eef={obs['robot0_eef_pos'].round(3)} rot={st[3:6].round(3)} grip={obs['robot0_gripper_qpos'].round(3)} | act0={chunk[0].round(3)} grip_seq={chunk[:,6].round(1)}")
    a=chunk[t%5][:7].copy(); a[6]=1.0-2.0*(a[6]>0.5)  # inv_pm1: 训练 1=开→env -1, 0=闭→env +1
    obs,_,done,_=env.step(a.tolist())
    if done: print(f"[traj] SUCCESS at t={t}"); break
print(f"[traj] done={done} final_eef={obs['robot0_eef_pos'].round(3)}")
env.close()
