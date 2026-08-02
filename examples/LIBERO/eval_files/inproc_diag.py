#!/usr/bin/env python
"""in-process 闭环诊断: 直接 policy.infer(不走 websocket), 跑 task0 少量 trial + 详细日志.
对比: 同时在训练帧上推理, 看 live-obs 动作 vs 训练帧动作。"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath("kai0/src"))
os.environ.setdefault('LIBERO_CONFIG_PATH', os.environ['LIBERO_HOME']+'/libero')
import av
from openpi.training import config as _config
from openpi.policies import policy_config as _pc
from examples.LIBERO.eval_files.libero_benchmark_adapters import get_benchmark_adapter, quat2axisangle
from libero.libero import benchmark as bm

CKPT=os.environ['CKPT']; D=os.environ['D']
policy=_pc.create_trained_policy(_config.get_config("pi05_libero_a0"), CKPT)
print("[diag] policy loaded")

# 训练帧参照 (ep188 = task5 = benchmark task0)
df=pd.read_parquet(D+'/data/chunk-000/episode_000188.parquet')
base=[f.to_ndarray(format='rgb24') for f in av.open(D+'/videos/chunk-000/observation.images.image/episode_000188.mp4').decode(video=0)]
wrist=[f.to_ndarray(format='rgb24') for f in av.open(D+'/videos/chunk-000/observation.images.wrist_image/episode_000188.mp4').decode(video=0)]
tstate=np.stack(df['observation.state'].values).astype(np.float32)
tact=np.stack(df['action'].values).astype(np.float32)
prompt='put both the alphabet soup and the tomato sauce in the basket'
o_train={"observation/image":np.ascontiguousarray(base[0]),"observation/wrist_image":np.ascontiguousarray(wrist[0]),"observation/state":tstate[0],"prompt":prompt}
a_train=np.asarray(policy.infer(o_train)["actions"])[0]
print(f"[diag] 训练帧0: pred={a_train.round(3)} recorded={tact[0].round(3)}")

adapter=get_benchmark_adapter('libero'); ts=bm.get_benchmark_dict()['libero_10']()
task=ts.get_task(0); init=ts.get_task_init_states(0)
FLIP=os.environ.get('EVAL_IMG_FLIP','both'); GM=os.environ.get('EVAL_GRIPPER','pm1')
def flip(a): return a[::-1,::-1] if FLIP=='both' else (a[::-1] if FLIP=='vert' else a)
def build_obs(obs):
    e={"observation/image":np.ascontiguousarray(flip(obs['agentview_image'])),"prompt":prompt}
    if 'robot0_eye_in_hand_image' in obs: e["observation/wrist_image"]=np.ascontiguousarray(flip(obs['robot0_eye_in_hand_image']))
    e["observation/state"]=np.concatenate((obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),np.asarray(obs['robot0_gripper_qpos'],dtype=np.float32).reshape(-1))).astype(np.float32)
    return e

nsucc=0
for trial in range(3):
    env,desc=adapter.build_env(task,512,0); env.reset(); obs=env.set_init_state(init[trial])
    for _ in range(10): obs,_,_,_=env.step([0.]*6+[-1.])
    el=build_obs(obs)
    if trial==0:
        a_live=np.asarray(policy.infer(el)["actions"])[0]
        print(f"[diag] live init[0] state={el['observation/state'].round(3)}")
        print(f"[diag] live init[0]: pred={a_live.round(3)}  (训练帧0 pred={a_train.round(3)})")
        wr=el.get('observation/wrist_image',np.array([0]))
        print(f"[diag] live img range[{el['observation/image'].min()},{el['observation/image'].max()}] wrist range[{wr.min()},{wr.max()}] wrist_shape={wr.shape}")
    chunk=None; done=False; t=0; maxs=520
    while t<maxs:
        if chunk is None or t%5==0:
            chunk=np.asarray(policy.infer(build_obs(obs))["actions"])
        a=chunk[t%5][:7].copy()
        if GM=='pm1': a[6]=2*(a[6]>0.5)-1
        obs,_,done,_=env.step(a.tolist()); t+=1
        if done: break
    print(f"[diag] trial{trial}: done={done} steps={t} final_eef={obs['robot0_eef_pos'].round(3)}")
    nsucc+=int(done); env.close()
print(f"[diag] task0 3trial SR={nsucc}/3")
