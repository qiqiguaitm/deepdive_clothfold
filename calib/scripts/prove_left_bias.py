#!/usr/bin/env python3
"""证明左臂关节零位偏差 (train/test, 模型对比)。

不变量: 板固定+base固定 => p(i)=inv(FK(q_i+δq)·T_link6_cam·T_cam_board(i))[:3,3] 恒定。
两个模型都在 15 train 帧上联合拟合手眼外参 T_link6_cam:
  Null : δq≡0, 只拟合外参(6 dof)           —— "无零位偏差" 假设
  Bias : 同时拟合 δq(6) + 外参(6)=12 dof    —— "有零位偏差" 假设
在留出的 5 test 帧上评估 p 相对 train 质心的散布 (test 帧不参与任何拟合)。
判据: 若 Bias 使 held-out 散布大幅下降, 而 Null 不能 -> 存在真实且可泛化的关节零位偏差。
右臂作对照: δq 应≈0 且两模型 held-out 接近 -> 证明是左臂特有, 非过拟合/巧合。
"""
import glob
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

CAL = '/data1/tim/workspace/deepdive_kai0/calib'
sys.path.insert(0, CAL)
import verify_projection as vp  # noqa: E402
from piper_fk import PiperFK  # noqa: E402

S = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CAL, 'data/my_calib')
CALFILE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(S, 'calibration.yaml')
fk = PiperFK()
calib = vp.load_calibration(CALFILE)
TEST_IDX = [0, 4, 8, 12, 16]        # 固定留出 5 帧 (均匀分布)


def se3(p):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(p[:3])[0]
    T[:3, 3] = p[3:6]
    return T


def load_arm(arm):
    Tlc0 = calib['transforms']['T_link6_camL' if arm == 'left' else 'T_link6_camR']
    fs = sorted(glob.glob(os.path.join(S, arm, 'pose_*.npz')))
    out = []
    for p in fs:
        fr = vp.load_frame(p, os.path.basename(p), arm)
        q = np.asarray(np.load(p, allow_pickle=True)['joint_angles'])
        out.append((fr, q))
    return Tlc0, out


def p_base_in_board(fr, q, dq, Tlc):
    return np.linalg.inv(fk.fk_homogeneous(q + dq) @ Tlc @ fr.T_cam_board)[:3, 3]


def fit(train, Tlc0, with_bias):
    r0 = cv2.Rodrigues(Tlc0[:3, :3])[0].ravel()
    x0 = np.concatenate([r0, Tlc0[:3, 3], np.zeros(6)])

    def unpack(x):
        Tlc = se3(x[:6])
        dq = x[6:12] if with_bias else np.zeros(6)
        return Tlc, dq

    def resid(x):
        Tlc, dq = unpack(x)
        ps = np.array([p_base_in_board(fr, q, dq, Tlc) for fr, q in train])
        return (ps - ps.mean(0)).ravel()

    lo = np.full(12, -np.inf); hi = np.full(12, np.inf)
    if with_bias:
        lo[6:] = -np.radians(5); hi[6:] = np.radians(5)
    else:
        lo[6:] = -1e-9; hi[6:] = 1e-9      # freeze δq≈0
    r = least_squares(resid, x0, bounds=(lo, hi))
    return unpack(r.x)


def scat(subset, dq, Tlc, centroid):
    ps = np.array([p_base_in_board(fr, q, dq, Tlc) for fr, q in subset])
    return np.linalg.norm((ps - centroid) * 1000, axis=1)


def run(arm):
    Tlc0, frames = load_arm(arm)
    n = len(frames)
    test = [frames[i] for i in TEST_IDX if i < n]
    train = [frames[i] for i in range(n) if i not in TEST_IDX]

    print(f'\n===== {arm.upper()} arm  (train={len(train)}, held-out test={len(test)}) =====')
    res = {}
    for name, wb in [('Null(δq=0)', False), ('Bias(δq≠0)', True)]:
        Tlc, dq = fit(train, Tlc0, wb)
        c = np.mean([p_base_in_board(fr, q, dq, Tlc) for fr, q in train], axis=0)
        dtr, dte = scat(train, dq, Tlc, c), scat(test, dq, Tlc, c)
        res[name] = dte
        tag = f'  δq(deg)={np.round(np.degrees(dq), 2).tolist()}' if wb else ''
        print(f'  {name:11s}: TRAIN散布 mean={dtr.mean():5.2f}mm max={dtr.max():5.2f}'
              f'   |  HELD-OUT mean={dte.mean():5.2f}mm max={dte.max():5.2f}{tag}')
    imp = (1 - res['Bias(δq≠0)'].mean() / res['Null(δq=0)'].mean()) * 100
    print(f'  --> 加入关节零位后 held-out 散布下降 {imp:.0f}%'
          f'  (per-pose held-out mm: Null {np.round(res["Null(δq=0)"],1).tolist()}'
          f' -> Bias {np.round(res["Bias(δq≠0)"],1).tolist()})')


run('left')
run('right')
