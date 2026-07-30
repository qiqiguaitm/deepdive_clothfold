#!/usr/bin/env python3
"""跨相机重投影可视化 (my_calib session).

从 left / right 各挑一帧, 经"世界系外参链"把它们看到的 board 重投影到 head 相机:
  1. 稠密: board 是已知平面 → 用 plane-induced homography 把该臂 RGB 的 board 区域
     warp 到 head 图 (通过标定链 board→world→head_cam, 不用 depth), 叠加显示。
  2. 稀疏: 该臂检测的 charuco 角点 3D→world→head_cam→像素, 与 head 自检角点(绿)对比。

红=左臂投来, 蓝=右臂投来, 绿=head 自己检测(真值)。红/蓝落在绿上 = 跨相机外参一致。
"""
import glob
import os
import sys

import cv2
import numpy as np

CAL = '/data1/tim/workspace/deepdive_kai0/calib'
sys.path.insert(0, CAL)
import verify_projection as vp  # noqa: E402
from board_def import BoardSpec, get_board  # noqa: E402

SESSION = os.path.join(CAL, 'data/my_calib')
OUT = os.path.join(CAL, 'verify_out', 'reproject_my_calib.png')

spec = BoardSpec.from_yaml(os.path.join(CAL, 'board_9x14.yaml'))
board = get_board(spec)
calib = vp.load_calibration(os.path.join(SESSION, 'calibration.yaml'))
T_world_camF = calib['transforms']['T_world_camF']
T_camF_world = np.linalg.inv(T_world_camF)

# ── head 帧 ─────────────────────────────────────────────────────────────
head = vp.load_frame(os.path.join(SESSION, 'head.npz'), 'head', None)
K_h, dist_h = head.K, head.dist


def pick_best_pose(arm):
    """挑角点最多的一帧 (检测最稳)。"""
    best, best_n = None, -1
    for p in sorted(glob.glob(os.path.join(SESSION, arm, 'pose_*.npz'))):
        d = np.load(p, allow_pickle=True)
        n = len(d['charuco_ids'])
        if n > best_n:
            best, best_n = p, n
    return best, best_n


def board_plane_M(K, T_cam_board):
    """平面单应矩阵 M: [X,Y,1]_board(米) -> 像素 (Z=0 平面)。 M = K [r1 r2 t]."""
    R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


# board 四个外角 (米), 用于限定 warp 区域
Wm = spec.cols * spec.square_mm / 1000.0
Hm = spec.rows * spec.square_mm / 1000.0
quad_board = np.array([[0, 0], [Wm, 0], [Wm, Hm], [0, Hm]], np.float64)


def reproject_arm(arm, color):
    """返回 (warped_overlay_on_head, arm_corner_px_in_head, arm_label)。"""
    path, ncorner = pick_best_pose(arm)
    rel = os.path.join(arm, os.path.basename(path))
    fr = vp.load_frame(path, rel, arm)
    Tb, Tc = vp._arm_extrinsics(calib, arm)      # T_world_base, T_link6_cam
    T_world_cam = Tb @ fr.T_base_ee @ Tc
    T_world_board = T_world_cam @ fr.T_cam_board
    T_camF_board_pred = T_camF_world @ T_world_board   # board pose in head, 由外参预测

    d = np.load(path, allow_pickle=True)
    arm_rgb = cv2.cvtColor(d['rgb_image'], cv2.COLOR_BGR2RGB)  # 存的是 BGR
    K_a = d['camera_matrix']
    dist_a = d['dist_coeffs']

    # 去畸变到各自 pinhole (K 不变), 再用无畸变单应对齐
    arm_u = cv2.undistort(arm_rgb, K_a, dist_a, None, K_a)

    M_a = board_plane_M(K_a, fr.T_cam_board)          # board -> arm 像素
    M_h = board_plane_M(K_h, T_camF_board_pred)       # board -> head 像素(预测)
    src_quad = cv2.perspectiveTransform(quad_board[None], M_a)[0].astype(np.float32)
    dst_quad = cv2.perspectiveTransform(quad_board[None], M_h)[0].astype(np.float32)
    Hwarp = cv2.getPerspectiveTransform(src_quad, dst_quad)

    hh, hw = head.rgb.shape[:2]
    warped = cv2.warpPerspective(arm_u, Hwarp, (hw, hh))
    mask = np.zeros((hh, hw), np.uint8)
    cv2.fillConvexPoly(mask, dst_quad.astype(np.int32), 255)

    # 稀疏角点: 该臂检测角点 3D(board)->world->head_cam->像素
    P = vp.board_corners_3d(board, fr.ids)
    Pw = (T_world_board @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    Pc = (T_camF_world @ np.c_[Pw, np.ones(len(Pw))].T).T[:, :3]
    front = Pc[:, 2] > 0
    px, _ = cv2.projectPoints(Pc[front], np.zeros(3), np.zeros(3), K_h, dist_h)
    px = px.reshape(-1, 2)

    return warped, mask, px, dst_quad, os.path.basename(path), ncorner


def corner_px_error(px, ids_arm, fr_head):
    """与 head 自检角点按 id 匹配, 返回中位/均值像素误差。"""
    hmap = {int(i): c for i, c in zip(fr_head_ids, head.corners_2d)}
    errs = []
    for i, p in zip(ids_arm, px):
        if int(i) in hmap:
            errs.append(np.linalg.norm(p - hmap[int(i)]))
    return (np.median(errs), np.mean(errs), len(errs)) if errs else (np.nan, np.nan, 0)


# head 自检角点 ids
fr_head_ids = np.load(os.path.join(SESSION, 'head.npz'), allow_pickle=True)['charuco_ids'].reshape(-1)

wL, mL, pxL, quadL, nameL, ncL = reproject_arm('left', (255, 0, 0))
wR, mR, pxR, quadR, nameR, ncR = reproject_arm('right', (0, 0, 255))

# arm 角点 ids (与 px 对齐顺序一致, load_frame 里 ids 顺序即 corners 顺序)
frL = vp.load_frame(os.path.join(SESSION, 'left', nameL), 'L', 'left')
frR = vp.load_frame(os.path.join(SESSION, 'right', nameR), 'R', 'right')
eL = corner_px_error(pxL, frL.ids, head)
eR = corner_px_error(pxR, frR.ids, head)

head_rgb = cv2.cvtColor(head.rgb, cv2.COLOR_BGR2RGB)


def blend(base, warped, mask, alpha=0.55):
    out = base.copy()
    m = mask.astype(bool)
    out[m] = (alpha * warped[m] + (1 - alpha) * base[m]).astype(np.uint8)
    return out


def draw_pts(img, px, color, cross=False, size=4):
    for u, v in px.astype(int):
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            if cross:
                cv2.drawMarker(img, (u, v), color, cv2.MARKER_TILTED_CROSS, 9, 1)
            else:
                cv2.circle(img, (u, v), size, color, -1)


# ── 面板 1: head + 左臂稠密 warp + 角点 ────────────────────────────────
p1 = blend(head_rgb, wL, mL)
cv2.polylines(p1, [quadL.astype(np.int32)], True, (255, 0, 0), 2)
draw_pts(p1, head.corners_2d, (0, 255, 0), cross=True)
draw_pts(p1, pxL, (255, 0, 0))

# ── 面板 2: head + 右臂稠密 warp + 角点 ────────────────────────────────
p2 = blend(head_rgb, wR, mR)
cv2.polylines(p2, [quadR.astype(np.int32)], True, (0, 0, 255), 2)
draw_pts(p2, head.corners_2d, (0, 255, 0), cross=True)
draw_pts(p2, pxR, (0, 0, 255))

# ── 面板 3: 仅角点, 三色对齐验证 ──────────────────────────────────────
p3 = head_rgb.copy()
draw_pts(p3, head.corners_2d, (0, 255, 0), cross=True)
draw_pts(p3, pxL, (255, 0, 0), size=3)
draw_pts(p3, pxR, (0, 0, 255), size=3)


def label(img, lines):
    y = 22
    for t in lines:
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        y += 24


label(p1, [f'LEFT {nameL} -> head  (dense board warp)',
           f'corner err: med={eL[0]:.1f}px mean={eL[1]:.1f}px  n={eL[2]}'])
label(p2, [f'RIGHT {nameR} -> head  (dense board warp)',
           f'corner err: med={eR[0]:.1f}px mean={eR[1]:.1f}px  n={eR[2]}'])
label(p3, ['corners only: green=head(GT) red=left blue=right',
           f'L med={eL[0]:.1f}px  R med={eR[0]:.1f}px'])

grid = np.hstack([p1, np.full((p1.shape[0], 6, 3), 255, np.uint8), p2,
                  np.full((p1.shape[0], 6, 3), 255, np.uint8), p3])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
cv2.imwrite(OUT, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print(f'left  pose={nameL} ncorners={ncL}  reproj err med={eL[0]:.2f}px mean={eL[1]:.2f}px (n={eL[2]})')
print(f'right pose={nameR} ncorners={ncR}  reproj err med={eR[0]:.2f}px mean={eR[1]:.2f}px (n={eR[2]})')
print(f'saved -> {OUT}')
