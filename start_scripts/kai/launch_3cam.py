"""
Launch 4 RealSense cameras via ROS2 realsense2_camera nodes.

  D435  (top)   → namespace: camera_f  | RGB 640x480   + Depth 640x480  @ 15fps
  D435I (mid)   → namespace: camera_m  | RGB 640x480   (+Depth gated by macro, off)
  D405-A (left) → namespace: camera_l  | RGB 640x480   (+Depth gated by macro)
  D405-B (right)→ namespace: camera_r  | RGB 640x480   (+Depth gated by macro)

  (Filename kept as launch_3cam.py for the run.sh/CLAUDE.md reference; it now
   brings up 4 nodes — camera_m is the second head cam added 2026-07-08.)

  Per-camera depth on/off comes from config/camera_depth_flags.py
  (ENABLE_DEPTH_TOP_HEAD / _HAND_LEFT / _HAND_RIGHT). Wrist depth is
  currently OFF; flip the macro to bring it back.

  FPS: 默认 30 (与训练数据 30 Hz 对齐). 历史上 launch_3cam 默认 15 是为了
  缓解 "3 相机 + 3 路 depth 同走 USB 3 hub" 的带宽压力 — 当时 30 fps 会触发
  hand_left "Incomplete video frame" 退化到 1-10 Hz. 现在 D405 wrist depth
  通过 camera_depth_flags 宏关掉, 实际只剩 1 路 depth (D435) + 3 路 color,
  带宽约掉了一半, 30 fps 跑得动. 若仍丢帧, 用 CAM_FPS=15 ros2 launch ... 临时降回去.

Usage:
  ros2 launch scripts/launch_3cam.py
"""
import importlib.util
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

_DEFAULT_FPS = int(os.environ.get('CAM_FPS', '30'))

# mid_head 相机来源: '1' (默认) = WHEELTEC C100 UVC 摄像头 (uvc_camera_node.py,
# 走稳定 /dev/cam_mid_head); '0' = 回退原 D435I RealSense。发布 topic 一致, 下游无感。
_MID_HEAD_UVC = os.environ.get('MID_HEAD_UVC', '1') == '1'
# start_data_collect.sh auto-detects this.  It may also be set explicitly for
# direct ros2 launch use.  Disabled means no camera_m process is spawned.
_ENABLE_MID_HEAD = os.environ.get('KAI0_ENABLE_MID_HEAD', '1') == '1'
_MID_HEAD_DEVICE = os.environ.get('KAI0_CAMERA_MID_HEAD_DEVICE', '/dev/cam_mid_head')
_SERIAL_TOP = os.environ.get('KAI0_CAMERA_TOP_HEAD_SERIAL', '254622070889')
_SERIAL_LEFT = os.environ.get('KAI0_CAMERA_HAND_LEFT_SERIAL', '409122273074')
_SERIAL_RIGHT = os.environ.get('KAI0_CAMERA_HAND_RIGHT_SERIAL', '409122271568')
_SERIAL_MID = os.environ.get('KAI0_CAMERA_MID_HEAD_SERIAL', '254522074228')


def make_uvc_mid_head_node(width=640, height=480, fps=_DEFAULT_FPS):
    """把 WHEELTEC UVC 相机作为 mid_head 发布到 /camera_m/color/image_raw,
    drop-in 替换 D435I。device 默认稳定符号链接 /dev/cam_mid_head。"""
    script = str(Path(__file__).resolve().parent / 'uvc_camera_node.py')
    return ExecuteProcess(
        cmd=['python3', script, '--ros-args',
             '-p', f'device:={_MID_HEAD_DEVICE}',
             '-p', 'ns:=/camera_m',
             '-p', f'width:={width}', '-p', f'height:={height}', '-p', f'fps:={fps}'],
        output='screen',
    )


def _load_depth_enabled_map() -> dict:
    """Probe upward for config/camera_depth_flags.py."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / 'config' / 'camera_depth_flags.py'
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                'kai0_camera_depth_flags_3cam', candidate)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return dict(mod.CAMERA_DEPTH_ENABLED)
    return {}


_DEPTH_ENABLED_MAP = _load_depth_enabled_map()


def make_camera_node(name, namespace, serial,
                     rgb_w, rgb_h, depth_w, depth_h, fps=_DEFAULT_FPS,
                     is_d405=False, enable_depth=True):
    params = {
        'serial_no': serial,
        'camera_name': name,
        'enable_color': True,
        'enable_depth': enable_depth,
        'enable_infra1': False,
        'enable_infra2': False,
        'enable_gyro': False,
        'enable_accel': False,
        # D435 has dedicated RGB module → 用 rgb_camera.color_profile
        # D405 把 color 共享在 stereo (depth) 模块 → 用 depth_module.color_profile
        # 同时设两个: 不适用的会被驱动忽略 (我们之前只设 rgb_camera.color_profile,
        # 结果 D405 的 color 没人管, 默认跑成 848x480x30)
        'rgb_camera.color_profile': f'{rgb_w}x{rgb_h}x{fps}',
        'depth_module.color_profile': f'{rgb_w}x{rgb_h}x{fps}',
        # 抗闪烁:
        #   D435 rolling-shutter RGB → power_line_frequency=1 (50Hz) 修横纹
        #   D405 global-shutter color → PLF 无效, 必须锁定曝光到覆盖 LED PWM
        #     周期的值 (20ms 经实测在 sim01 工位下闪烁消失且亮度足够).
        # 1=50Hz, 2=60Hz, 3=auto. 两个模块都设以覆盖 D435/D405 不同挂载.
        'rgb_camera.power_line_frequency': 1,
        'depth_module.power_line_frequency': 1,
        'align_depth.enable': False,
    }
    if enable_depth:
        params['depth_module.depth_profile'] = f'{depth_w}x{depth_h}x{fps}'
    if is_d405:
        params['depth_module.enable_auto_exposure'] = False
        params['depth_module.exposure'] = 20000  # μs
    return Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name=name,
        namespace=namespace,
        output='screen',
        parameters=[params],
    )


def generate_launch_description():
    cam_f = make_camera_node(
        name='camera_f', namespace='',
        serial=_SERIAL_TOP,
        rgb_w=640, rgb_h=480, depth_w=640, depth_h=480,
        enable_depth=_DEPTH_ENABLED_MAP.get('top_head', False),
    )
    cam_m = None
    if _ENABLE_MID_HEAD and _MID_HEAD_UVC:
        # WHEELTEC C100 UVC 摄像头替换 D435I 作为 mid_head (2026-07-10)
        cam_m = make_uvc_mid_head_node(width=640, height=480)
    elif _ENABLE_MID_HEAD:
        cam_m = make_camera_node(
            name='camera_m', namespace='',
            serial=_SERIAL_MID,
            rgb_w=640, rgb_h=480, depth_w=640, depth_h=480,
            # D435I: dedicated RGB module like D435 (is_d405=False → color on image_raw)
            enable_depth=_DEPTH_ENABLED_MAP.get('mid_head', False),
        )
    cam_l = make_camera_node(
        name='camera_l', namespace='',
        serial=_SERIAL_LEFT,
        rgb_w=640, rgb_h=480, depth_w=640, depth_h=480,
        is_d405=True,
        enable_depth=_DEPTH_ENABLED_MAP.get('hand_left', False),
    )
    cam_r = make_camera_node(
        name='camera_r', namespace='',
        serial=_SERIAL_RIGHT,
        rgb_w=640, rgb_h=480, depth_w=640, depth_h=480,
        is_d405=True,
        enable_depth=_DEPTH_ENABLED_MAP.get('hand_right', False),
    )
    nodes = [cam_f, cam_l, cam_r]
    if cam_m is not None:
        nodes.insert(1, cam_m)
    return LaunchDescription(nodes)
