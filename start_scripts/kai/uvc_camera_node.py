#!/usr/bin/env python3
"""
uvc_camera_node.py — 把一路 UVC 摄像头 (WHEELTEC C100/C70) 当作 mid_head 发布到 ROS2。

设计为 realsense2_camera 的 **drop-in 替换**: 发布的 topic / 消息类型 / 编码与
原 D435I mid_head 完全一致, 因此 ros_bridge / dataset_writer / 前端全部不用改,
录出来照样落 observation.images.mid_head/。

发布:
  <ns>/color/image_raw    sensor_msgs/Image      (rgb8, 640x480, 30Hz)
  <ns>/color/camera_info  sensor_msgs/CameraInfo (供 ros_bridge 显示 fps/latency)

关键采集设置 (踩过的坑):
  - MJPG        : 640x480 才能满 30fps (YUYV 高分辨率限速)
  - BUFFERSIZE=2: =1 会掉到一半帧率 (15fps)
  - 设备默认走稳定符号链接 /dev/cam_mid_head (见 config/99-wheeltec-cam.rules),
    免疫 USB 换口/换号; 找不到再回退 by-id, 再回退扫描。

用法 (一般由 launch_3cam.py 拉起, 也可单独跑):
  python3 uvc_camera_node.py --ros-args \
     -p device:=/dev/cam_mid_head -p ns:=/camera_m -p width:=640 -p height:=480 -p fps:=30
"""
import glob

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def resolve_device(arg: str):
    """device 参数: 具体路径/设备号直接用; 'auto' 则 稳定符号链接 -> by-id -> 扫描。"""
    if arg and arg != "auto":
        return int(arg) if str(arg).isdigit() else arg
    for cand in ("/dev/cam_mid_head",):
        if glob.glob(cand):
            return cand
    byid = sorted(glob.glob("/dev/v4l/by-id/usb-HJ_USB_2.0_Camera*-video-index0"))
    if byid:
        return byid[0]
    vids = sorted(glob.glob("/dev/video*"))
    return vids[0] if vids else 0


class UvcCameraNode(Node):
    def __init__(self):
        super().__init__("uvc_camera_node")
        p = self.declare_parameter
        device = p("device", "auto").value
        ns = p("ns", "/camera_m").value.rstrip("/")
        self.width = int(p("width", 640).value)
        self.height = int(p("height", 480).value)
        self.fps = int(p("fps", 30).value)
        self.frame_id = p("frame_id", "camera_m_color_optical_frame").value

        dev = resolve_device(device)
        self.get_logger().info(f"opening UVC device {dev} @ {self.width}x{self.height} {self.fps}fps")
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open UVC device {dev} (ls /dev/video* / 检查在位)")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)   # >=2 否则掉半帧率

        self.pub_img = self.create_publisher(Image, f"{ns}/color/image_raw", 10)
        self.pub_info = self.create_publisher(CameraInfo, f"{ns}/color/camera_info", 10)
        self._info = self._make_camera_info()
        # 预热丢弃上电头几帧
        for _ in range(10):
            self.cap.read()
        self.timer = self.create_timer(1.0 / self.fps, self._tick)
        self._warned = False

    def _make_camera_info(self) -> CameraInfo:
        ci = CameraInfo()
        ci.width = self.width
        ci.height = self.height
        ci.distortion_model = "plumb_bob"
        ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        fx = fy = float(self.width)          # 占位内参 (RGB 录制不依赖标定)
        cx, cy = self.width / 2.0, self.height / 2.0
        ci.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return ci

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if not self._warned:
                self.get_logger().warning("UVC read() 失败 (相机掉线?), 持续重试")
                self._warned = True
            return
        self._warned = False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # 与 realsense color/image_raw 同为 rgb8
        stamp = self.get_clock().now().to_msg()
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = rgb.shape[0], rgb.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = rgb.shape[1] * 3
        msg.data = np.ascontiguousarray(rgb).tobytes()
        self.pub_img.publish(msg)
        self._info.header.stamp = stamp
        self._info.header.frame_id = self.frame_id
        self.pub_info.publish(self._info)

    def destroy_node(self):
        try:
            self.cap.release()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = UvcCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
