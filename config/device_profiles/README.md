# Per-device profiles

At startup KAI0 selects `config/device_profiles/<hostname -s>.yml`.
Override it with `KAI0_DEVICE_PROFILE=/absolute/path/profile.yml`.

Each physical station should have a distinct `dataset_chunk` and its own
immutable CAN/camera IDs.  Example:

```yaml
machine_id: robot02
dataset_chunk: 2
can_left_slave: "USB_CAN_SERIAL_L"
can_right_slave: "USB_CAN_SERIAL_R"
camera_top_head_serial: "REALSENSE_SERIAL_TOP"
camera_hand_left_serial: "REALSENSE_SERIAL_LEFT"
camera_hand_right_serial: "REALSENSE_SERIAL_RIGHT"
camera_mid_head_enabled: 0
camera_mid_head_type: uvc
camera_mid_head_device: /dev/cam_mid_head
```

The profile is intentionally flat so both Bash hardware setup scripts and
Python ROS launch code can consume it without another configuration service.
Legacy `config/dongle_serials.yml` and the current camera defaults remain as
fallbacks when no host profile exists.
