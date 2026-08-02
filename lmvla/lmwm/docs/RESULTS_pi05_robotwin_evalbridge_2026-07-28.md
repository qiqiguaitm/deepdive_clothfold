# RESULTS: pi05(openpi) × RoboTwin eval 桥 — 通路验证 (2026-07-28)

**目标**: 让 openpi 训练的 `pi05_robotwin_a0/a1` 能在 RoboTwin sim 上评出 success rate。
**做法(最省改动)**: 复用 starVLA 的 `robotwin_batch_bridge.py` 整套 env 驱动(slot/Pipe/SR/summary.json),
只把 model client 从 starVLA `ModelClient` 换成 openpi `WebsocketClientPolicy` 适配器,server 命令从 starVLA
`server_policy` 换成 openpi `serve_policy.py`。**a0(无 hint)已本机协议打通**;a1(在线 hint)组件已写好并单测通过,
待 a0 真评 + a1 训练 ckpt。

---

## 1. 通路(已摸清 + 验证)

```
auto_eval_robotwin.sh
  → batched_eval_runner.py (master→worker)
      ├─ [server] ROBOTWIN_SERVER_BACKEND=openpi → serve_policy.py --policy.config <cfg> --policy.dir <ckpt目录>
      │            (kai0/.venv; server 端做 repack-free 推理: AlohaInputs→Normalize→pi05→AlohaOutputs→Unnormalize)
      └─ [bridge] robotwin_batch_bridge.py (RoboTwin wrapper env; sapien/mplib)
                    ROBOTWIN_MODEL_INTERFACE=openpi → model2robotwin_openpi.OpenpiRobotwinModelClient
                      self.client = openpi_client.websocket_client_policy.WebsocketClientPolicy(host,port)
                      step_batch: 逐 example 构 pi05 obs → client.infer(obs) → {"actions":[T,14]} → 直接喂 env
```

### openpi server obs 契约(关键,已确认)
- `create_trained_policy` **推理时 repack 用空 Group**(policy_config.py:46,84);config 里的 repack(`observation.images.cam_high`→…)
  **只用于训练数据加载, 推理不走**。故 client 必须直接发 `AlohaInputs` 期望的格式:
  ```python
  obs = {"images": {"cam_high": CHW_uint8, "cam_left_wrist": CHW_uint8, "cam_right_wrist": CHW_uint8},
         "state": raw_robotwin_joint[14], "prompt": instruction}   # a1 另加 "lmwm_hint":[768]
  ```
- 图像必须 **[3,H,W] uint8**(AlohaInputs `_decode_aloha` 内部 `rearrange('c h w -> h w c')`);RoboTwin 原生 HWC → 适配器转 CHW。
  分辨率任意,server 端 `ResizeImages`→224。
- state = **原始 RoboTwin joint 14 维**(`joint_action.vector`),不做任何客户端归一化;server 端 `AlohaInputs._decode_state(adapt_to_pi)`
  做 joint flip + gripper linear→angular(与训练同款),故 client 直发原始 state 即对齐训练。
- 返回 `{"actions":[8,14]}`:server 端 `AlohaOutputs(_encode_actions, adapt_to_pi)` + `Unnormalize` 已把动作转回 RoboTwin joint 空间。

### ⚠️ 归一化(核心正确性,已证)
- **openpi server 已在内部 Normalize + Unnormalize + AlohaOutputs**,返回的是 **真实 14 维 joint 动作**。
- openpi ModelClient **不做任何客户端 unnormalize**(与 starVLA `ModelClient` 的 `unnormalize_actions` 相反 —— starVLA server 发
  `normalized_actions`,client 再反归一化;openpi 必须**跳过**,否则双反=错)。
- **证据(本机 smoke)**: 返回动作落在合理 rad 区间(joints≈[-1.8,1.5]),gripper idx 6/13≈0.01~0.09(= aloha gripper 线性空间,
  `_gripper_from_angular` 输出域),**不是** [-1,1] 归一化域。若发生双反,数值会离谱越界。→ 单次(仅 server 端)反归一化确认正确。
- `env_action_type = "qpos"`, 14 维 joint, 与 starVLA joint 版一致。

### metadata bypass(已确认必要)
- openpi server metadata = 空 dict(`server_meta_keys=[]`),无 `ckpt_path` 字段。starVLA `ModelClient._validate_server_metadata`
  会因此炸;openpi 适配器**不做该校验**(仅打印)。

---

## 2. 本机协议 smoke(通过 ✓)

- **Server**: `serve_policy.py --policy.config pi05_robotwin_a0 --policy.dir <smoke_ckpt>`,
  smoke_ckpt = `pi05_base/params`(符号链)+ `assets/pi05_robotwin_a0/robotwin2.0/norm_stats.json`(符号链)。
  → 加载成功,监听 8111,norm stats 正确读取。
- **Client**: `OpenpiRobotwinModelClient` 连 server,喂**合成 RoboTwin observation**(head/left/right_camera rgb + joint_action.vector[14]),
  跑 2 trials × 20 步(交替 needs_query/step_cached 模拟 replan)。
- **结果**:
  - server 连通 ✓;`server_meta_keys=[]` → metadata bypass 生效 ✓
  - obs map 无 shape 错:example={lang, image×3 (240,320,3)uint8, state(14,)} ✓
  - step_batch 返回 `[14]` finite 动作,直接可喂 `env.take_action(action, action_type="qpos")` ✓
  - 归一化正确(见上,动作在真实 joint 域,非归一化域)✓
- **注意**: 这是 **协议级 smoke**,用 **pi05_base(未在 robotwin 上训)** + **合成 obs(无 robotwin sim 渲染)**,
  故**没有真实 SR 数字** —— 验证的是「wire 协议/obs 契约/动作形态/归一化」,不是任务成功率。真 SR 需 (a) 训练好的 a0 ckpt +
  (b) RoboTwin sim 渲染,均在 North-H20(见 §4)。

---

## 3. 产出文件

| 文件 | 作用 |
|---|---|
| `lmvla/lawam/examples/Robotwin/eval_files/model2robotwin_openpi.py` | **openpi ModelClient 适配器**(self-contained,内联 `_SlotState`+joint example,不 import starVLA);`get_model` 供 bridge 路由 |
| `lmvla/lawam/examples/Robotwin/eval_files/hint_online_robotwin.py` | **a1 在线 hint computer**(泛化自 libero `hint_online.py`;LMWM=`lmwm_robotwin_rvalley/lmwm.pt` din768;encoder=dinov3-base;cam_high 正立无 flip,resize 256) |
| `train_scripts/kai/volc/pi05_robotwin_eval_a0_x4_8h20.yaml` | **编排 yaml**(North-H20 8卡,a0×4 seed on GPU0-3;openpi server backend;自愈 wrapper openpi_client) |

**改动(env 门控,不设开关时 starVLA 路径逐字节不变)**:
- `robotwin_batch_bridge.py`: `ROBOTWIN_MODEL_INTERFACE=openpi` → import `model2robotwin_openpi.get_model`(默认 starvla)。
- `batched_eval_runner.py`: `ROBOTWIN_SERVER_BACKEND=openpi` → 起 `serve_policy.py`(而非 starVLA `server_policy`)、ckpt 允许为**目录**(而非 .pt 文件)。

---

## 4. a0 真评 / a1 状态

- **a0 真评**: 未跑。需 (1) a0 训练 ckpt(North-H20 `snshh`,即 `checkpoints/pi05_robotwin_a0_bj/pi05_robotwin_a0/<step>/`,
  含 params/+assets/);(2) RoboTwin sim(North-H20 huanqian wrapper)。yaml 已就绪,把 `CKPT` 指向训练输出目录后
  经 gsy 提交即可(6 积木×50×4seed)。
- **a1(在线 hint)**: hint computer 单测通过(DINOv3-base+LMWM robotwin ckpt → hint[768] finite, norm≈7.1)。
  a1 eval = a0 yaml + `ROBOTWIN_OPENPI_CONFIG=pi05_robotwin_a1_prefix_bj` + `ROBOTWIN_HINT_ENCODER=dinov3-base`(触发 client 逐帧
  对 head_camera 算 hint 塞入 `obs["lmwm_hint"]`)+ ckpt 换 a1 步目录。
  - **a1 serving 已确认可行**: a1 config 的离线 `HintLookupTransform` 虽在 `create()` 里 eager 加载 npz,但推理 repack 被 `create_trained_policy`
    绕过 → 离线 lookup 不参与推理;在线 hint 由 client 注入,`AlohaInputs` 透传(aloha_policy.py:88)→ model(lmwm_hint_dim=768)。
    只要 North-E hint npz 存在(a1 正在其上训练,故存在)serve 不炸。
  - a1 待 #39 a1 ckpt 训完(grid 98%)。

---

## 5. 卡点 / 风险(诚实报)

1. **无真 SR(本机)**: 本机缺 (a) robotwin 训练 ckpt、(b) 确认可用的 RoboTwin sim 渲染,故只做了协议 smoke。真 SR 必须在 North-H20 跑 yaml。
   —— 这是设计预期(任务说「先本机 smoke 通协议」),非阻塞。
2. **a1 bridge 重依赖**: 在线 hint 让 bridge(robotwin wrapper env)额外加载 torch+crave+DINOv3+LMWM,与 openpi server 同卡。
   H20 96GB 够,但需 wrapper env 能 import crave.encoders + DINOv3 权重可达(本机 kai0/.venv 已验可 import;North-H20 wrapper 待验)。
3. **replan=8**: pi05 `action_horizon=8`,故 `ROBOTWIN_REPLAN_STEPS≤8`(yaml 设 8,执行整段 chunk 后 requery);
   starVLA 版用 36(其 horizon 更长)—— 不可照搬。
4. **openpi_client 进 wrapper env**: bridge 跑在 huanqian conda,需 `openpi_client`(+websockets/msgpack)。yaml 已 (a) 加 openpi-client src 到
   PYTHONPATH、(b) 自愈 pip install websockets/msgpack。North-H20 首跑需确认自愈生效。
