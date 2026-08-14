import { useState } from "react";
import { api } from "../api/client";
import type { RecorderSnap, StatusPayload } from "../types";
import { collectFailures } from "./StatusBar";

interface Props {
  rec: RecorderSnap | null;
  status: StatusPayload | null;
  connected: boolean;
  templateId: string;
  operator: string;
  onChanged: () => void;
}

export function Controls({ rec, status, connected, templateId, operator, onChanged }: Props) {
  const [outcome, setOutcome] = useState<"success" | "partial_success" | "failure" | "aborted">("success");
  const [rolloutMode, setRolloutMode] = useState<"demonstration" | "autonomous" | "intervention" | "recovery">("demonstration");
  const [failureModes, setFailureModes] = useState("");
  const [interventionCount, setInterventionCount] = useState(0);
  const [recoverySuccess, setRecoverySuccess] = useState<"na" | "yes" | "no">("na");
  const [unsafeEvent, setUnsafeEvent] = useState(false);
  const [timeLimitReached, setTimeLimitReached] = useState(false);
  const [note, setNote] = useState("");
  const [tags, setTags] = useState("");
  const [busy, setBusy] = useState(false);

  const state = rec?.state || "IDLE";
  const hasMeta = !!templateId && !!operator;
  const canStart = state === "IDLE" && hasMeta;
  const canEnd = state === "RECORDING";
  // 开始按钮保持可点，即便 state===ERROR / 缺 meta / 系统红灯，
  // 这样 onStart 里的弹窗校验才能触发（否则 disabled 按钮吃掉 click，看不到弹窗）。
  const startDisabled = state === "RECORDING" || state === "SAVING" || busy;

  const wrap = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try { await fn(); onChanged(); }
    catch (e: any) { alert(e.message || String(e)); }
    finally { setBusy(false); }
  };

  const onStart = () => {
    if (!hasMeta) {
      alert("请先选择任务 + Prompt 并填写操作员姓名。");
      return;
    }
    if (!connected || !status) {
      alert("系统异常：未连接到后端状态流，无法继续进行，请修复后再采集。");
      return;
    }
    const failures = collectFailures(status);
    if (failures.length > 0) {
      alert(`系统异常，无法继续进行，请修复后再采集：\n- ${failures.join("\n- ")}`);
      return;
    }
    if (state !== "IDLE") {
      alert(`当前录制状态为 ${state}，无法开始新的录制。请先丢弃当前会话。`);
      return;
    }
    wrap(() => api.startRec(templateId, operator));
  };

  return (
    <div className="panel area-ctrl">
      <h3>录制控制</h3>
      <div className="meta-form">
        <span>结果</span>
        <select value={outcome} onChange={e => setOutcome(e.target.value as typeof outcome)}>
          <option value="success">成功</option>
          <option value="partial_success">部分成功</option>
          <option value="failure">失败</option>
          <option value="aborted">中止</option>
        </select>
        <span>轨迹模式</span>
        <select value={rolloutMode} onChange={e => setRolloutMode(e.target.value as typeof rolloutMode)}>
          <option value="demonstration">人工示范</option>
          <option value="autonomous">自主运行</option>
          <option value="intervention">人工干预</option>
          <option value="recovery">恢复尝试</option>
        </select>
        <span>场景标签</span>
        <input value={tags} onChange={e => setTags(e.target.value)} placeholder="逗号分隔，如 light_dim,desk_a" />
        <span>失败模式</span>
        <input value={failureModes} onChange={e => setFailureModes(e.target.value)} placeholder="逗号分隔，如 missed_grasp,timeout" />
        <span>干预次数</span>
        <input type="number" min={0} value={interventionCount} onChange={e => setInterventionCount(Math.max(0, Number(e.target.value) || 0))} />
        <span>恢复成功</span>
        <select value={recoverySuccess} onChange={e => setRecoverySuccess(e.target.value as typeof recoverySuccess)}>
          <option value="na">不适用</option><option value="yes">是</option><option value="no">否</option>
        </select>
        <label><input type="checkbox" checked={unsafeEvent} onChange={e => setUnsafeEvent(e.target.checked)} /> 不安全事件</label>
        <label><input type="checkbox" checked={timeLimitReached} onChange={e => setTimeLimitReached(e.target.checked)} /> 达到时限</label>
        <span>备注</span>
        <textarea value={note} onChange={e => setNote(e.target.value)} rows={2} style={{ gridColumn: "span 3" }} />
      </div>
      <div className="controls">
        <button className="btn-start" disabled={startDisabled}
          onClick={onStart}>● 开始</button>
        <button className="btn-save" disabled={!canEnd || busy}
          onClick={() => wrap(() => api.saveRec({
            success: outcome === "success",
            outcome,
            rollout_mode: rolloutMode,
            failure_modes: failureModes.split(",").map(s => s.trim()).filter(Boolean),
            intervention_count: interventionCount,
            recovery_success: recoverySuccess === "na" ? null : recoverySuccess === "yes",
            unsafe_event: unsafeEvent,
            time_limit_reached: timeLimitReached,
            note,
            scene_tags: tags.split(",").map(s => s.trim()).filter(Boolean),
          }))}>■ 保存</button>
        <button className="btn-discard" disabled={state === "IDLE" || busy}
          onClick={() => wrap(() => api.discardRec())}>✕ 丢弃</button>
      </div>
      {!canStart && state === "IDLE" && (
        <p style={{ color: "var(--muted)", marginTop: 6 }}>请先选择任务 + Prompt 并填写操作员姓名。</p>
      )}
    </div>
  );
}
