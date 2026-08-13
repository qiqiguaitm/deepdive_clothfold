import { useState } from "react";
import { api } from "../api";
import type { DaggerStatus } from "../types";

function recentMs(ts: number | null | undefined): string {
  if (!ts) return "—";
  // Note: server reports monotonic ts; we don't know browser monotonic so
  // we just compare to "now" wall as a rough freshness indicator.
  const age = Math.max(0, performance.now() / 1000 - ts);
  if (age > 60) return ">60s ago";
  return `${age.toFixed(1)}s ago`;
}

export default function StateCard({ s }: { s: DaggerStatus | null }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const call = async (fn: () => Promise<unknown>) => {
    setBusy(true); setErr(null);
    try { await fn(); } catch (e: any) { setErr(e?.message ?? String(e)); }
    finally { setBusy(false); }
  };
  const state = s?.state ?? "unknown";
  const cls = state === "unknown" ? "state-unknown" : `state-${state}`;
  const rec = !!s?.recording;
  return (
    <div className="card state-card">
      <h2>State</h2>
      <div style={{ marginBottom: 12, display: "flex", alignItems: "center", gap: 12 }}>
        <span className={`state-badge ${cls}`}>{state}</span>
        {rec && (
          <span style={{ color: "#f8514a", fontWeight: 600 }}>
            <span className="rec-dot" />REC
          </span>
        )}
      </div>
      <div className="kv">
        <div className="k">Operation mode</div>
        <div className="v">
          <select className="select" value={s?.operation_mode ?? "observe"}
                  disabled={busy || !!s?.policy_execute}
                  onChange={e => call(() => api.deploymentMode(e.target.value as any))}>
            <option value="observe">OBSERVE</option>
            <option value="deploy">DEPLOY</option>
            <option value="dagger">DAGGER</option>
          </select>
        </div>
        <div className="k">Stack</div>
        <div className="v">
          {s?.stack_running ? (
            <><span className="led led-on" />running (pid {s.stack_pid})</>
          ) : (
            <><span className="led led-off" />stopped</>
          )}
        </div>
        <div className="k">ROS bridge</div>
        <div className="v">{s?.ros_alive ? "alive" : "down"}</div>
        <div className="k">policy execute</div>
        <div className="v">
          {s?.policy_execute === null || s?.policy_execute === undefined
            ? "—"
            : s.policy_execute ? "enabled" : "halted"}
        </div>
        <div className="k">policy node</div>
        <div className="v">{s?.policy_node_ready ? "ready" : "loading / down"}</div>
        <div className="k">Master L / R</div>
        <div className="v">
          <span className={`led ${s?.button_left ? "led-on" : "led-off"}`} />
          L {s?.master_available_left ? (s.button_left ? "TEACH" : "READY") : "N/A"}
          <span style={{ marginLeft: 14 }} />
          <span className={`led ${s?.button_right ? "led-on" : "led-off"}`} />
          R {s?.master_available_right ? (s.button_right ? "TEACH" : "READY") : "N/A"}
        </div>
        <div className="k">last pedal</div>
        <div className="v">{recentMs(s?.last_pedal_ts ?? null)}</div>
      </div>
      <div className="row-buttons" style={{ marginTop: 12 }}>
        <button disabled={busy || !s?.session_running}
                onClick={() => call(() => api.preflight())}>Preflight</button>
        {s?.policy_execute ?
          <button className="danger" disabled={busy}
                  onClick={() => call(() => api.execute(false))}>STOP EXECUTION</button> :
          <button className="primary" disabled={busy || !s?.session_running || s?.operation_mode === "observe"}
                  onClick={() => call(() => api.execute(true))}>ENABLE EXECUTION</button>}
      </div>
      {s?.preflight && <div className={s.preflight.ok ? "hint" : "error"}>
        Preflight: {s.preflight.ok ? "PASS" : `FAIL · ${s.preflight.failures.join(", ")}`}
      </div>}
      {err && <div className="error">{err}</div>}
    </div>
  );
}
