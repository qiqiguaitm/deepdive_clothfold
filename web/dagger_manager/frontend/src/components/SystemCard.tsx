import { useEffect, useState } from "react";
import { api } from "../api";
import type { CkptEntry, ControlPolicyConfig, ControlUpdatePlan, DaggerStatus } from "../types";

interface Props { s: DaggerStatus | null; }

// Supported ckpt directory groups. ckpt_v0 = JAX in-process; ckpt_v1 = V1
// Triton serve + websocket. Extend as new groups are packed (ckpt_v2, ...).
const ALLOWED_GROUPS = new Set<string>(["ckpt_v0", "ckpt_v1"]);

// A ckpt is launchable when its sidecar + variant-specific assets are present.
function ckptOk(c: CkptEntry): boolean {
  if (!c.has_sidecar) return false;
  if (c.variant === "v1") return c.has_v1_pkl && c.has_norm_stats;
  return c.has_norm_stats || !c.config_name;
}

export default function SystemCard({ s }: Props) {
  const [ckpts, setCkpts] = useState<CkptEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [presets, setPresets] = useState<Record<string, ControlPolicyConfig>>({});
  const [preset, setPreset] = useState("production_default");
  const [control, setControl] = useState<ControlPolicyConfig | null>(null);
  const [plan, setPlan] = useState<ControlUpdatePlan | null>(null);

  const reload = async () => {
    setErr(null);
    try {
      const all = await api.ckpts();
      setCkpts(all.filter(c => ALLOWED_GROUPS.has(c.group) && c.has_sidecar));
    } catch (e: any) { setErr(e?.message ?? String(e)); }
  };
  useEffect(() => { reload(); }, []);

  const selectedEntry = ckpts.find(c => c.path === selected) ?? null;
  const selectedValid = selectedEntry ? ckptOk(selectedEntry) : false;

  // While a session is running, lock the selection to the running ckpt.
  useEffect(() => {
    if (s?.session_running && s.ckpt) setSelected(s.ckpt);
  }, [s?.session_running, s?.ckpt]);

  useEffect(() => {
    const variant = selectedEntry?.variant ?? "v0";
    api.controlPresets(variant).then(p => {
      setPresets(p);
      if (!s?.session_running) setControl(p[preset] ?? p.production_default);
    }).catch((e: any) => setErr(e?.message ?? String(e)));
  }, [selectedEntry?.variant]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (s?.session_running && s.control_policy) setControl(s.control_policy);
  }, [s?.session_running, s?.control_policy]);

  // With the bundled lifecycle (start_dagger_collect.sh starts both infra +
  // web), the web cannot itself start/stop infra — the shell terminal owns
  // that. So "system up" here = session (policy_inference) running.
  // infraReady = dagger_recorder publishing /dagger/state, our proxy for
  // "cameras + arms + recorder are all alive".
  const infraReady = s?.state !== null && s?.state !== undefined;
  const sessionUp = !!s?.session_running;
  const systemUp = sessionUp;
  const starting = false;  // brief — system_start is async + short readiness wait

  const start = async () => {
    if (!selected || !selectedEntry) return;
    setErr(null); setBusy(true);
    try { await api.systemStart({ ckpt: selected, variant: selectedEntry.variant,
                                  control_policy: control ?? undefined }); }
    catch (e: any) { setErr(e?.message ?? String(e)); }
    finally { setBusy(false); }
  };

  const choosePreset = (name: string) => {
    setPreset(name); setPlan(null);
    if (presets[name]) setControl(structuredClone(presets[name]));
  };

  const number = (path: string, value: number) => {
    if (!control) return;
    const next = structuredClone(control) as any;
    const [section, key] = path.split(".");
    next[section][key] = value;
    setControl(next); setPlan(null);
  };

  const previewOrApply = async (apply: boolean) => {
    if (!control) return;
    setErr(null); setBusy(true);
    try {
      if (apply) {
        const result = await api.controlApply(control);
        setPlan(result.plan);
      } else setPlan(await api.controlPlan(control));
    } catch (e: any) { setErr(e?.message ?? String(e)); }
    finally { setBusy(false); }
  };
  const stop = async () => {
    setErr(null); setBusy(true);
    try { await api.systemStop(); }
    catch (e: any) { setErr(e?.message ?? String(e)); }
    finally { setBusy(false); }
  };

  return (
    <div className="card ckpt-card">
      <h2>System</h2>
      <div className="system-status-strip">
        <div className="k">Infra</div>
        <div className="v" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className={`led ${infraReady ? "led-on" : "led-off"}`} />
          {infraReady ? "ready (shell-managed)" : "starting up…"}
        </div>
        <div className="k">Session</div>
        <div className="v" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className={`led ${sessionUp ? "led-on" : "led-off"}`} />
          {sessionUp ? `loaded (pid ${s?.session_pid})` : "no policy loaded"}
          <div style={{ marginLeft: "auto" }}>
            {sessionUp ? (
              <button className="danger" onClick={stop} disabled={busy}>
                Stop
              </button>
            ) : (
              <button className="primary" onClick={start}
                      disabled={!selected || !selectedValid || !infraReady || busy}
                      title={!infraReady ? "infra not ready yet" :
                             !selected ? "select a ckpt below" :
                             !selectedValid ? "selected ckpt is missing required assets" :
                             selectedEntry?.variant === "v1"
                               ? "start V1 serve + websocket client (~30s)"
                               : "load JAX policy (~22s)"}>
                Start
              </button>
            )}
          </div>
        </div>
        <div className="system-selected">
          <span>Selected checkpoint</span>
          <div style={{ fontFamily: "monospace", fontSize: 12 }}>
          {selectedEntry ? (
            <>
              <VariantBadge variant={selectedEntry.variant} /> {selected}
            </>
          ) : <span style={{ color: "#8b949e" }}>—</span>}
          </div>
        </div>
      </div>

      <div className="setup-columns">
      <section className="setup-pane">
      <h2 style={{ marginTop: 4, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span>Checkpoint (ckpt_v0 / ckpt_v1)</span>
        <button onClick={reload} disabled={busy}
                style={{ fontSize: 11, padding: "3px 8px" }}>↻</button>
      </h2>
      <div className="ckpt-list">
        {ckpts.map((c) => {
          const ok = ckptOk(c);
          const locked = systemUp && selected !== c.path;
          return (
            <div
              key={c.path}
              className={`ckpt-row ${selected === c.path ? "selected" : ""}`}
              onClick={() => !systemUp && setSelected(c.path)}
              style={{ opacity: locked ? 0.4 : 1, cursor: systemUp ? "default" : "pointer" }}
            >
              <div>{ok ? "✓" : <span className="bad">!</span>}</div>
              <div>
                <div style={{ fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}>
                  <VariantBadge variant={c.variant} />
                  {c.name}
                </div>
                <div className="meta">
                  {c.config_name ?? "—"}
                  {c.task_hint && <> · {c.task_hint}</>}
                  {c.config_name && !c.has_norm_stats && <span className="bad"> · no norm_stats</span>}
                  {c.variant === "v1" && !c.has_v1_pkl && <span className="bad"> · no v1_p200.pkl</span>}
                </div>
              </div>
            </div>
          );
        })}
        {ckpts.length === 0 && <div className="hint">no ckpt_v0 / ckpt_v1 ckpts found</div>}
      </div>
      <div className="hint" style={{ marginTop: 8 }}>
        Infra (CAN/cameras/arms/dagger_recorder/pedal) is managed by the
        shell — Ctrl-C the start_dagger_collect.sh terminal to bring it
        down. Start loads the chosen ckpt: <b>v0</b> = JAX in-process (~22s);
        <b> v1</b> = V1 Triton serve + websocket client (~30s).
      </div>
      </section>

      <section className="setup-pane control-pane">
      <h2 style={{ marginTop: 16 }}>Control Policy · RTC / EMA</h2>
      <div className="control-default-summary">
        <div>
          <strong>{preset.replace(/_/g, " ")}</strong>
          <span>自动按 {selectedEntry?.variant?.toUpperCase() ?? "V0"} 模型配置，一般无需修改</span>
        </div>
        <div className="control-summary-values">
          <code>RTC {control?.rtc.enabled ? "ON" : "OFF"}</code>
          <code>horizon {control?.rtc.execute_horizon ?? "—"}</code>
          <code>EMA α={control?.publish_filter.alpha ?? "—"}</code>
          <code>{control?.timing.inference_rate_hz ?? "—"} Hz</code>
        </div>
      </div>
      <details className="advanced-control">
      <summary>高级参数（仅调试或消融实验时修改）</summary>
      <div className="control-policy-grid">
        <label>Preset
          <select className="select" value={preset}
                  disabled={sessionUp}
                  onChange={e => choosePreset(e.target.value)}>
            {Object.keys(presets).map(name => <option key={name}>{name}</option>)}
          </select>
        </label>
        <label>Inference Hz
          <input type="number" step="0.5" value={control?.timing.inference_rate_hz ?? 3}
                 onChange={e => number("timing.inference_rate_hz", +e.target.value)} />
          <small>HOT</small>
        </label>
        <label>Publish Hz
          <input type="number" value={control?.timing.publish_rate_hz ?? 30}
                 onChange={e => number("timing.publish_rate_hz", +e.target.value)} />
          <small className="restart">RESTART</small>
        </label>
        <label>Speed
          <input type="number" min="0.5" max="2" step="0.1"
                 value={control?.timing.speed_factor ?? 1}
                 onChange={e => number("timing.speed_factor", +e.target.value)} />
          <small>HOT</small>
        </label>
        <label className="check-line">RTC enabled
          <input type="checkbox" checked={control?.rtc.enabled ?? true}
                 onChange={e => control && setControl({...control, rtc: {...control.rtc, enabled: e.target.checked}})} />
          <small className="restart">RESTART</small>
        </label>
        <label>RTC horizon
          <input type="number" value={control?.rtc.execute_horizon ?? 16}
                 onChange={e => number("rtc.execute_horizon", +e.target.value)} />
          <small className="idle">SAFE IDLE</small>
        </label>
        <label>RTC weight
          <input type="number" min="0" max="5" step="0.1"
                 value={control?.rtc.max_guidance_weight ?? 0.5}
                 onChange={e => number("rtc.max_guidance_weight", +e.target.value)} />
          <small className="idle">SAFE IDLE</small>
        </label>
        <label>Latency steps
          <input type="number" value={control?.rtc.latency_steps ?? 8}
                 onChange={e => number("rtc.latency_steps", +e.target.value)} />
          <small className="idle">SAFE IDLE</small>
        </label>
        <label>Blend min/max
          <span className="inline-inputs">
            <input type="number" value={control?.chunk_blend.min_steps ?? 8}
                   onChange={e => number("chunk_blend.min_steps", +e.target.value)} />
            <input type="number" value={control?.chunk_blend.max_steps ?? 12}
                   onChange={e => number("chunk_blend.max_steps", +e.target.value)} />
          </span>
          <small className="idle">SAFE IDLE</small>
        </label>
        <label>EMA alpha
          <input type="number" min="0.01" max="1" step="0.05"
                 value={control?.publish_filter.alpha ?? 0.5}
                 onChange={e => number("publish_filter.alpha", +e.target.value)} />
          <small>HOT · 1.0=OFF</small>
        </label>
      </div>
      <div className="row-buttons" style={{ marginTop: 10 }}>
        <button disabled={!sessionUp || !control || busy}
                onClick={() => previewOrApply(false)}>Preview changes</button>
        <button className="primary" disabled={!sessionUp || !control || busy}
                onClick={() => previewOrApply(true)}>Apply hot changes</button>
      </div>
      {plan && <div className="control-plan">
        {plan.changes.length === 0 ? "No changes" : plan.changes.map(c =>
          <div key={c.field}><code>{c.field}</code>: {String(c.old)} → {String(c.new)}
            <b className={c.classification}>{c.classification}</b></div>)}
        {plan.warnings.map(w => <div className="warn" key={w}>⚠ {w}</div>)}
      </div>}
      </details>
      </section>
      </div>
      {err && <div className="error">{err}</div>}
    </div>
  );
}

function VariantBadge({ variant }: { variant: "v0" | "v1" }) {
  const isV1 = variant === "v1";
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, letterSpacing: "0.04em",
      padding: "1px 6px", borderRadius: 4,
      background: isV1 ? "#1f6feb33" : "#3fb95033",
      color: isV1 ? "#79c0ff" : "#3fb950",
      border: `1px solid ${isV1 ? "#1f6feb66" : "#3fb95066"}`,
      fontFamily: "ui-sans-serif, system-ui, sans-serif",
    }}>
      {variant.toUpperCase()}
    </span>
  );
}
