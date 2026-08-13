import { useCallback, useEffect, useRef, useState } from "react";
import { api, connectStatusWs } from "./api";
import ArmsPanel from "./components/ArmsPanel";
import CameraGrid from "./components/CameraGrid";
import ControlsCard from "./components/ControlsCard";
import EpisodesCard from "./components/EpisodesCard";
import HistoryCard from "./components/HistoryCard";
import ReplayCard from "./components/ReplayCard";
import StateCard from "./components/StateCard";
import SystemCard from "./components/SystemCard";
import type { DaggerStatus, EpisodeEntry } from "./types";

export default function App() {
  const [snap, setSnap] = useState<DaggerStatus | null>(null);
  const [selectedEp, setSelectedEp] = useState<EpisodeEntry | null>(null);
  const [conn, setConn] = useState<"connecting" | "open" | "closed">("connecting");
  const [task, setTask] = useState("Task_A");
  const [tasks, setTasks] = useState<{ task: string; has_data: boolean }[]>([]);
  const [episodes, setEpisodes] = useState<EpisodeEntry[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const [view, setView] = useState<"live" | "setup" | "data">("live");
  const [stopBusy, setStopBusy] = useState(false);

  // Episode data lives here so EpisodesCard (counts) + HistoryCard (list)
  // share one source — single fetch per task, refreshed when a recording
  // finishes (dagger/inference counts in the WS snapshot bump).
  const reloadEpisodes = useCallback(async () => {
    try { setEpisodes(await api.episodes(task)); } catch { /* backend may be mid-restart */ }
  }, [task]);

  useEffect(() => { api.tasks().then(setTasks).catch(() => {}); }, []);
  useEffect(() => { reloadEpisodes(); }, [reloadEpisodes]);
  useEffect(() => { reloadEpisodes(); /* eslint-disable-next-line */ },
    [snap?.dagger_episodes, snap?.inference_episodes]);

  // WebSocket with auto-reconnect — backend restart shouldn't require browser
  // refresh.
  useEffect(() => {
    let timer: number | null = null;
    let stop = false;
    const open = () => {
      setConn("connecting");
      const ws = connectStatusWs(setSnap);
      wsRef.current = ws;
      ws.onopen = () => setConn("open");
      ws.onclose = () => {
        setConn("closed");
        if (!stop) timer = window.setTimeout(open, 1500);
      };
      ws.onerror = () => { try { ws.close(); } catch {} };
    };
    open();
    return () => {
      stop = true;
      if (timer) clearTimeout(timer);
      try { wsRef.current?.close(); } catch {}
    };
  }, []);

  const state = snap?.state ?? "—";
  const stateCls = state === "—" ? "state-unknown" : `state-${state}`;
  const mode = (snap?.operation_mode ?? "observe").toUpperCase();
  const stopExecution = async () => {
    setStopBusy(true);
    try { await api.execute(false); } catch (e) { console.error(e); }
    finally { setStopBusy(false); }
  };

  return (
    <div className="app">
      {/* ── top bar: at-a-glance status chips ── */}
      <header className="top-bar">
        <div className="brand-block">
          <div className="brand-mark">K0</div>
          <div><h1>KAI0 Operator Console</h1><p>Deployment · DAgger · Replay</p></div>
        </div>
        <span className={`mode-badge mode-${mode.toLowerCase()}`}>{mode}</span>
        <span className={`state-badge ${stateCls}`}>{state}</span>
        {snap?.recording && (
          <span className="chip rec"><span className="rec-dot" />REC</span>
        )}
        <span className={`chip ${snap?.session_running ? "on" : ""}`}>
          {snap?.session_running ? "● policy" : "○ no policy"}
        </span>
        <span className={`chip ${snap?.ros_alive ? "on" : ""}`}>
          {snap?.ros_alive ? "● infra" : "○ infra"}
        </span>
        {(() => {
          const sf = snap?.speed_factor ?? 1.0;
          const fast = sf > 1.001;
          return (
            <span
              className={`chip ${fast ? "rec" : ""}`}
              title="脚踏板油门(切换): 踩一下开加速, 再踩一下回默认; 用过油门的 rollout 会在 episode meta 标 used_throttle=true"
            >
              {fast ? `⏩ ${sf.toFixed(2)}× 油门` : "1.0× 默认速"}
            </span>
          );
        })()}
        <span className="spacer" />
        <div className="conn">ws: {conn}</div>
        <button className="estop" disabled={stopBusy || !snap?.policy_execute}
                onClick={stopExecution}>■ STOP EXECUTION</button>
      </header>

      <nav className="workspace-tabs" aria-label="Workspace">
        <button className={view === "live" ? "active" : ""} onClick={() => setView("live")}>
          <span>01</span> Live Control
        </button>
        <button className={view === "setup" ? "active" : ""} onClick={() => setView("setup")}>
          <span>02</span> Model Setup
          {!snap?.session_running && <i />}
        </button>
        <button className={view === "data" ? "active" : ""} onClick={() => setView("data")}>
          <span>03</span> Data &amp; Replay
        </button>
      </nav>

      {view === "live" && <main className="workspace live-workspace">
        <div className="workspace-heading">
          <div><span>REAL-TIME OPERATIONS</span><h2>Live Control</h2></div>
          <p>Verify camera feeds and preflight before enabling robot execution.</p>
        </div>
        <CameraGrid cameras={snap?.cameras ?? {}} />
        <div className="live-control-grid">
          <StateCard s={snap} />
          <ControlsCard s={snap} />
          <ArmsPanel />
          <EpisodesCard s={snap} task={task} tasks={tasks}
                        episodes={episodes} onTask={setTask} />
        </div>
      </main>}

      {view === "setup" && <main className="workspace setup-workspace">
        <div className="workspace-heading">
          <div><span>POLICY CONFIGURATION</span><h2>Model Setup</h2></div>
          <p>Choose a checkpoint and configure the resolved RTC / EMA control policy.</p>
        </div>
        <SystemCard s={snap} />
      </main>}

      {view === "data" && <main className="workspace data-workspace">
        <div className="workspace-heading">
          <div><span>DATASET REVIEW</span><h2>Data &amp; Replay</h2></div>
          <p>Inspect inference, intervention and stitched rollout episodes.</p>
        </div>
        <EpisodesCard s={snap} task={task} tasks={tasks}
                      episodes={episodes} onTask={setTask} />
        <div className="hr-region">
          <div className="hr-history">
            <HistoryCard task={task} episodes={episodes}
                         selected={selectedEp} onSelect={setSelectedEp}
                         onReload={reloadEpisodes} />
          </div>
          <div className="hr-replay">
            <ReplayCard s={snap} ep={selectedEp} task={task}
                        onDeleted={() => { setSelectedEp(null); reloadEpisodes(); }} />
          </div>
        </div>
      </main>}
    </div>
  );
}
