import type { CameraHealth } from "../types";

interface Props { cameras: Record<string, CameraHealth>; }

// Row1: 左腕 | 头部 | 右腕   Row2:      | 中部 |
// mid_head 用 col:2 钉到中间列, 排在头部正下方 (grid 3 列: 1fr 1.4fr 1fr).
const TILES: { key: string; label: string; col?: number }[] = [
  { key: "hand_left", label: "左腕 hand_left (D405)" },
  { key: "top_head", label: "头部 top_head (D435)" },
  { key: "hand_right", label: "右腕 hand_right (D405)" },
  { key: "mid_head", label: "中部 mid_head (D435I)", col: 2 },
];

export function CameraGrid({ cameras }: Props) {
  return (
    <div className="panel area-cams">
      <h3>相机预览</h3>
      <div className="cam-grid">
        {TILES.map(t => {
          const h = cameras[t.key];
          const live = h && h.fps > 0;
          return (
            <div key={t.key} className="cam-tile" style={t.col ? { gridColumn: String(t.col) } : undefined}>
              <span className="cam-label">{t.label}</span>
              {live ? (
                <img
                  src={`/api/camera/${t.key}/mjpeg`}
                  alt={t.key}
                  style={{ width: "100%", background: "#000", borderRadius: 4 }}
                />
              ) : (
                <span style={{ color: "#888" }}>● 等待 ROS2 视频流…</span>
              )}
              {h && <span className="cam-stat">{h.fps} fps · {h.latency_ms} ms · drop {h.dropped}</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
