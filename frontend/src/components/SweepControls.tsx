import { useState, useEffect } from "react";
import { getControlStatus, togglePause, emergencyStop, clearEmergencyStop } from "../api/client";

export default function SweepControls() {
  const [paused, setPaused] = useState(false);
  const [stopped, setStopped] = useState(false);
  const [pauseQueued, setPauseQueued] = useState(false);

  useEffect(() => {
    getControlStatus().then((s) => {
      setPaused(s.paused);
      setStopped(s.stopped);
      setPauseQueued(s.pause_queued);
    }).catch(console.error);
  }, []);

  async function handlePause() {
    const res = await togglePause();
    setPaused(res.paused);
    setPauseQueued(res.paused && !paused);
  }

  async function handleEStop() {
    await emergencyStop();
    setStopped(true);
    setPaused(false);
  }

  async function handleClearEStop() {
    await clearEmergencyStop();
    setStopped(false);
  }

  if (stopped) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 4, padding: "0 4px" }}>
        <div style={{
          width: 40, height: 40,
          display: "flex", alignItems: "center", justifyContent: "center",
          borderRadius: "var(--radius-sm)",
          background: "var(--red-glow)",
          border: "1px solid var(--red)",
          color: "var(--red)",
          fontSize: 9,
          fontFamily: "var(--font-mono)",
          fontWeight: 600,
          animation: "pulse 1s infinite",
        }}>
          STOP
        </div>
        <button
          className="nav-btn"
          onClick={handleClearEStop}
          style={{ fontSize: 10, color: "var(--green)" }}
          title="Clear E-Stop"
        >
          ↺
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, padding: "0 4px" }}>
      <div className="tooltip-wrapper" data-tooltip={paused ? "Resume" : pauseQueued ? "Pause queued" : "Pause"}>
        <button
          className={`nav-btn ${paused || pauseQueued ? "active" : ""}`}
          onClick={handlePause}
          style={pauseQueued ? { color: "var(--amber)", opacity: 0.6 } : {}}
        >
          {paused ? "▶" : "⏸"}
        </button>
      </div>
      <div className="tooltip-wrapper" data-tooltip="Emergency Stop">
        <button
          className="nav-btn"
          onClick={handleEStop}
          style={{ color: "var(--red)", fontWeight: 800, fontSize: 14 }}
        >
          ■
        </button>
      </div>
    </div>
  );
}
