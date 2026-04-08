import { useState } from "react";
import { startCalibrationSweep } from "../api/client";

export default function Calibration() {
  const [sweeping, setSweeping] = useState(false);

  async function handleSweep() {
    setSweeping(true);
    await startCalibrationSweep();
    setTimeout(() => setSweeping(false), 3000);
  }

  return (
    <div className="glass-panel-solid" style={{ padding: 14 }}>
      <div className="label" style={{ marginBottom: 8 }}>Panorama Calibration</div>
      <p style={{
        color: "var(--text-tertiary)",
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        lineHeight: 1.5,
        marginBottom: 10,
      }}>
        Run a full sweep to build the panoramic room map. In dev mode, use arrow keys to manually rotate.
      </p>
      <button
        className="btn"
        onClick={handleSweep}
        disabled={sweeping}
        style={{
          width: "100%",
          opacity: sweeping ? 0.5 : 1,
          cursor: sweeping ? "not-allowed" : "pointer",
        }}
      >
        {sweeping ? "Sweeping..." : "Start Sweep"}
      </button>
    </div>
  );
}
