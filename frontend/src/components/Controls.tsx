import { useEffect, useState } from "react";
import { getControlStatus, setArmed, manualFire, setVirtualAngle, startCalibrationSweep } from "../api/client";

interface ControlsProps {
  servoPan?: number;
  servoTilt?: number;
}

export default function Controls({ servoPan = 90, servoTilt = 45 }: ControlsProps) {
  const [armed, setArmedState] = useState(true);
  const [pan, setPan] = useState(90);
  const [tilt, setTilt] = useState(45);
  const [firing, setFiring] = useState(false);
  const [sweeping, setSweeping] = useState(false);

  useEffect(() => {
    getControlStatus().then((s) => setArmedState(s.armed)).catch(console.error);
  }, []);

  async function toggleArm() {
    const newState = !armed;
    await setArmed(newState);
    setArmedState(newState);
  }

  async function handleFire() {
    setFiring(true);
    await manualFire(pan, tilt);
    setTimeout(() => setFiring(false), 500);
  }

  async function handleSetAngle() {
    await setVirtualAngle(pan, tilt);
  }

  async function handleCalibrationSweep() {
    setSweeping(true);
    await startCalibrationSweep();
    setTimeout(() => setSweeping(false), 3000);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Arm / Disarm */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 10 }}>System</div>

        <button
          onClick={toggleArm}
          style={{
            width: "100%",
            padding: "12px 16px",
            background: armed ? "var(--red-glow)" : "var(--green-dim)",
            border: `1px solid ${armed ? "var(--red)" : "var(--green)"}`,
            color: armed ? "var(--red)" : "var(--green)",
            borderRadius: "var(--radius-sm)",
            cursor: "pointer",
            fontFamily: "var(--font-display)",
            fontSize: 13,
            fontWeight: 700,
            letterSpacing: "0.06em",
            transition: "all 0.2s",
            boxShadow: armed ? "var(--shadow-glow-red)" : "0 0 20px var(--green-dim)",
          }}
        >
          {armed ? "ARMED" : "DISARMED"}
          <span style={{
            display: "block",
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            fontWeight: 400,
            opacity: 0.7,
            marginTop: 2,
          }}>
            Click to {armed ? "disarm" : "arm"} system
          </span>
        </button>

        <div style={{
          marginTop: 10,
          padding: "6px 10px",
          background: "var(--bg-deep)",
          borderRadius: "var(--radius-sm)",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--text-tertiary)",
          fontVariantNumeric: "tabular-nums",
        }}>
          Current: {servoPan.toFixed(0)}&deg; pan / {servoTilt.toFixed(0)}&deg; tilt
        </div>
      </div>

      {/* Manual Control */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 10 }}>Manual Aim</div>

        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div>
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--text-tertiary)",
              marginBottom: 2,
            }}>
              <span>Pan</span>
              <span style={{ color: "var(--amber)", fontVariantNumeric: "tabular-nums" }}>{pan}&deg;</span>
            </div>
            <input
              type="range" min={0} max={180} value={pan}
              onChange={(e) => setPan(Number(e.target.value))}
            />
          </div>

          <div>
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--text-tertiary)",
              marginBottom: 2,
            }}>
              <span>Tilt</span>
              <span style={{ color: "var(--amber)", fontVariantNumeric: "tabular-nums" }}>{tilt}&deg;</span>
            </div>
            <input
              type="range" min={0} max={180} value={tilt}
              onChange={(e) => setTilt(Number(e.target.value))}
            />
          </div>

          <div style={{ display: "flex", gap: 6 }}>
            <button className="btn btn-primary" onClick={handleSetAngle} style={{ flex: 1 }}>
              Aim
            </button>
            <button
              className={`btn ${firing ? "" : "btn-danger"}`}
              onClick={handleFire}
              disabled={firing}
              style={{ flex: 1, opacity: firing ? 0.5 : 1, cursor: firing ? "not-allowed" : "pointer" }}
            >
              {firing ? "Firing..." : "Fire"}
            </button>
          </div>
        </div>
      </div>

      {/* Calibration */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 10 }}>Calibration</div>
        <button
          className="btn"
          onClick={handleCalibrationSweep}
          disabled={sweeping}
          style={{
            width: "100%",
            opacity: sweeping ? 0.5 : 1,
            cursor: sweeping ? "not-allowed" : "pointer",
          }}
        >
          {sweeping ? "Sweeping..." : "Run Calibration Sweep"}
        </button>
      </div>
    </div>
  );
}
