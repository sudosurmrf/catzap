import { useEffect, useState } from "react";
import { getControlStatus, setArmed, manualFire } from "../api/client";

export default function Controls() {
  const [armed, setArmedState] = useState(true);
  const [pan, setPan] = useState(90);
  const [tilt, setTilt] = useState(90);
  const [firing, setFiring] = useState(false);

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

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      <div style={{ padding: 16, background: "#222", borderRadius: 8 }}>
        <h3 style={{ color: "#ccc", fontFamily: "monospace", fontSize: 14, marginTop: 0, marginBottom: 12 }}>System</h3>
        <button
          onClick={toggleArm}
          style={{
            width: "100%",
            padding: "12px 16px",
            background: armed ? "#f94144" : "#4cc9f0",
            color: armed ? "white" : "#1a1a2e",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            fontFamily: "monospace",
            fontSize: 14,
            fontWeight: "bold",
          }}
        >
          {armed ? "ARMED — Click to Disarm" : "DISARMED — Click to Arm"}
        </button>
      </div>

      <div style={{ padding: 16, background: "#222", borderRadius: 8 }}>
        <h3 style={{ color: "#ccc", fontFamily: "monospace", fontSize: 14, marginTop: 0, marginBottom: 12 }}>Manual Fire</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <label style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}>
            Pan: {pan} deg
            <input type="range" min={0} max={180} value={pan} onChange={(e) => setPan(Number(e.target.value))} style={{ width: "100%" }} />
          </label>
          <label style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}>
            Tilt: {tilt} deg
            <input type="range" min={0} max={180} value={tilt} onChange={(e) => setTilt(Number(e.target.value))} style={{ width: "100%" }} />
          </label>
          <button
            onClick={handleFire}
            disabled={firing}
            style={{
              padding: "12px 16px",
              background: firing ? "#888" : "#f94144",
              color: "white",
              border: "none",
              borderRadius: 6,
              cursor: firing ? "not-allowed" : "pointer",
              fontFamily: "monospace",
              fontSize: 14,
              fontWeight: "bold",
            }}
          >
            {firing ? "FIRING..." : "FIRE!"}
          </button>
        </div>
      </div>
    </div>
  );
}
