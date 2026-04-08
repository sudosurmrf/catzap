import { useState } from "react";
import { addCalibrationPoint, clearCalibration } from "../api/client";

export default function Calibration() {
  const [pixelX, setPixelX] = useState(0.5);
  const [pixelY, setPixelY] = useState(0.5);
  const [panAngle, setPanAngle] = useState(90);
  const [tiltAngle, setTiltAngle] = useState(90);
  const [pointsCount, setPointsCount] = useState(0);

  async function handleAddPoint() {
    const result = await addCalibrationPoint(pixelX, pixelY, panAngle, tiltAngle);
    setPointsCount(result.points_count);
  }

  async function handleClear() {
    await clearCalibration();
    setPointsCount(0);
  }

  return (
    <div style={{ padding: 16, background: "#222", borderRadius: 8 }}>
      <h3 style={{ color: "#ccc", fontFamily: "monospace", fontSize: 14, marginTop: 0 }}>
        Servo Calibration ({pointsCount} points)
      </h3>
      <p style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}>
        Map pixel positions to servo angles. Click on the live feed to get pixel coordinates, then set the servo angles that aim at that point.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
        <label style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}>
          Pixel X: {pixelX.toFixed(2)}
          <input type="range" min={0} max={1} step={0.01} value={pixelX} onChange={(e) => setPixelX(Number(e.target.value))} style={{ width: "100%" }} />
        </label>
        <label style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}>
          Pixel Y: {pixelY.toFixed(2)}
          <input type="range" min={0} max={1} step={0.01} value={pixelY} onChange={(e) => setPixelY(Number(e.target.value))} style={{ width: "100%" }} />
        </label>
        <label style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}>
          Pan Angle: {panAngle} deg
          <input type="range" min={0} max={180} value={panAngle} onChange={(e) => setPanAngle(Number(e.target.value))} style={{ width: "100%" }} />
        </label>
        <label style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}>
          Tilt Angle: {tiltAngle} deg
          <input type="range" min={0} max={180} value={tiltAngle} onChange={(e) => setTiltAngle(Number(e.target.value))} style={{ width: "100%" }} />
        </label>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button
          onClick={handleAddPoint}
          style={{ padding: "8px 16px", background: "#4cc9f0", color: "#1a1a2e", border: "none", borderRadius: 6, cursor: "pointer", fontFamily: "monospace" }}
        >
          Add Calibration Point
        </button>
        <button
          onClick={handleClear}
          style={{ padding: "8px 16px", background: "#f94144", color: "white", border: "none", borderRadius: 6, cursor: "pointer", fontFamily: "monospace" }}
        >
          Clear All
        </button>
      </div>
    </div>
  );
}
