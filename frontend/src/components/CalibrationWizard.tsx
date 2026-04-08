import { useEffect, useState } from "react";
import {
  getRoomModelStatus,
  calibrateDepthScale,
  getFurniture,
  deleteFurniture,
  startCalibrationSweep,
} from "../api/client";

type CalibrationStep = "idle" | "measuring";

export default function CalibrationWizard() {
  const [roomStatus, setRoomStatus] = useState<any>(null);
  const [furniture, setFurniture] = useState<any[]>([]);
  const [sweeping, setSweeping] = useState(false);

  // Calibration flow
  const [calStep, setCalStep] = useState<CalibrationStep>("idle");
  const [refDistance, setRefDistance] = useState("");
  const [calibrated, setCalibrated] = useState(false);

  useEffect(() => {
    refresh();
  }, []);

  function refresh() {
    getRoomModelStatus().then(setRoomStatus).catch(console.error);
    getFurniture().then(setFurniture).catch(console.error);
  }

  async function handleCalibrate() {
    const dist = parseFloat(refDistance);
    if (isNaN(dist) || dist <= 0) return;
    await calibrateDepthScale(dist);
    setCalibrated(true);
    setCalStep("idle");
    refresh();
  }

  async function handleDeleteFurniture(id: string) {
    await deleteFurniture(id);
    refresh();
  }

  async function handleSweep() {
    setSweeping(true);
    await startCalibrationSweep();
    setTimeout(() => { setSweeping(false); refresh(); }, 3000);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Room model status */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 8 }}>Room Model</div>
        {roomStatus ? (
          <div style={{
            fontFamily: "var(--font-mono)", fontSize: 11,
            color: "var(--text-tertiary)", lineHeight: 1.8,
          }}>
            <div>
              Status:{" "}
              <span style={{ color: roomStatus.initialized ? "var(--green)" : "var(--text-ghost)" }}>
                {roomStatus.initialized ? "Active" : "Not initialized"}
              </span>
            </div>
            {roomStatus.initialized && (
              <>
                <div>Room: {roomStatus.width_cm} x {roomStatus.depth_cm} cm</div>
                <div>
                  Depth scale:{" "}
                  <span style={{ color: roomStatus.depth_scale > 1 ? "var(--green)" : "var(--amber)" }}>
                    {roomStatus.depth_scale > 1 ? roomStatus.depth_scale.toFixed(1) : "not calibrated"}
                  </span>
                </div>
              </>
            )}
          </div>
        ) : (
          <div style={{ color: "var(--text-ghost)", fontFamily: "var(--font-mono)", fontSize: 11 }}>
            Loading...
          </div>
        )}
      </div>

      {/* Panorama sweep */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 8 }}>Step 1 — Build Panorama</div>
        <p style={{
          fontFamily: "var(--font-mono)", fontSize: 11,
          color: "var(--text-tertiary)", lineHeight: 1.5, marginBottom: 8,
        }}>
          Run a full sweep so the camera captures every angle of the room. This builds the panoramic map that zones are drawn on.
        </p>
        <button
          className="btn"
          onClick={handleSweep}
          disabled={sweeping}
          style={{ width: "100%", opacity: sweeping ? 0.5 : 1 }}
        >
          {sweeping ? "Sweeping..." : "Start Panorama Sweep"}
        </button>
      </div>

      {/* Depth calibration */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 8 }}>Step 2 — Calibrate Depth</div>

        {calStep === "idle" && !calibrated && (
          <>
            <p style={{
              fontFamily: "var(--font-mono)", fontSize: 11,
              color: "var(--text-tertiary)", lineHeight: 1.6, marginBottom: 10,
            }}>
              The depth camera estimates relative distances. To convert these into real-world centimeters, you need to give it one reference measurement.
            </p>
            <div style={{
              padding: "10px 12px",
              background: "var(--bg-deep)",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-subtle)",
              fontFamily: "var(--font-mono)", fontSize: 11,
              color: "var(--text-secondary)", lineHeight: 1.7,
              marginBottom: 10,
            }}>
              <div style={{ fontWeight: 600, color: "var(--amber)", marginBottom: 4 }}>How to calibrate:</div>
              <div>1. Pick an object visible in the camera feed</div>
              <div>2. Measure its real-world distance from the camera with a tape measure (in cm)</div>
              <div style={{ fontSize: 10, color: "var(--text-ghost)", marginLeft: 16, marginTop: 2 }}>
                e.g., "the front edge of the table is 180cm from the camera"
              </div>
              <div style={{ marginTop: 4 }}>3. Enter that distance below</div>
            </div>
            <button className="btn btn-primary" onClick={() => setCalStep("measuring")} style={{ width: "100%" }}>
              I have a measurement ready
            </button>
          </>
        )}

        {calStep === "measuring" && (
          <>
            <p style={{
              fontFamily: "var(--font-mono)", fontSize: 11,
              color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: 8,
            }}>
              Enter the distance from the camera to your reference object:
            </p>
            <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
              <input
                type="text"
                value={refDistance}
                onChange={(e) => setRefDistance(e.target.value)}
                placeholder="e.g., 180"
                style={{ flex: 1, fontSize: 12 }}
                autoFocus
              />
              <span style={{
                fontFamily: "var(--font-mono)", fontSize: 12,
                color: "var(--text-tertiary)",
                display: "flex", alignItems: "center",
              }}>
                cm
              </span>
            </div>
            <div style={{
              padding: "8px 10px",
              background: "var(--bg-deep)",
              borderRadius: "var(--radius-sm)",
              fontFamily: "var(--font-mono)", fontSize: 10,
              color: "var(--text-ghost)", lineHeight: 1.5, marginBottom: 10,
            }}>
              Tip: Choose something at mid-range (1–3 meters). A table edge, chair, or doorframe works well. Avoid objects very close to or far from the camera.
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <button
                className="btn btn-primary"
                onClick={handleCalibrate}
                disabled={!refDistance || isNaN(parseFloat(refDistance)) || parseFloat(refDistance) <= 0}
                style={{
                  flex: 1,
                  opacity: refDistance && parseFloat(refDistance) > 0 ? 1 : 0.4,
                }}
              >
                Set Depth Scale
              </button>
              <button className="btn" onClick={() => setCalStep("idle")}>
                Back
              </button>
            </div>
          </>
        )}

        {calStep === "idle" && calibrated && (
          <div style={{
            display: "flex", flexDirection: "column", gap: 8,
          }}>
            <div style={{
              padding: "10px 12px",
              background: "var(--green-dim)",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--green)",
              fontFamily: "var(--font-mono)", fontSize: 11,
              color: "var(--green)",
              display: "flex", alignItems: "center", gap: 6,
            }}>
              <span style={{ fontSize: 14 }}>&#10003;</span>
              Depth calibrated — {roomStatus?.depth_scale?.toFixed(1)} scale factor
            </div>
            <button className="btn btn-sm" onClick={() => { setCalibrated(false); setCalStep("idle"); setRefDistance(""); }}>
              Recalibrate
            </button>
          </div>
        )}
      </div>

      {/* Step 3 — Create zones */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 8 }}>Step 3 — Create Zones</div>
        <p style={{
          fontFamily: "var(--font-mono)", fontSize: 11,
          color: "var(--text-tertiary)", lineHeight: 1.6, marginBottom: 4,
        }}>
          Click the <span style={{ color: "var(--amber)", fontWeight: 600 }}>⬡</span> button in the sidebar (or press Esc to exit) to enter zone editing mode. Draw directly on the live feed or panorama to create exclusion zones.
        </p>
        <p style={{
          fontFamily: "var(--font-mono)", fontSize: 10,
          color: "var(--text-ghost)", lineHeight: 1.5,
        }}>
          For 3D zones, choose "Auto 3D" or "Manual 3D" after drawing, then set the height. A 3D zone creates a furniture volume — a cat ON or INSIDE the volume will trigger.
        </p>
      </div>

      {/* Furniture objects */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <div className="label">Furniture Objects</div>
          <button className="btn btn-sm" onClick={refresh} style={{ padding: "2px 8px", fontSize: 9 }}>
            Refresh
          </button>
        </div>

        {furniture.length === 0 ? (
          <div style={{
            textAlign: "center", padding: 16,
            color: "var(--text-ghost)", fontFamily: "var(--font-mono)", fontSize: 11,
            lineHeight: 1.6,
          }}>
            No furniture yet. Create a 3D zone to automatically register furniture.
          </div>
        ) : (
          <>
            <div style={{
              fontFamily: "var(--font-mono)", fontSize: 10,
              color: "var(--text-ghost)", marginBottom: 6,
            }}>
              {furniture.length} object{furniture.length !== 1 ? "s" : ""} — used for occlusion tracking
            </div>
            {furniture.map((f) => (
              <div
                key={f.id}
                style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "7px 10px", background: "var(--bg-deep)",
                  borderRadius: "var(--radius-sm)",
                  borderLeft: "3px solid var(--amber)",
                  marginBottom: 3,
                  fontFamily: "var(--font-mono)", fontSize: 11,
                }}
              >
                <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  <span style={{ color: "var(--amber)" }}>{f.name}</span>
                  <span style={{ fontSize: 9, color: "var(--text-ghost)" }}>
                    {f.height_min}–{f.height_max} cm
                    {f.depth_anchored && " · depth-anchored"}
                  </span>
                </div>
                <button
                  className="btn btn-danger btn-sm"
                  onClick={() => handleDeleteFurniture(f.id)}
                  style={{ padding: "2px 8px", fontSize: 10 }}
                >
                  Remove
                </button>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
