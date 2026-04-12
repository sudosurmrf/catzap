import { useEffect, useState } from "react";
import {
  getFurniture,
  deleteFurniture,
  startCalibrationSweep,
} from "../api/client";

export default function CalibrationWizard() {
  const [furniture, setFurniture] = useState<any[]>([]);
  const [sweeping, setSweeping] = useState(false);

  useEffect(() => {
    refresh();
  }, []);

  function refresh() {
    getFurniture().then(setFurniture).catch(console.error);
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

      {/* Step 2 — Create zones */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 8 }}>Step 2 — Create Zones</div>
        <p style={{
          fontFamily: "var(--font-mono)", fontSize: 11,
          color: "var(--text-tertiary)", lineHeight: 1.6,
        }}>
          Click the <span style={{ color: "var(--amber)", fontWeight: 600 }}>⬡</span> button in the sidebar (or press Esc to exit) to enter zone editing mode. Draw directly on the live feed or panorama to create exclusion zones.
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
            No furniture objects registered.
          </div>
        ) : (
          <>
            <div style={{
              fontFamily: "var(--font-mono)", fontSize: 10,
              color: "var(--text-ghost)", marginBottom: 6,
            }}>
              {furniture.length} object{furniture.length !== 1 ? "s" : ""}
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
