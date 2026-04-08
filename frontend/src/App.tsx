import { useCallback, useEffect, useState } from "react";
import LiveFeed from "./components/LiveFeed";
import EventLog from "./components/EventLog";
import CatStats from "./components/CatStats";
import Controls from "./components/Controls";
import ZoneEditor from "./components/ZoneEditor";
import Settings from "./components/Settings";
import type { Zone } from "./types";
import { getZones } from "./api/client";

type Tab = "live" | "events" | "stats" | "settings";

export default function App() {
  const [tab, setTab] = useState<Tab>("live");
  const [zones, setZones] = useState<Zone[]>([]);
  const [editingZones, setEditingZones] = useState(false);
  const [canvasEl, setCanvasEl] = useState<HTMLCanvasElement | null>(null);

  useEffect(() => {
    getZones().then(setZones).catch(console.error);
  }, []);

  const refreshZones = () => getZones().then(setZones).catch(console.error);

  const handleCanvasRef = useCallback((el: HTMLCanvasElement | null) => {
    setCanvasEl(el);
  }, []);

  return (
    <div
      style={{
        maxWidth: 1200,
        margin: "0 auto",
        padding: 16,
        background: "#1a1a2e",
        minHeight: "100vh",
        color: "#ccc",
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <h1 style={{ margin: 0, fontSize: 24, fontFamily: "monospace", color: "#4cc9f0" }}>
          CatZap
        </h1>
        <nav style={{ display: "flex", gap: 8 }}>
          {(["live", "events", "stats", "settings"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                padding: "8px 16px",
                background: tab === t ? "#4cc9f0" : "#333",
                color: tab === t ? "#1a1a2e" : "#ccc",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                fontFamily: "monospace",
                textTransform: "capitalize",
              }}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      {tab === "live" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ position: "relative" }}>
            <LiveFeed zones={zones} canvasRefCallback={handleCanvasRef} />
            {editingZones && (
              <ZoneEditor
                zones={zones}
                canvasEl={canvasEl}
                onSave={() => {
                  setEditingZones(false);
                  refreshZones();
                }}
                onCancel={() => setEditingZones(false)}
              />
            )}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={() => setEditingZones(!editingZones)}
              style={{
                padding: "8px 16px",
                background: editingZones ? "#f94144" : "#f72585",
                color: "white",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                fontFamily: "monospace",
              }}
            >
              {editingZones ? "Cancel" : "+ Draw Zone"}
            </button>
          </div>
          <Controls />
        </div>
      )}

      {tab === "events" && <EventLog />}
      {tab === "stats" && <CatStats />}
      {tab === "settings" && <Settings onUpdate={refreshZones} />}
    </div>
  );
}
