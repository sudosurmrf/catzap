import { useEffect, useState } from "react";
import type { Cat } from "../types";
import { getCats, createCat, deleteCat } from "../api/client";
import Calibration from "./Calibration";

interface SettingsProps {
  onUpdate: () => void;
}

export default function Settings({ onUpdate }: SettingsProps) {
  const [cats, setCats] = useState<Cat[]>([]);
  const [newCatName, setNewCatName] = useState("");

  useEffect(() => {
    getCats().then(setCats).catch(console.error);
  }, []);

  async function handleAddCat() {
    if (!newCatName.trim()) return;
    await createCat(newCatName.trim());
    setNewCatName("");
    const updated = await getCats();
    setCats(updated);
  }

  async function handleDeleteCat(id: string) {
    await deleteCat(id);
    const updated = await getCats();
    setCats(updated);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ padding: 16, background: "#222", borderRadius: 8 }}>
        <h3 style={{ color: "#ccc", fontFamily: "monospace", fontSize: 14, marginTop: 0 }}>Cat Management</h3>
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          <input
            type="text"
            placeholder="Cat name..."
            value={newCatName}
            onChange={(e) => setNewCatName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAddCat()}
            style={{ padding: "8px 12px", background: "#333", border: "1px solid #555", borderRadius: 6, color: "#ccc", fontFamily: "monospace", flex: 1 }}
          />
          <button
            onClick={handleAddCat}
            disabled={!newCatName.trim()}
            style={{
              padding: "8px 16px",
              background: newCatName.trim() ? "#4cc9f0" : "#555",
              color: newCatName.trim() ? "#1a1a2e" : "#888",
              border: "none",
              borderRadius: 6,
              cursor: newCatName.trim() ? "pointer" : "not-allowed",
              fontFamily: "monospace",
            }}
          >
            Add Cat
          </button>
        </div>
        {cats.map((cat) => (
          <div key={cat.id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 12px", background: "#2a2a2a", borderRadius: 6, marginBottom: 4, fontFamily: "monospace", fontSize: 12 }}>
            <span style={{ color: "#ccc" }}>{cat.name}</span>
            <button
              onClick={() => handleDeleteCat(cat.id)}
              style={{ padding: "4px 8px", background: "#f94144", color: "white", border: "none", borderRadius: 4, cursor: "pointer", fontSize: 11 }}
            >
              Delete
            </button>
          </div>
        ))}
      </div>
      <Calibration />
    </div>
  );
}
