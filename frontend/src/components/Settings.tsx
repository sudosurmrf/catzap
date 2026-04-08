import { useEffect, useState, useRef } from "react";
import type { Cat } from "../types";
import {
  getCats, createCat, deleteCat,
  uploadPhotos, getClassifierInfo, startTraining, getTrainingStatus,
} from "../api/client";
import CalibrationWizard from "./CalibrationWizard";

interface SettingsProps {
  onUpdate: () => void;
}

export default function Settings({ onUpdate }: SettingsProps) {
  const [cats, setCats] = useState<Cat[]>([]);
  const [newCatName, setNewCatName] = useState("");
  const [photoCounts, setPhotoCounts] = useState<Record<string, number>>({});
  const [trainingState, setTrainingState] = useState<string>("idle");
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingAccuracy, setTrainingAccuracy] = useState(0);
  const fileInputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  useEffect(() => {
    getCats().then(setCats).catch(console.error);
  }, []);

  useEffect(() => {
    getClassifierInfo().then((info) => {
      const counts: Record<string, number> = {};
      info.per_cat.forEach((c) => { counts[c.name] = c.photo_count; });
      setPhotoCounts(counts);
    }).catch(console.error);
  }, [cats]);

  useEffect(() => {
    if (trainingState !== "training") return;
    const interval = setInterval(async () => {
      const status = await getTrainingStatus();
      setTrainingState(status.state);
      setTrainingProgress(status.progress);
      setTrainingAccuracy(status.accuracy);
    }, 1000);
    return () => clearInterval(interval);
  }, [trainingState]);

  async function handleUpload(catId: string, files: FileList | null) {
    if (!files || files.length === 0) return;
    await uploadPhotos(catId, files);
    const info = await getClassifierInfo();
    const counts: Record<string, number> = {};
    info.per_cat.forEach((c) => { counts[c.name] = c.photo_count; });
    setPhotoCounts(counts);
  }

  async function handleTrain() {
    setTrainingState("training");
    setTrainingProgress(0);
    try {
      await startTraining();
    } catch (e: any) {
      setTrainingState("error");
    }
  }

  const allCatsReady = cats.length > 0 && cats.every((c) => (photoCounts[c.name] || 0) >= 10);

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
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Cat Management */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 10 }}>Cat Management</div>

        <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
          <input
            type="text"
            placeholder="Cat name..."
            value={newCatName}
            onChange={(e) => setNewCatName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAddCat()}
            style={{ flex: 1, fontSize: 11 }}
          />
          <button
            className="btn btn-primary btn-sm"
            onClick={handleAddCat}
            disabled={!newCatName.trim()}
            style={{ opacity: newCatName.trim() ? 1 : 0.4 }}
          >
            Add
          </button>
        </div>

        {cats.length === 0 && (
          <div style={{
            textAlign: "center",
            padding: 16,
            color: "var(--text-ghost)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
          }}>
            No cats added yet
          </div>
        )}

        {cats.map((cat) => (
          <div
            key={cat.id}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "7px 10px",
              background: "var(--bg-deep)",
              borderRadius: "var(--radius-sm)",
              marginBottom: 3,
              fontFamily: "var(--font-mono)",
              fontSize: 12,
            }}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ color: "var(--text-secondary)" }}>{cat.name}</span>
              <span style={{ fontSize: 9, color: "var(--text-ghost)" }}>
                {photoCounts[cat.name] || 0} photos
              </span>
            </div>
            <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
              <input
                type="file"
                accept="image/*"
                multiple
                ref={(el) => { fileInputRefs.current[cat.id] = el; }}
                style={{ display: "none" }}
                onChange={(e) => handleUpload(cat.id, e.target.files)}
              />
              <button
                className="btn btn-sm"
                onClick={() => fileInputRefs.current[cat.id]?.click()}
                style={{ padding: "2px 8px", fontSize: 10 }}
              >
                Upload
              </button>
              <button
                className="btn btn-danger btn-sm"
                onClick={() => handleDeleteCat(cat.id)}
                style={{ padding: "2px 8px", fontSize: 10 }}
              >
                Remove
              </button>
            </div>
          </div>
        ))}

        {/* Train button */}
        {cats.length > 0 && (
          <div style={{ marginTop: 8 }}>
            {trainingState === "training" ? (
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
                <div style={{ color: "var(--amber)", marginBottom: 4 }}>
                  Training... {trainingProgress.toFixed(0)}%
                </div>
                <div style={{
                  height: 4, background: "var(--bg-elevated)", borderRadius: 2, overflow: "hidden",
                }}>
                  <div style={{
                    width: `${trainingProgress}%`, height: "100%",
                    background: "var(--amber)", borderRadius: 2,
                    transition: "width 0.3s ease",
                  }} />
                </div>
              </div>
            ) : (
              <button
                className="btn btn-primary btn-sm"
                onClick={handleTrain}
                disabled={!allCatsReady || trainingState === "training"}
                style={{ width: "100%", opacity: allCatsReady ? 1 : 0.4 }}
              >
                {trainingState === "complete"
                  ? `Retrain (${(trainingAccuracy * 100).toFixed(0)}% acc)`
                  : "Train Classifier"}
              </button>
            )}
            {!allCatsReady && cats.length > 0 && (
              <div style={{ fontSize: 9, color: "var(--text-ghost)", marginTop: 4, textAlign: "center" }}>
                Need at least 10 photos per cat to train
              </div>
            )}
          </div>
        )}
      </div>

      <CalibrationWizard />
    </div>
  );
}
