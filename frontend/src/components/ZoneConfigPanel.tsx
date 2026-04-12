import { useState, useEffect } from "react";
import type { Zone } from "../types";
import { createZone, deleteZone, updateZone } from "../api/client";

interface ZoneConfigPanelProps {
  zones: Zone[];
  pendingPolygon: number[][] | null;
  onSaved: () => void;
  onClearPending: () => void;
  onExit: () => void;
  selectedZoneId?: string | null;
  onSelectZone?: (id: string | null) => void;
  onZoneUpdated?: () => void;
}

export default function ZoneConfigPanel({
  zones, pendingPolygon, onSaved, onClearPending, onExit,
  selectedZoneId, onSelectZone, onZoneUpdated,
}: ZoneConfigPanelProps) {
  const [zoneName, setZoneName] = useState("");
  const [saving, setSaving] = useState(false);

  // Edit mode state
  const [editName, setEditName] = useState("");
  const [editOverlapThreshold, setEditOverlapThreshold] = useState(0.3);
  const [editEnabled, setEditEnabled] = useState(true);
  const [editSaving, setEditSaving] = useState(false);

  // New zone state
  const [overlapThreshold, setOverlapThreshold] = useState(0.3);
  const [enabled, setEnabled] = useState(true);

  const selectedZone = selectedZoneId ? zones.find((z) => z.id === selectedZoneId) : null;
  const hasPoly = pendingPolygon && pendingPolygon.length >= 3;
  const canSave = hasPoly && zoneName.trim().length > 0 && !saving;

  // Sync edit fields when selection changes
  useEffect(() => {
    if (selectedZone) {
      setEditName(selectedZone.name);
      setEditOverlapThreshold(selectedZone.overlap_threshold ?? 0.3);
      setEditEnabled(selectedZone.enabled ?? true);
    }
  }, [selectedZoneId]);

  // ── Save new zone ───────────────────────────────
  async function handleSave() {
    if (!canSave || !pendingPolygon) return;
    setSaving(true);
    try {
      await createZone({
        name: zoneName.trim(),
        polygon: pendingPolygon,
        overlap_threshold: overlapThreshold,
        enabled,
      });
      setZoneName("");
      setOverlapThreshold(0.3);
      setEnabled(true);
      onSaved();
    } finally {
      setSaving(false);
    }
  }

  // ── Save edited zone ────────────────────────────
  async function handleEditSave() {
    if (!selectedZone || editSaving) return;
    setEditSaving(true);
    try {
      await updateZone(selectedZone.id, {
        name: editName.trim() || selectedZone.name,
        polygon: selectedZone.polygon,
        overlap_threshold: editOverlapThreshold,
        enabled: editEnabled,
      });
      onZoneUpdated?.();
    } finally {
      setEditSaving(false);
    }
  }

  async function handleDeleteZone(id: string) {
    if (selectedZoneId === id) onSelectZone?.(null);
    await deleteZone(id);
    onSaved();
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* ── Selected zone editing ─────────────── */}
      {selectedZone ? (
        <div className="glass-panel-solid" style={{ padding: 14 }}>
          <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10,
          }}>
            <div className="label" style={{ color: "var(--cyan)" }}>Edit Zone</div>
            <button className="btn btn-sm" onClick={() => onSelectZone?.(null)}
              style={{ padding: "2px 8px", fontSize: 10 }}>Deselect</button>
          </div>

          <input type="text" value={editName} onChange={(e) => setEditName(e.target.value)}
            style={{ width: "100%", fontSize: 12, marginBottom: 10 }} />

          {/* Sensitivity (overlap_threshold) */}
          <div style={{ marginBottom: 10 }}>
            <div className="label" style={{ marginBottom: 4 }}>
              Sensitivity: {Math.round(editOverlapThreshold * 100)}%
            </div>
            <input type="range" min={0} max={1} step={0.01}
              value={editOverlapThreshold}
              onChange={(e) => setEditOverlapThreshold(parseFloat(e.target.value))}
              style={{ width: "100%" }} />
          </div>

          {/* Enabled toggle */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
            <input type="checkbox" id="edit-enabled" checked={editEnabled}
              onChange={(e) => setEditEnabled(e.target.checked)} />
            <label htmlFor="edit-enabled" className="label" style={{ cursor: "pointer" }}>
              Zone enabled
            </label>
          </div>

          <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
            <button className="btn btn-primary" onClick={handleEditSave} disabled={editSaving}
              style={{ flex: 1, opacity: editSaving ? 0.4 : 1, justifyContent: "center" }}>
              {editSaving ? "Saving..." : "Save Changes"}
            </button>
            <button className="btn" onClick={() => onSelectZone?.(null)}>Cancel</button>
          </div>
        </div>

      ) : !hasPoly ? (
        /* ── Draw prompt ───────────────────────── */
        <div className="glass-panel-solid" style={{ padding: 16, textAlign: "center" }}>
          <div style={{
            fontFamily: "var(--font-display)", fontSize: 13, fontWeight: 600,
            color: "var(--amber)", marginBottom: 8,
          }}>
            Draw a Zone
          </div>
          <p style={{
            fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-tertiary)",
            lineHeight: 1.6, margin: 0,
          }}>
            Draw a 2D outline on the live feed or panorama.
          </p>
          <div style={{
            marginTop: 12, padding: "6px 10px", background: "var(--bg-deep)",
            borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)", fontSize: 10,
            color: "var(--text-ghost)",
          }}>
            Click and drag to draw freehand — or click an existing zone to edit
          </div>
        </div>

      ) : (
        /* ── New zone config (post-draw) ────────── */
        <div className="glass-panel-solid" style={{ padding: 14 }}>
          <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10,
          }}>
            <div className="label">Configure Zone</div>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--green)" }}>
              {pendingPolygon!.length} pts
            </span>
          </div>

          <input type="text" placeholder="Zone name (e.g., Kitchen Table)..."
            value={zoneName} onChange={(e) => setZoneName(e.target.value)}
            style={{ width: "100%", fontSize: 12, marginBottom: 10 }} autoFocus />

          {/* Sensitivity (overlap_threshold) */}
          <div style={{ marginBottom: 10 }}>
            <div className="label" style={{ marginBottom: 4 }}>
              Sensitivity: {Math.round(overlapThreshold * 100)}%
            </div>
            <input type="range" min={0} max={1} step={0.01}
              value={overlapThreshold}
              onChange={(e) => setOverlapThreshold(parseFloat(e.target.value))}
              style={{ width: "100%" }} />
          </div>

          {/* Enabled toggle */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
            <input type="checkbox" id="new-enabled" checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)} />
            <label htmlFor="new-enabled" className="label" style={{ cursor: "pointer" }}>
              Zone enabled
            </label>
          </div>

          <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
            <button className="btn btn-primary" onClick={handleSave} disabled={!canSave}
              style={{ flex: 1, opacity: canSave ? 1 : 0.4, justifyContent: "center" }}>
              {saving ? "Saving..." : "Save Zone"}
            </button>
            <button className="btn" onClick={onClearPending}>Redraw</button>
          </div>
        </div>
      )}

      {/* ── Existing zones list ────────────────── */}
      {zones.length > 0 && (
        <div className="glass-panel-solid" style={{ padding: 14 }}>
          <div className="label" style={{ marginBottom: 8 }}>
            Existing Zones ({zones.length})
          </div>
          {zones.map((zone) => {
            const isSel = zone.id === selectedZoneId;
            return (
              <div key={zone.id}
                onClick={() => !isSel && onSelectZone?.(zone.id)}
                style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "7px 10px",
                  background: isSel ? "var(--bg-elevated)" : "var(--bg-deep)",
                  borderRadius: "var(--radius-sm)",
                  borderLeft: `3px solid ${isSel ? "var(--cyan)" : zone.enabled ? "var(--red)" : "var(--text-ghost)"}`,
                  marginBottom: 3, fontFamily: "var(--font-mono)", fontSize: 11,
                  cursor: isSel ? "default" : "pointer", transition: "all 0.15s",
                }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  <span style={{ color: isSel ? "var(--cyan)" : zone.enabled ? "var(--red)" : "var(--text-ghost)" }}>
                    {zone.name}
                  </span>
                  <span style={{ fontSize: 9, color: "var(--text-ghost)" }}>
                    {zone.enabled ? "active" : "disabled"} · {Math.round((zone.overlap_threshold ?? 0.3) * 100)}% sensitivity
                  </span>
                </div>
                <div style={{ display: "flex", gap: 4 }}>
                  <button className="btn btn-sm"
                    onClick={(e) => { e.stopPropagation(); onSelectZone?.(zone.id); }}
                    style={{ padding: "2px 8px", fontSize: 10, color: "var(--cyan)" }}>
                    Edit
                  </button>
                  <button className="btn btn-danger btn-sm"
                    onClick={(e) => { e.stopPropagation(); handleDeleteZone(zone.id); }}
                    style={{ padding: "2px 8px", fontSize: 10 }}>
                    Del
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <button className="btn btn-danger" onClick={onExit}
        style={{ width: "100%", justifyContent: "center" }}>
        Done Editing
      </button>
    </div>
  );
}
