import { useState, useEffect } from "react";
import type { Zone, ZoneTransform } from "../types";
import { DEFAULT_TRANSFORM } from "../types";
import { createZone, deleteZone, createFurniture, updateZone, updateFurniture, estimateHeight } from "../api/client";
import HeightSlider from "./HeightSlider";

interface ZoneConfigPanelProps {
  zones: Zone[];
  pendingPolygon: number[][] | null;
  onSaved: () => void;
  onClearPending: () => void;
  onExit: () => void;
  selectedZoneId?: string | null;
  onSelectZone?: (id: string | null) => void;
  onZoneUpdated?: () => void;
  transform?: ZoneTransform;
  onTransformChange?: (t: ZoneTransform) => void;
}

const MODE_LABELS: Record<string, { label: string; desc: string; color: string }> = {
  "2d": { label: "2D Flat", desc: "Standard flat zone — no height", color: "var(--text-tertiary)" },
  "auto_3d": { label: "Auto 3D", desc: "Depth camera estimates volume", color: "var(--cyan)" },
  "manual_3d": { label: "Manual 3D", desc: "Set height range manually", color: "var(--amber)" },
};

// Apply transform to polygon (must match PanoramaView logic)
function applyTransform(poly: number[][], t: ZoneTransform): number[][] {
  let sx = 0, sy = 0;
  for (const [x, y] of poly) { sx += x; sy += y; }
  const cx = sx / poly.length, cy = sy / poly.length;
  return poly.map(([x, y]) => {
    let dx = x - cx, dy = y - cy;
    dx *= t.scaleX;
    dy *= t.scaleY;
    const skewedX = dx + dy * t.skewX;
    const skewedY = dy + dx * t.skewY;
    return [cx + skewedX, cy + skewedY];
  });
}

export default function ZoneConfigPanel({
  zones, pendingPolygon, onSaved, onClearPending, onExit,
  selectedZoneId, onSelectZone, onZoneUpdated,
  transform = { ...DEFAULT_TRANSFORM }, onTransformChange,
}: ZoneConfigPanelProps) {
  const [zoneName, setZoneName] = useState("");
  const [saving, setSaving] = useState(false);
  const [estimating, setEstimating] = useState(false);

  // Edit mode state
  const [editName, setEditName] = useState("");
  const [editSaving, setEditSaving] = useState(false);

  const selectedZone = selectedZoneId ? zones.find((z) => z.id === selectedZoneId) : null;
  const hasPoly = pendingPolygon && pendingPolygon.length >= 3;
  const canSave = hasPoly && zoneName.trim().length > 0 && !saving;
  const is3d = transform.height > 0;

  // Sync edit fields when selection changes
  useEffect(() => {
    if (selectedZone) {
      setEditName(selectedZone.name);
    }
  }, [selectedZoneId]);

  // ── Save new zone ───────────────────────────────
  async function handleSave() {
    if (!canSave || !pendingPolygon) return;
    setSaving(true);
    const name = zoneName.trim();

    // Bake the transform into the final polygon
    const finalPoly = applyTransform(pendingPolygon, transform);
    const heightMax = transform.height;
    const mode = heightMax > 0 ? "manual_3d" : "2d";

    try {
      if (heightMax > 0) {
        await createFurniture({
          name,
          base_polygon: finalPoly,
          height_min: 0,
          height_max: heightMax,
        });
      }
      await createZone({
        name,
        polygon: finalPoly,
        mode,
        height_min: 0,
        height_max: heightMax,
      });
      setZoneName("");
      onSaved();
    } finally {
      setSaving(false);
    }
  }

  // ── Auto estimate height ────────────────────────
  async function handleAutoEstimate() {
    if (!pendingPolygon || pendingPolygon.length < 3) return;
    setEstimating(true);
    try {
      const result = await estimateHeight(pendingPolygon);
      if (result.estimated && onTransformChange) {
        onTransformChange({ ...transform, height: result.height_max });
      }
    } catch (e) {
      console.warn("Height estimation failed:", e);
    } finally {
      setEstimating(false);
    }
  }

  // ── Save edited zone ────────────────────────────
  async function handleEditSave() {
    if (!selectedZone || editSaving) return;
    setEditSaving(true);
    const heightMax = transform.height;
    const mode = heightMax > 0 ? "manual_3d" : "2d";
    // Bake width/length/skew transform into the polygon (same as new zone save)
    const finalPoly = applyTransform(selectedZone.polygon, transform);
    try {
      await updateZone(selectedZone.id, {
        name: editName.trim() || selectedZone.name,
        polygon: finalPoly,
        mode,
        height_min: 0,
        height_max: heightMax,
      });
      if (selectedZone.furniture_id && heightMax > 0) {
        await updateFurniture(selectedZone.furniture_id, {
          name: editName.trim() || selectedZone.name,
          height_min: 0,
          height_max: heightMax,
        });
      }
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

  // ── Transform readout component ─────────────────
  function TransformReadout() {
    return (
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr 1fr",
        gap: 4,
        marginTop: 8,
      }}>
        {/* Width (X) */}
        <div style={{
          padding: "6px 8px",
          background: "var(--bg-deep)",
          borderRadius: "var(--radius-sm)",
          borderLeft: "3px solid rgba(239, 68, 68, 0.7)",
        }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "rgba(239, 68, 68, 0.7)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
            Width
          </div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-primary)", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
            {transform.scaleX.toFixed(1)}x
          </div>
        </div>
        {/* Length (Y) */}
        <div style={{
          padding: "6px 8px",
          background: "var(--bg-deep)",
          borderRadius: "var(--radius-sm)",
          borderLeft: "3px solid rgba(34, 197, 94, 0.7)",
        }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "rgba(34, 197, 94, 0.7)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
            Length
          </div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-primary)", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
            {transform.scaleY.toFixed(1)}x
          </div>
        </div>
        {/* Height (Z) */}
        <div style={{
          padding: "6px 8px",
          background: "var(--bg-deep)",
          borderRadius: "var(--radius-sm)",
          borderLeft: "3px solid rgba(59, 130, 246, 0.7)",
        }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "rgba(59, 130, 246, 0.7)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
            Height
          </div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-primary)", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
            {transform.height.toFixed(0)}cm
          </div>
        </div>
        {/* Skew */}
        {(transform.skewX !== 0 || transform.skewY !== 0) && (
          <div style={{
            gridColumn: "1 / -1",
            padding: "4px 8px",
            background: "var(--bg-deep)",
            borderRadius: "var(--radius-sm)",
            borderLeft: "3px solid var(--amber-dim)",
            fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-ghost)",
          }}>
            Skew: {transform.skewX.toFixed(2)} / {transform.skewY.toFixed(2)}
          </div>
        )}
        {(transform.slantX !== 0 || transform.slantY !== 0) && (
          <div style={{
            gridColumn: "1 / -1",
            padding: "4px 8px",
            background: "var(--bg-deep)",
            borderRadius: "var(--radius-sm)",
            borderLeft: "3px solid var(--amber-dim)",
            fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-ghost)",
          }}>
            Slant: {transform.slantX.toFixed(2)} / {transform.slantY.toFixed(2)}
          </div>
        )}
      </div>
    );
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
            style={{ width: "100%", fontSize: 12, marginBottom: 6 }} />

          <TransformReadout />

          <div style={{
            marginTop: 8, padding: "6px 10px", background: "var(--bg-deep)",
            borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)", fontSize: 10,
            color: "var(--text-ghost)", textAlign: "center", lineHeight: 1.5,
          }}>
            Drag <span style={{ color: "rgba(239,68,68,0.9)" }}>W</span> /
            <span style={{ color: "rgba(34,197,94,0.9)" }}> L</span> /
            <span style={{ color: "rgba(59,130,246,0.9)" }}> H</span> arrows to scale.
            Diamonds to skew.
          </div>

          <div style={{ display: "flex", gap: 6, marginTop: 10 }}>
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
            Draw a 2D outline on the live feed or panorama. Then use the
            transform gizmo to scale, extrude height, and skew into 3D.
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
            style={{ width: "100%", fontSize: 12, marginBottom: 6 }} autoFocus />

          {/* Transform readout — live from gizmo */}
          <TransformReadout />

          <div style={{
            marginTop: 8, padding: "6px 10px", background: "var(--bg-deep)",
            borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)", fontSize: 10,
            color: "var(--text-ghost)", textAlign: "center", lineHeight: 1.5,
          }}>
            Drag <span style={{ color: "rgba(239,68,68,0.9)" }}>W</span> to widen,
            <span style={{ color: "rgba(34,197,94,0.9)" }}> L</span> to lengthen,
            <span style={{ color: "rgba(59,130,246,0.9)" }}> H</span> to extrude height.
            Pull <span style={{ color: "var(--amber)" }}>diamonds</span> to skew.
          </div>

          {/* Auto estimate button */}
          <button className="btn btn-sm" onClick={handleAutoEstimate} disabled={estimating}
            style={{
              marginTop: 6, width: "100%", justifyContent: "center",
              opacity: estimating ? 0.4 : 1, fontFamily: "var(--font-mono)", fontSize: 10,
            }}>
            {estimating ? "Estimating..." : "Auto-detect height from depth camera"}
          </button>

          {/* Manual height override */}
          {is3d && (
            <div style={{ marginTop: 8 }}>
              <HeightSlider
                heightMin={0}
                heightMax={Math.round(transform.height)}
                onChangeMin={() => {}}
                onChangeMax={(v) => onTransformChange?.({ ...transform, height: v })}
              />
            </div>
          )}

          <div style={{ display: "flex", gap: 6, marginTop: 10 }}>
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
            const z3d = zone.mode && zone.mode !== "2d";
            const isSel = zone.id === selectedZoneId;
            return (
              <div key={zone.id}
                onClick={() => !isSel && onSelectZone?.(zone.id)}
                style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "7px 10px",
                  background: isSel ? "var(--bg-elevated)" : "var(--bg-deep)",
                  borderRadius: "var(--radius-sm)",
                  borderLeft: `3px solid ${isSel ? "var(--cyan)" : z3d ? "var(--amber)" : "var(--red)"}`,
                  marginBottom: 3, fontFamily: "var(--font-mono)", fontSize: 11,
                  cursor: isSel ? "default" : "pointer", transition: "all 0.15s",
                }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  <span style={{ color: isSel ? "var(--cyan)" : z3d ? "var(--amber)" : "var(--red)" }}>
                    {zone.name}
                  </span>
                  <span style={{ fontSize: 9, color: "var(--text-ghost)" }}>
                    {z3d ? `3D · ${zone.height_min}-${zone.height_max} cm` : "2D flat"}
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
