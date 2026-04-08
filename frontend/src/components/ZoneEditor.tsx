import { useRef, useState, useCallback } from "react";
import type { Zone } from "../types";
import { createZone, deleteZone } from "../api/client";

interface ZoneEditorProps {
  zones: Zone[];
  canvasEl: HTMLCanvasElement | null;
  onSave: () => void;
  onCancel: () => void;
}

function simplifyPoints(points: number[][], tolerance: number): number[][] {
  if (points.length <= 3) return points;

  function rdp(pts: number[][], start: number, end: number, tol: number): number[][] {
    let maxDist = 0;
    let maxIdx = start;

    const [x1, y1] = pts[start];
    const [x2, y2] = pts[end];
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;

    for (let i = start + 1; i < end; i++) {
      const [px, py] = pts[i];
      let dist: number;
      if (lenSq === 0) {
        dist = Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
      } else {
        const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq));
        const projX = x1 + t * dx;
        const projY = y1 + t * dy;
        dist = Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);
      }
      if (dist > maxDist) {
        maxDist = dist;
        maxIdx = i;
      }
    }

    if (maxDist > tol) {
      const left = rdp(pts, start, maxIdx, tol);
      const right = rdp(pts, maxIdx, end, tol);
      return [...left.slice(0, -1), ...right];
    }
    return [pts[start], pts[end]];
  }

  return rdp(points, 0, points.length - 1, tolerance);
}

export default function ZoneEditor({ zones, canvasEl, onSave, onCancel }: ZoneEditorProps) {
  const [rawPoints, setRawPoints] = useState<number[][]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [zoneName, setZoneName] = useState("");

  const points = rawPoints.length > 2 ? simplifyPoints(rawPoints, 0.005) : rawPoints;

  const getNormalizedPos = useCallback((e: React.MouseEvent) => {
    if (!canvasEl) return null;
    const rect = canvasEl.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    return [x, y];
  }, [canvasEl]);

  function handlePointerDown(e: React.MouseEvent) {
    e.preventDefault();
    const pos = getNormalizedPos(e);
    if (!pos) return;
    setIsDrawing(true);
    setRawPoints([pos]);
  }

  function handlePointerMove(e: React.MouseEvent) {
    if (!isDrawing) return;
    const pos = getNormalizedPos(e);
    if (!pos) return;
    setRawPoints((prev) => [...prev, pos]);
  }

  function handlePointerUp() {
    if (!isDrawing) return;
    setIsDrawing(false);
    if (rawPoints.length > 2) {
      setRawPoints((prev) => [...prev, prev[0]]);
    }
  }

  async function handleSaveZone() {
    if (points.length < 3 || !zoneName.trim()) return;
    await createZone({ name: zoneName.trim(), polygon: points });
    setRawPoints([]);
    setZoneName("");
    onSave();
  }

  async function handleDeleteZone(id: string) {
    await deleteZone(id);
    onSave();
  }

  const pathData = rawPoints.length > 1
    ? `M ${rawPoints.map(([x, y]) => `${x},${y}`).join(" L ")}${!isDrawing && rawPoints.length > 2 ? " Z" : ""}`
    : "";

  // Position the SVG overlay to exactly match the canvas
  const canvasRect = canvasEl?.getBoundingClientRect();
  const parentRect = canvasEl?.parentElement?.getBoundingClientRect();
  const offsetTop = canvasRect && parentRect ? canvasRect.top - parentRect.top : 0;
  const offsetLeft = canvasRect && parentRect ? canvasRect.left - parentRect.left : 0;

  return (
    <>
      {/* Drawing overlay — positioned exactly over the canvas */}
      <div
        onMouseDown={handlePointerDown}
        onMouseMove={handlePointerMove}
        onMouseUp={handlePointerUp}
        onMouseLeave={() => { if (isDrawing) handlePointerUp(); }}
        style={{
          position: "absolute",
          top: offsetTop,
          left: offsetLeft,
          width: canvasRect?.width ?? "100%",
          height: canvasRect?.height ?? "100%",
          cursor: isDrawing ? "none" : "crosshair",
          zIndex: 10,
          userSelect: "none",
          borderRadius: 8,
          border: `2px solid ${isDrawing ? "#4cc9f0" : "#f72585"}`,
          boxSizing: "border-box",
        }}
      >
        <svg
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
          }}
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
        >
          {pathData && (
            <path
              d={pathData}
              fill={isDrawing ? "none" : "rgba(247, 37, 133, 0.2)"}
              stroke={isDrawing ? "#4cc9f0" : "#f72585"}
              strokeWidth="0.004"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          )}

          {isDrawing && rawPoints.length > 0 && (
            <circle
              cx={rawPoints[0][0]}
              cy={rawPoints[0][1]}
              r="0.01"
              fill="#4cc9f0"
              opacity={0.8}
            />
          )}
        </svg>

        {/* Drawing mode label */}
        <div
          style={{
            position: "absolute",
            top: 8,
            left: 8,
            padding: "4px 10px",
            background: "rgba(247, 37, 133, 0.85)",
            color: "white",
            borderRadius: 4,
            fontFamily: "monospace",
            fontSize: 11,
            pointerEvents: "none",
          }}
        >
          DRAW MODE — click and drag to draw zone
        </div>
      </div>

      {/* Controls below the feed */}
      <div style={{ position: "relative", zIndex: 11, marginTop: 8 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <input
            type="text"
            placeholder="Zone name..."
            value={zoneName}
            onChange={(e) => setZoneName(e.target.value)}
            style={{
              padding: "8px 12px",
              background: "#333",
              border: "1px solid #555",
              borderRadius: 6,
              color: "#ccc",
              fontFamily: "monospace",
              flex: 1,
            }}
          />
          <button
            onClick={handleSaveZone}
            disabled={points.length < 3 || !zoneName.trim()}
            style={{
              padding: "8px 16px",
              background: points.length >= 3 && zoneName.trim() ? "#4cc9f0" : "#555",
              color: points.length >= 3 && zoneName.trim() ? "#1a1a2e" : "#888",
              border: "none",
              borderRadius: 6,
              cursor: points.length >= 3 && zoneName.trim() ? "pointer" : "not-allowed",
              fontFamily: "monospace",
            }}
          >
            Save Zone ({points.length} pts)
          </button>
          <button
            onClick={() => setRawPoints([])}
            style={{ padding: "8px 16px", background: "#333", color: "#ccc", border: "none", borderRadius: 6, cursor: "pointer", fontFamily: "monospace" }}
          >
            Clear
          </button>
          <button
            onClick={onCancel}
            style={{ padding: "8px 16px", background: "#f94144", color: "white", border: "none", borderRadius: 6, cursor: "pointer", fontFamily: "monospace" }}
          >
            Done
          </button>
        </div>

        {zones.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <h3 style={{ color: "#ccc", fontFamily: "monospace", fontSize: 14, marginBottom: 8 }}>
              Existing Zones
            </h3>
            {zones.map((zone) => (
              <div
                key={zone.id}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "8px 12px",
                  background: "#222",
                  borderRadius: 6,
                  marginBottom: 4,
                  fontFamily: "monospace",
                  fontSize: 12,
                }}
              >
                <span style={{ color: "#f94144" }}>{zone.name}</span>
                <button
                  onClick={() => handleDeleteZone(zone.id)}
                  style={{ padding: "4px 8px", background: "#f94144", color: "white", border: "none", borderRadius: 4, cursor: "pointer", fontSize: 11 }}
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
