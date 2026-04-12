import { useRef, useState, useCallback } from "react";
import type { Zone } from "../types";
import { createZone, deleteZone } from "../api/client";

interface SweepConfig {
  panMin: number;
  panMax: number;
  tiltMin: number;
  tiltMax: number;
}

interface ZoneEditorProps {
  zones: Zone[];
  panoramaBase64: string | null;
  sweepConfig: SweepConfig;
  onSave: () => void;
  onCancel: () => void;
  /** Angle-space polygon drawn from the live feed */
  feedDrawnPolygon?: number[][] | null;
  onClearFeedDraw?: () => void;
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

export default function ZoneEditor({
  zones, panoramaBase64, sweepConfig, onSave, onCancel,
  feedDrawnPolygon, onClearFeedDraw,
}: ZoneEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [rawPoints, setRawPoints] = useState<number[][]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [zoneName, setZoneName] = useState("");
  const [saving, setSaving] = useState(false);
  const [, setDrawSource] = useState<"panorama" | "feed">("panorama");

  // If a polygon was drawn on the live feed, use it
  const hasFeedDraw = feedDrawnPolygon && feedDrawnPolygon.length >= 3;
  // For panorama draws, simplify raw points
  const panoPoints = rawPoints.length > 2 ? simplifyPoints(rawPoints, 0.005) : rawPoints;
  const activePointCount = hasFeedDraw ? feedDrawnPolygon!.length : panoPoints.length;

  const normalizedToAngle = useCallback((nx: number, ny: number): number[] => {
    const pan = sweepConfig.panMin + nx * (sweepConfig.panMax - sweepConfig.panMin);
    const tilt = sweepConfig.tiltMin + ny * (sweepConfig.tiltMax - sweepConfig.tiltMin);
    return [pan, tilt];
  }, [sweepConfig]);

  const angleToNormalized = useCallback((pan: number, tilt: number): number[] => {
    return [
      (pan - sweepConfig.panMin) / (sweepConfig.panMax - sweepConfig.panMin),
      (tilt - sweepConfig.tiltMin) / (sweepConfig.tiltMax - sweepConfig.tiltMin),
    ];
  }, [sweepConfig]);

  const getNormalizedPos = useCallback((e: React.MouseEvent) => {
    if (!containerRef.current) return null;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    return [x, y];
  }, []);

  function handlePointerDown(e: React.MouseEvent) {
    if (hasFeedDraw) return; // already have a polygon from feed
    e.preventDefault();
    const pos = getNormalizedPos(e);
    if (!pos) return;
    setIsDrawing(true);
    setRawPoints([pos]);
    setDrawSource("panorama");
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

  function handleClearDraw() {
    setRawPoints([]);
    if (onClearFeedDraw) onClearFeedDraw();
  }

  async function handleSaveZone() {
    const totalPoints = hasFeedDraw ? feedDrawnPolygon!.length : panoPoints.length;
    if (totalPoints < 3 || !zoneName.trim() || saving) return;
    setSaving(true);

    const name = zoneName.trim();

    // Get the angle-space polygon
    let anglePoints: number[][];
    if (hasFeedDraw) {
      anglePoints = feedDrawnPolygon!;
    } else {
      anglePoints = panoPoints.map(([nx, ny]) => normalizedToAngle(nx, ny));
    }

    try {
      await createZone({
        name,
        polygon: anglePoints,
      });

      setRawPoints([]);
      setZoneName("");
      if (onClearFeedDraw) onClearFeedDraw();
      onSave();
    } finally {
      setSaving(false);
    }
  }

  async function handleDeleteZone(id: string) {
    await deleteZone(id);
    onSave();
  }

  // SVG path for panorama-drawn polygon
  const panoPathData = rawPoints.length > 1
    ? `M ${rawPoints.map(([x, y]) => `${x},${y}`).join(" L ")}${!isDrawing && rawPoints.length > 2 ? " Z" : ""}`
    : "";

  // SVG path for feed-drawn polygon (convert from angle-space to panorama-normalized)
  const feedPathData = hasFeedDraw
    ? (() => {
        const normalized = feedDrawnPolygon!.map(([pan, tilt]) => angleToNormalized(pan, tilt));
        return `M ${normalized.map(([x, y]) => `${x},${y}`).join(" L ")} Z`;
      })()
    : "";

  const currentPathData = hasFeedDraw ? feedPathData : panoPathData;
  const canSave = activePointCount >= 3 && zoneName.trim().length > 0 && !saving;

  return (
    <>
      {/* Panorama with drawing overlay */}
      <div
        ref={containerRef}
        onMouseDown={handlePointerDown}
        onMouseMove={handlePointerMove}
        onMouseUp={handlePointerUp}
        onMouseLeave={() => { if (isDrawing) handlePointerUp(); }}
        style={{
          position: "relative",
          cursor: hasFeedDraw ? "default" : (isDrawing ? "none" : "crosshair"),
          userSelect: "none",
          border: `2px solid ${hasFeedDraw ? "var(--cyan)" : isDrawing ? "var(--amber)" : "var(--red)"}`,
          boxSizing: "border-box",
          overflow: "hidden",
          minHeight: 120,
          background: "var(--bg-deep)",
        }}
      >
        {panoramaBase64 ? (
          <img
            src={`data:image/jpeg;base64,${panoramaBase64}`}
            style={{ width: "100%", display: "block", pointerEvents: "none" }}
            alt="Panorama"
          />
        ) : (
          <div style={{
            padding: 40,
            textAlign: "center",
            color: "var(--text-ghost)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
          }}>
            No panorama yet — use arrow keys to sweep
          </div>
        )}

        <svg
          style={{
            position: "absolute",
            top: 0, left: 0,
            width: "100%", height: "100%",
            pointerEvents: "none",
          }}
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
        >
          {/* Existing zones — 2D polygon rendering */}
          {zones.map((zone) => {
            if (!zone.enabled) return null;
            const zoneNormalized = zone.polygon.map(([pan, tilt]) => angleToNormalized(pan, tilt));
            const d = zoneNormalized.length > 1
              ? `M ${zoneNormalized.map(([x, y]) => `${x},${y}`).join(" L ")} Z`
              : "";
            return (
              <path
                key={zone.id}
                d={d}
                fill="rgba(239, 68, 68, 0.12)"
                stroke="rgba(239, 68, 68, 0.6)"
                strokeWidth="0.003"
                strokeDasharray="0.01 0.005"
              />
            );
          })}

          {/* Current drawing */}
          {currentPathData && (
            <path
              d={currentPathData}
              fill={isDrawing ? "none" : "rgba(245, 158, 11, 0.15)"}
              stroke="var(--amber)"
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
              fill="var(--amber)"
              opacity={0.8}
            />
          )}
        </svg>

        {/* Draw source label */}
        <div style={{
          position: "absolute",
          top: 8, left: 8,
          display: "flex", gap: 4,
        }}>
          <span style={{
            padding: "3px 10px",
            background: "rgba(245, 158, 11, 0.85)",
            color: "var(--bg-deep)",
            borderRadius: "var(--radius-sm)",
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            fontWeight: 600,
            pointerEvents: "none",
            letterSpacing: "0.04em",
          }}>
            {hasFeedDraw ? "DRAWN FROM FEED" : "DRAW ON PANORAMA"}
          </span>
          {hasFeedDraw && (
            <span style={{
              padding: "3px 10px",
              background: "rgba(34, 211, 238, 0.2)",
              color: "var(--cyan)",
              borderRadius: "var(--radius-sm)",
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              pointerEvents: "none",
            }}>
              or draw below on the live feed
            </span>
          )}
        </div>
      </div>

      {/* Controls below */}
      <div style={{ padding: "10px 12px", background: "var(--bg-base)" }}>
        {/* Source indicator */}
        {!hasFeedDraw && panoPoints.length === 0 && (
          <div style={{
            padding: "6px 10px",
            background: "var(--bg-deep)",
            borderRadius: "var(--radius-sm)",
            border: "1px dashed var(--border-base)",
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--text-ghost)",
            textAlign: "center",
            marginBottom: 8,
          }}>
            Draw on the panorama above, or draw directly on the live camera feed below
          </div>
        )}

        {/* Name + save row */}
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input
            type="text"
            placeholder="Zone name..."
            value={zoneName}
            onChange={(e) => setZoneName(e.target.value)}
            style={{ flex: 1, fontSize: 11 }}
          />
          <button
            className="btn btn-primary btn-sm"
            onClick={handleSaveZone}
            disabled={!canSave}
            style={{ opacity: canSave ? 1 : 0.4 }}
          >
            {saving ? "Saving..." : `Save (${activePointCount} pts)`}
          </button>
          <button className="btn btn-sm" onClick={handleClearDraw}>
            Clear
          </button>
          <button className="btn btn-danger btn-sm" onClick={onCancel}>
            Done
          </button>
        </div>

        {/* Existing zones list */}
        {zones.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div className="label" style={{ marginBottom: 6 }}>Existing Zones</div>
            {zones.map((zone) => (
              <div
                key={zone.id}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "7px 10px",
                  background: "var(--bg-deep)",
                  borderRadius: "var(--radius-sm)",
                  borderLeft: "3px solid var(--red)",
                  marginBottom: 3,
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                }}
              >
                <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  <span style={{ color: "var(--red)" }}>
                    {zone.name}
                  </span>
                  <span style={{ fontSize: 9, color: "var(--text-ghost)" }}>
                    {zone.polygon.length} pts
                  </span>
                </div>
                <button
                  className="btn btn-danger btn-sm"
                  onClick={() => handleDeleteZone(zone.id)}
                  style={{ padding: "2px 8px", fontSize: 10 }}
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
