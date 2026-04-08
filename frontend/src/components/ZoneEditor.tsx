import { useRef, useState, useCallback } from "react";
import type { Zone } from "../types";
import { createZone, deleteZone, createFurniture } from "../api/client";
import HeightSlider from "./HeightSlider";

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

const MODE_LABELS: Record<string, { label: string; desc: string; color: string }> = {
  "2d": { label: "2D Flat", desc: "Standard flat zone — no height", color: "var(--text-tertiary)" },
  "auto_3d": { label: "Auto 3D", desc: "Uses depth camera to estimate volume", color: "var(--cyan)" },
  "manual_3d": { label: "Manual 3D", desc: "You set the height range manually", color: "var(--amber)" },
};

/** Render a 3D prism in SVG — base polygon with extruded walls */
function Prism3D({ polygon, heightMin, heightMax, color, maxHeight = 300 }: {
  polygon: number[][];
  heightMin: number;
  heightMax: number;
  color: string;
  maxHeight?: number;
}) {
  if (polygon.length < 3 || heightMax <= heightMin) return null;

  // Scale: 1 unit of height = some visual offset in Y (going "up" = negative Y)
  // and a small X offset for the oblique projection feel
  const hScale = 0.15; // how much Y offset per unit of normalized height
  const xSkew = 0.04;  // slight horizontal offset for depth illusion
  const normalizedHeight = (heightMax - heightMin) / maxHeight;
  const offsetY = -normalizedHeight * hScale;
  const offsetX = normalizedHeight * xSkew;

  const basePath = `M ${polygon.map(([x, y]) => `${x},${y}`).join(" L ")} Z`;
  const topPolygon = polygon.map(([x, y]) => [x + offsetX, y + offsetY]);
  const topPath = `M ${topPolygon.map(([x, y]) => `${x},${y}`).join(" L ")} Z`;

  // Side walls — connect each base vertex to its top counterpart
  const wallPaths = polygon.map(([bx, by], i) => {
    const [tx, ty] = topPolygon[i];
    const ni = (i + 1) % polygon.length;
    const [bnx, bny] = polygon[ni];
    const [tnx, tny] = topPolygon[ni];
    return `M ${bx},${by} L ${tx},${ty} L ${tnx},${tny} L ${bnx},${bny} Z`;
  });

  return (
    <g>
      {/* Side walls */}
      {wallPaths.map((d, i) => (
        <path key={`wall-${i}`} d={d} fill="rgba(168, 85, 247, 0.08)" stroke={color} strokeWidth="0.002" strokeDasharray="0.006 0.004" />
      ))}
      {/* Base */}
      <path d={basePath} fill="rgba(245, 158, 11, 0.18)" stroke={color} strokeWidth="0.003" strokeDasharray="0.01 0.005" />
      {/* Top face */}
      <path d={topPath} fill="rgba(168, 85, 247, 0.2)" stroke={color} strokeWidth="0.003" />
    </g>
  );
}

export default function ZoneEditor({
  zones, panoramaBase64, sweepConfig, onSave, onCancel,
  feedDrawnPolygon, onClearFeedDraw,
}: ZoneEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [rawPoints, setRawPoints] = useState<number[][]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [zoneName, setZoneName] = useState("");
  const [zoneMode, setZoneMode] = useState<"2d" | "auto_3d" | "manual_3d">("2d");
  const [heightMin, setHeightMin] = useState(0);
  const [heightMax, setHeightMax] = useState(75);
  const [saving, setSaving] = useState(false);
  const [drawSource, setDrawSource] = useState<"panorama" | "feed">("panorama");

  // If a polygon was drawn on the live feed, use it
  const hasFeedDraw = feedDrawnPolygon && feedDrawnPolygon.length >= 3;
  // For panorama draws, simplify raw points
  const panoPoints = rawPoints.length > 2 ? simplifyPoints(rawPoints, 0.005) : rawPoints;
  // The active polygon (either from panorama or feed)
  const activeAnglePolygon = hasFeedDraw ? feedDrawnPolygon! : null;
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
    const is3d = zoneMode !== "2d";

    // Get the angle-space polygon
    let anglePoints: number[][];
    if (hasFeedDraw) {
      anglePoints = feedDrawnPolygon!;
    } else {
      anglePoints = panoPoints.map(([nx, ny]) => normalizedToAngle(nx, ny));
    }

    try {
      if (is3d) {
        await createFurniture({
          name,
          base_polygon: anglePoints,
          height_min: heightMin,
          height_max: heightMax,
        });
      }

      await createZone({
        name,
        polygon: anglePoints,
        mode: zoneMode,
        height_min: is3d ? heightMin : 0,
        height_max: is3d ? heightMax : 0,
      });

      setRawPoints([]);
      setZoneName("");
      setZoneMode("2d");
      setHeightMin(0);
      setHeightMax(75);
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
          {/* Existing zones — with 3D prism rendering */}
          {zones.map((zone) => {
            if (!zone.enabled) return null;
            const is3dZone = zone.mode && zone.mode !== "2d";
            const zoneNormalized = zone.polygon.map(([pan, tilt]) => angleToNormalized(pan, tilt));

            if (is3dZone && zone.height_max > zone.height_min) {
              return (
                <Prism3D
                  key={zone.id}
                  polygon={zoneNormalized}
                  heightMin={zone.height_min}
                  heightMax={zone.height_max}
                  color="rgba(245, 158, 11, 0.6)"
                />
              );
            }

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

          {/* Current drawing — with live 3D prism when height is set */}
          {currentPathData && (
            <>
              {zoneMode !== "2d" && heightMax > heightMin ? (
                (() => {
                  const poly = hasFeedDraw
                    ? feedDrawnPolygon!.map(([pan, tilt]) => angleToNormalized(pan, tilt))
                    : (panoPoints.length >= 3 ? panoPoints : []);
                  return poly.length >= 3 ? (
                    <Prism3D
                      polygon={poly}
                      heightMin={heightMin}
                      heightMax={heightMax}
                      color="rgba(245, 158, 11, 0.8)"
                    />
                  ) : (
                    <path
                      d={currentPathData}
                      fill="rgba(245, 158, 11, 0.15)"
                      stroke="var(--amber)"
                      strokeWidth="0.004"
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                  );
                })()
              ) : (
                <path
                  d={currentPathData}
                  fill={isDrawing ? "none" : "rgba(245, 158, 11, 0.15)"}
                  stroke="var(--amber)"
                  strokeWidth="0.004"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              )}
            </>
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

        {/* Mode label */}
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

        {/* Zone type selector */}
        {activePointCount >= 3 && (
          <div style={{ marginTop: 10 }}>
            <div style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--text-ghost)",
              marginBottom: 6,
              letterSpacing: "0.04em",
              textTransform: "uppercase",
            }}>
              Zone type
            </div>

            <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
              {(["2d", "auto_3d", "manual_3d"] as const).map((m) => {
                const cfg = MODE_LABELS[m];
                const active = zoneMode === m;
                return (
                  <button
                    key={m}
                    onClick={() => setZoneMode(m)}
                    style={{
                      flex: 1,
                      padding: "8px 6px",
                      background: active ? "var(--bg-elevated)" : "var(--bg-deep)",
                      border: `1px solid ${active ? cfg.color : "var(--border-subtle)"}`,
                      borderRadius: "var(--radius-sm)",
                      color: active ? cfg.color : "var(--text-ghost)",
                      cursor: "pointer",
                      fontFamily: "var(--font-mono)",
                      fontSize: 10,
                      fontWeight: active ? 600 : 400,
                      textAlign: "center",
                      transition: "all 0.15s",
                    }}
                  >
                    {cfg.label}
                  </button>
                );
              })}
            </div>

            <div style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--text-ghost)",
              marginBottom: zoneMode !== "2d" ? 8 : 0,
              fontStyle: "italic",
            }}>
              {MODE_LABELS[zoneMode].desc}
            </div>

            {zoneMode !== "2d" && (
              <HeightSlider
                heightMin={heightMin}
                heightMax={heightMax}
                onChangeMin={setHeightMin}
                onChangeMax={setHeightMax}
              />
            )}
          </div>
        )}

        {/* Existing zones list */}
        {zones.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div className="label" style={{ marginBottom: 6 }}>Existing Zones</div>
            {zones.map((zone) => {
              const is3d = zone.mode && zone.mode !== "2d";
              return (
                <div
                  key={zone.id}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "7px 10px",
                    background: "var(--bg-deep)",
                    borderRadius: "var(--radius-sm)",
                    borderLeft: `3px solid ${is3d ? "var(--amber)" : "var(--red)"}`,
                    marginBottom: 3,
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                  }}
                >
                  <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                    <span style={{ color: is3d ? "var(--amber)" : "var(--red)" }}>
                      {zone.name}
                    </span>
                    <span style={{ fontSize: 9, color: "var(--text-ghost)" }}>
                      {is3d
                        ? `${zone.mode === "auto_3d" ? "Auto" : "Manual"} 3D · ${zone.height_min}–${zone.height_max} cm`
                        : "2D flat"
                      }
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
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}
