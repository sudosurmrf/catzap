import { useEffect, useRef, useState, useCallback } from "react";
import type { Detection, Violation, Zone, FrameData } from "../types";
import StateIndicator from "./StateIndicator";
import DirectionArrow from "./DirectionArrow";
import MultiKillOverlay from "./MultiKillOverlay";
import RadarHUD from "./RadarHUD";
import CatLabelDropdown from "./CatLabelDropdown";

interface FireTarget {
  x: number;
  y: number;
  zone: string;
}

interface LiveFeedProps {
  zones: Zone[];
  onFrameData?: (data: FrameData) => void;
  drawMode?: boolean;
  onDrawComplete?: (anglePolygon: number[][]) => void;
  selectedZoneId?: string | null;
  onSelectZone?: (id: string | null) => void;
  onZonePolygonUpdate?: (zoneId: string, newPolygon: number[][]) => void;
  pendingPolygon?: number[][] | null;
}

const FOV_H = 65;
const FOV_V = 50;

function simplifyPoly(pts: number[][], tol: number): number[][] {
  if (pts.length <= 3) return pts;
  function rdp(p: number[][], s: number, e: number): number[][] {
    let mx = 0, mi = s;
    const [x1, y1] = p[s], [x2, y2] = p[e];
    const dx = x2 - x1, dy = y2 - y1, lsq = dx * dx + dy * dy;
    for (let i = s + 1; i < e; i++) {
      const [px, py] = p[i];
      const d = lsq === 0
        ? Math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        : (() => { const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lsq)); return Math.sqrt((px - x1 - t * dx) ** 2 + (py - y1 - t * dy) ** 2); })();
      if (d > mx) { mx = d; mi = i; }
    }
    if (mx > tol) { const l = rdp(p, s, mi), r = rdp(p, mi, e); return [...l.slice(0, -1), ...r]; }
    return [p[s], p[e]];
  }
  return rdp(pts, 0, pts.length - 1);
}


export default function LiveFeed({
  zones, onFrameData, drawMode, onDrawComplete,
  selectedZoneId, onSelectZone, onZonePolygonUpdate,
  pendingPolygon,
}: LiveFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [fps, setFps] = useState(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [, setViolations] = useState<Violation[]>([]);
  const [zapFlash, setZapFlash] = useState<FireTarget | null>(null);
  const [firedState, setFiredState] = useState(false);
  const [state, setState] = useState("SWEEPING");
  const [servoPan, setServoPan] = useState(90);
  const [servoTilt, setServoTilt] = useState(45);
  const [directionDelta, setDirectionDelta] = useState<{ pan: number; tilt: number } | null>(null);
  const zapTimeoutRef = useRef<number | null>(null);
  const latestFrameRef = useRef<string>("");

  // Drawing state
  const [drawPoints, setDrawPoints] = useState<number[][]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawTool, setDrawTool] = useState<"freehand" | "line">("freehand");
  const [linePoints, setLinePoints] = useState<number[][]>([]);
  const [linePreview, setLinePreview] = useState<number[] | null>(null);
  const linePointsRef = useRef(linePoints);
  linePointsRef.current = linePoints;
  const linePreviewRef = useRef(linePreview);
  linePreviewRef.current = linePreview;
  const drawToolRef = useRef(drawTool);
  drawToolRef.current = drawTool;
  const servoPanRef = useRef(servoPan);
  const servoTiltRef = useRef(servoTilt);
  servoPanRef.current = servoPan;
  servoTiltRef.current = servoTilt;

  const zonesRef = useRef(zones);
  zonesRef.current = zones;
  const selectedZoneIdRef = useRef(selectedZoneId);
  selectedZoneIdRef.current = selectedZoneId;
  const pendingPolygonRef = useRef(pendingPolygon);
  pendingPolygonRef.current = pendingPolygon;
  const rightDragZoneIdRef = useRef<string | null>(null);

  // Right-click drag state
  const [rightDragZoneId, setRightDragZoneId] = useState<string | null>(null);
  const [rightDragStart, setRightDragStart] = useState<{ x: number; y: number } | null>(null);
  const [rightDragOrigPoly, setRightDragOrigPoly] = useState<number[][] | null>(null);
  useEffect(() => { rightDragZoneIdRef.current = rightDragZoneId; }, [rightDragZoneId]);


  // ── Coordinate helpers ──────────────────────────
  function angleToPixelNorm(pan: number, tilt: number): [number, number] {
    return [
      0.5 + (pan - servoPanRef.current) / FOV_H,
      0.5 + (tilt - servoTiltRef.current) / FOV_V,
    ];
  }

  function pixelNormToAngle(px: number, py: number): [number, number] {
    return [
      servoPanRef.current + (px - 0.5) * FOV_H,
      servoTiltRef.current + (py - 0.5) * FOV_V,
    ];
  }

  // Map a mouse event to [0..1] coordinates relative to the actual canvas
  // content area — NOT the full container. The canvas uses objectFit: contain
  // which may letterbox the 4:3 image within a non-4:3 container; without
  // subtracting the letterbox offset, drawn shapes would appear shifted when
  // the zone transitions from the SVG overlay to the canvas rendering.
  const getNormPos = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const bw = canvas.width || 640;
    const bh = canvas.height || 480;
    const scale = Math.min(rect.width / bw, rect.height / bh);
    const cw = bw * scale;
    const ch = bh * scale;
    const ox = (rect.width - cw) / 2;
    const oy = (rect.height - ch) / 2;
    return {
      x: Math.max(0, Math.min(1, (e.clientX - rect.left - ox) / cw)),
      y: Math.max(0, Math.min(1, (e.clientY - rect.top - oy) / ch)),
    };
  }, []);

  // ── Geometry helpers ─────────────────────────────

  // ── Hit testing ─────────────────────────────────
  function pointInPoly(px: number, py: number, poly: number[][]): boolean {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i];
      const [xj, yj] = poly[j];
      if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi)
        inside = !inside;
    }
    return inside;
  }

  function hitTestZone(nx: number, ny: number): string | null {
    for (const zone of zones) {
      if (!zone.enabled) continue;
      const normPoly = zone.polygon.map(([pan, tilt]) => angleToPixelNorm(pan, tilt));
      if (pointInPoly(nx, ny, normPoly)) return zone.id;
    }
    return null;
  }

  // ── WebSocket ───────────────────────────────────
  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/feed`);
    let frameCount = 0;
    let lastFpsTime = Date.now();

    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data) as FrameData;
      if (data.frame) {
        const img = new Image();
        img.onload = () => {
          drawFrame(img, data.detections || [], data.violations || [], data.fired, data.fire_target, data.servo_pan, data.servo_tilt);
          frameCount++;
          const now = Date.now();
          if (now - lastFpsTime >= 1000) { setFps(frameCount); frameCount = 0; lastFpsTime = now; }
        };
        img.src = `data:image/jpeg;base64,${data.frame}`;
        latestFrameRef.current = data.frame;
        setDetections(data.detections || []);
        setViolations(data.violations || []);
        setState(data.state || "SWEEPING");
        setServoPan(data.servo_pan ?? 90);
        setServoTilt(data.servo_tilt ?? 45);
        setDirectionDelta(data.direction_delta ?? null);
        setFiredState(!!data.fired);
        onFrameData?.(data);
        if (data.fired && data.fire_target) {
          setZapFlash(data.fire_target);
          if (zapTimeoutRef.current) clearTimeout(zapTimeoutRef.current);
          zapTimeoutRef.current = window.setTimeout(() => setZapFlash(null), 1500);
        }
      }
    };
    return () => { ws.close(); if (zapTimeoutRef.current) clearTimeout(zapTimeoutRef.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // zones/transform/etc read via refs — no need to reconnect WebSocket on state changes

  useEffect(() => { if (!drawMode) { setDrawPoints([]); setIsDrawing(false); setLinePoints([]); setLinePreview(null); } }, [drawMode]);

  // ── Mouse handlers ──────────────────────────────
  function handleMouseDown(e: React.MouseEvent) {
    // Right-click: drag any zone to reposition
    if (e.button === 2 && onZonePolygonUpdate) {
      e.preventDefault();
      const pos = getNormPos(e);
      if (!pos) return;
      for (const zone of zones) {
        if (!zone.enabled) continue;
        const normPoly = zone.polygon.map(([pan, tilt]) => angleToPixelNorm(pan, tilt));
        if (pointInPoly(pos.x, pos.y, normPoly)) {
          setRightDragZoneId(zone.id);
          setRightDragStart(pos);
          setRightDragOrigPoly([...zone.polygon]);
          return;
        }
      }
      return;
    }

    // Left-click in edit mode: click on a zone to select it and start
    // dragging immediately. Uses the same drag state as right-click drag
    // so both paths share the same real-time move rendering.
    if (!drawMode && onZonePolygonUpdate && !pendingPolygon && e.button === 0) {
      const pos = getNormPos(e);
      if (!pos) return;
      const hitId = hitTestZone(pos.x, pos.y);
      if (hitId) {
        e.preventDefault();
        if (onSelectZone) onSelectZone(hitId);
        setRightDragZoneId(hitId);
        setRightDragStart(pos);
        const zone = zones.find((z) => z.id === hitId);
        if (zone) setRightDragOrigPoly([...zone.polygon]);
        return;
      }
      // Clicked empty space — deselect
      if (onSelectZone && selectedZoneId) onSelectZone(null);
      return;
    }

    // Drawing
    if (drawMode && e.button === 0) {
      e.preventDefault();
      const pos = getNormPos(e);
      if (!pos) return;

      if (drawTool === "line") {
        if (linePoints.length >= 3) {
          const [fx, fy] = linePoints[0];
          if (Math.sqrt((pos.x - fx) ** 2 + (pos.y - fy) ** 2) < 0.03) {
            const closed = [...linePoints, linePoints[0]];
            const anglePolygon = closed.map(([px, py]) => pixelNormToAngle(px, py));
            if (onDrawComplete) onDrawComplete(anglePolygon);
            setLinePoints([]); setLinePreview(null);
            return;
          }
        }
        setLinePoints((prev) => [...prev, [pos.x, pos.y]]);
      } else {
        setIsDrawing(true);
        setDrawPoints([[pos.x, pos.y]]);
      }
    }
  }

  function handleMouseMove(e: React.MouseEvent) {
    // Right-click drag
    if (rightDragZoneId && rightDragStart && rightDragOrigPoly) {
      const pos = getNormPos(e);
      if (!pos) return;
      const dPan = (pos.x - rightDragStart.x) * FOV_H;
      const dTilt = (pos.y - rightDragStart.y) * FOV_V;
      const newPoly = rightDragOrigPoly.map(([pan, tilt]) => [pan + dPan, tilt + dTilt]);
      const zone = zones.find((z) => z.id === rightDragZoneId);
      if (zone) zone.polygon = newPoly;
      return;
    }


    // Drawing
    if (!drawMode) return;
    const pos = getNormPos(e);
    if (!pos) return;

    if (drawTool === "line") {
      if (linePoints.length > 0) setLinePreview([pos.x, pos.y]);
    } else {
      if (!isDrawing) return;
      setDrawPoints((prev) => [...prev, [pos.x, pos.y]]);
    }
  }

  function handleMouseUp() {
    // Commit right-click drag
    if (rightDragZoneId && rightDragOrigPoly && onZonePolygonUpdate) {
      const zone = zones.find((z) => z.id === rightDragZoneId);
      if (zone) onZonePolygonUpdate(rightDragZoneId, zone.polygon);
      setRightDragZoneId(null); setRightDragStart(null); setRightDragOrigPoly(null);
      return;
    }


    // Freehand completion (line mode completes on click near first vertex)
    if (!drawMode || !isDrawing || drawTool === "line") return;
    setIsDrawing(false);
    if (drawPoints.length > 2 && onDrawComplete) {
      const simplified = simplifyPoly(drawPoints, 0.005);
      const closed = [...simplified, simplified[0]];
      const anglePolygon = closed.map(([px, py]) => pixelNormToAngle(px, py));
      onDrawComplete(anglePolygon);
      setDrawPoints([]);
    }
  }

  // ── Main draw function ──────────────────────────
  function drawFrame(
    img: HTMLImageElement, dets: Detection[], viols: Violation[],
    fired: boolean, fireTarget: FireTarget | null, sPan: number, sTilt: number,
  ) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = img.width; canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    const w = canvas.width, h = canvas.height;

    // Read current values from refs so we always draw the latest state
    const currentZones = zonesRef.current;
    const currentSelectedId = selectedZoneIdRef.current;
    const currentRightDrag = rightDragZoneIdRef.current;

    // ── Draw zones ──
    for (const zone of currentZones) {
      if (!zone.enabled) continue;

      const projected = zone.polygon.map(([pan, tilt]) => ({
        px: 0.5 + (pan - sPan) / FOV_H, py: 0.5 + (tilt - sTilt) / FOV_V,
      }));
      if (!projected.some((p) => p.px >= -0.5 && p.px <= 1.5 && p.py >= -0.5 && p.py <= 1.5)) continue;

      const isDragging = zone.id === currentRightDrag;
      const isSelected = zone.id === currentSelectedId;
      const basePx = projected.map((p) => ({ x: p.px * w, y: p.py * h }));

      ctx.strokeStyle = (isDragging || isSelected) ? "rgba(34,211,238,0.9)" : "rgba(239,68,68,0.7)";
      ctx.lineWidth = (isDragging || isSelected) ? 2.5 : 1.5;
      ctx.setLineDash((isDragging || isSelected) ? [] : [6, 4]);
      ctx.beginPath(); basePx.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); }); ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = (isDragging || isSelected) ? "rgba(34,211,238,0.1)" : "rgba(239,68,68,0.06)";
      ctx.fill(); ctx.setLineDash([]);
      if (basePx.length > 0) {
        ctx.fillStyle = (isDragging || isSelected) ? "rgba(34,211,238,0.9)" : "rgba(239,68,68,0.9)";
        ctx.font = "500 11px 'IBM Plex Mono', monospace";
        ctx.fillText(zone.name, basePx[0].x, basePx[0].y - 6);
      }
    }

    // ── Draw pending polygon (newly drawn, not yet saved) ──
    const pp = pendingPolygonRef.current;
    if (pp && pp.length >= 3) {
      const projected = pp.map(([pan, tilt]) => ({
        x: (0.5 + (pan - sPan) / FOV_H) * w,
        y: (0.5 + (tilt - sTilt) / FOV_V) * h,
      }));
      ctx.strokeStyle = "rgba(245,158,11,0.9)"; ctx.lineWidth = 2; ctx.setLineDash([]);
      ctx.beginPath(); projected.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); }); ctx.closePath();
      ctx.stroke(); ctx.fillStyle = "rgba(245,158,11,0.12)"; ctx.fill();
      if (projected.length > 0) {
        ctx.fillStyle = "rgba(245,158,11,0.9)"; ctx.font = "600 10px 'IBM Plex Mono', monospace";
        ctx.fillText("NEW ZONE", projected[0].x, projected[0].y - 4);
      }
    }

    // ── Detections ──
    for (const det of dets) {
      const [x1, y1, x2, y2] = det.bbox;
      const isViolating = viols.some((v) => currentZones.some((z) => z.id === v.zone_id));
      const color = isViolating ? "rgba(239,68,68,0.9)" : "rgba(34,211,238,0.8)";
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash([]);
      const bx = x1 * w, by = y1 * h, bw = (x2 - x1) * w, bh = (y2 - y1) * h;
      const corner = Math.min(bw, bh) * 0.2;
      ctx.beginPath();
      ctx.moveTo(bx, by + corner); ctx.lineTo(bx, by); ctx.lineTo(bx + corner, by);
      ctx.moveTo(bx + bw - corner, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + corner);
      ctx.moveTo(bx + bw, by + bh - corner); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw - corner, by + bh);
      ctx.moveTo(bx + corner, by + bh); ctx.lineTo(bx, by + bh); ctx.lineTo(bx, by + bh - corner);
      ctx.stroke();
      const label = det.cat_name ? `${det.cat_name} ${Math.round(det.confidence * 100)}%` : `Cat ${Math.round(det.confidence * 100)}%`;
      ctx.font = "500 11px 'IBM Plex Mono', monospace";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = isViolating ? "rgba(239,68,68,0.15)" : "rgba(34,211,238,0.12)";
      ctx.fillRect(bx, by - 18, tw + 10, 18);
      ctx.fillStyle = color; ctx.fillText(label, bx + 5, by - 5);
      if (isViolating) { ctx.fillStyle = "rgba(239,68,68,0.9)"; ctx.font = "600 10px 'IBM Plex Mono', monospace"; ctx.fillText("IN ZONE", bx, by + bh + 14); }
      // Draw cat name label
      const catName = det.cat_name || "Unknown";
      const catConf = det.cat_confidence ?? 0;
      ctx.fillStyle = catConf >= 0.6 ? "rgba(16, 185, 129, 0.9)" : "rgba(161, 161, 170, 0.7)";
      ctx.font = "bold 11px 'IBM Plex Mono'";
      ctx.fillText(catName, bx + 4, by + bh - 4);
    }

    // ── Fire crosshair ──
    if (fired && fireTarget) {
      const tx = fireTarget.x * w, ty = fireTarget.y * h, r = 28;
      ctx.fillStyle = "rgba(239,68,68,0.08)"; ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "rgba(239,68,68,0.9)"; ctx.lineWidth = 2; ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(tx, ty, r, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.arc(tx, ty, 4, 0, Math.PI * 2); ctx.fillStyle = "rgba(239,68,68,0.9)"; ctx.fill();
      ctx.beginPath();
      ctx.moveTo(tx - r - 8, ty); ctx.lineTo(tx - 6, ty); ctx.moveTo(tx + 6, ty); ctx.lineTo(tx + r + 8, ty);
      ctx.moveTo(tx, ty - r - 8); ctx.lineTo(tx, ty - 6); ctx.moveTo(tx, ty + 6); ctx.lineTo(tx, ty + r + 8);
      ctx.stroke();
    }

    // ── Line-mode drawing on canvas ──
    const lp = linePointsRef.current;
    const lpv = linePreviewRef.current;
    if (lp.length > 0 && drawToolRef.current === "line") {
      ctx.strokeStyle = "rgba(245,158,11,0.9)"; ctx.lineWidth = 2; ctx.setLineDash([]);
      ctx.beginPath();
      lp.forEach(([nx, ny], i) => {
        const px = nx * w, py = ny * h;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      if (lpv) ctx.lineTo(lpv[0] * w, lpv[1] * h);
      ctx.stroke();
      // Close preview
      if (lpv && lp.length >= 2) {
        ctx.strokeStyle = "rgba(245,158,11,0.4)"; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(lpv[0] * w, lpv[1] * h);
        ctx.lineTo(lp[0][0] * w, lp[0][1] * h); ctx.stroke(); ctx.setLineDash([]);
      }
      // Vertex dots
      lp.forEach(([nx, ny], i) => {
        ctx.fillStyle = i === 0 ? "rgba(34,211,238,0.9)" : "rgba(245,158,11,0.9)";
        ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(nx * w, ny * h, i === 0 ? 6 : 4, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
      });
      if (lp.length >= 3) {
        ctx.fillStyle = "rgba(34,211,238,0.8)"; ctx.font = "500 9px 'IBM Plex Mono', monospace";
        ctx.fillText("click to close", lp[0][0] * w + 10, lp[0][1] * h - 4);
      }
    }
  }

  // ── Cursor ──────────────────────────────────────
  let cursor = "default";
  if (rightDragZoneId) cursor = "grabbing";
  else if (drawMode) cursor = (isDrawing && drawTool === "freehand") ? "none" : "crosshair";

  // Use a 4:3 viewBox so the SVG content area matches the canvas's
  // objectFit: contain letterboxing. xMidYMid meet centers the SVG
  // content identically to how the canvas centers its bitmap.
  const drawPathData = drawMode && drawPoints.length > 1
    ? `M ${drawPoints.map(([x, y]) => `${x * 400},${y * 300}`).join(" L ")}${!isDrawing && drawPoints.length > 2 ? " Z" : ""}`
    : "";

  return (
    <div
      ref={overlayRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        if (isDrawing) handleMouseUp();
        if (rightDragZoneId) handleMouseUp();
      }}
      onContextMenu={(e) => e.preventDefault()}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        background: "var(--bg-deep)",
        cursor,
        userSelect: (drawMode || rightDragZoneId) ? "none" : "auto",
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block", objectFit: "contain", pointerEvents: "none" }}
      />

      {/* Drawing overlay SVG */}
      {drawMode && drawPathData && (
        <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
          viewBox="0 0 400 300" preserveAspectRatio="xMidYMid meet">
          <path d={drawPathData} fill={isDrawing ? "none" : "rgba(245,158,11,0.15)"}
            stroke="var(--amber)" strokeWidth="0.4" strokeLinejoin="round" strokeLinecap="round" />
        </svg>
      )}

      {/* Draw mode banner + tool toggle */}
      {drawMode && (
        <div style={{
          position: "absolute", top: 10, left: "50%", transform: "translateX(-50%)",
          display: "flex", alignItems: "center", gap: 6,
        }}>
          <div style={{
            padding: "4px 14px", background: "rgba(245,158,11,0.85)", color: "var(--bg-deep)",
            borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)", fontSize: 11,
            fontWeight: 600, pointerEvents: "none", letterSpacing: "0.04em",
          }}>
            {drawTool === "line" ? "LINE DRAW — click vertices, click start to close" : "FREEHAND — click and drag"}
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setDrawTool((t) => t === "freehand" ? "line" : "freehand");
              setDrawPoints([]); setIsDrawing(false); setLinePoints([]); setLinePreview(null);
            }}
            style={{
              padding: "4px 10px", background: "rgba(0,0,0,0.6)", backdropFilter: "blur(8px)",
              border: "1px solid var(--amber-dim)", borderRadius: "var(--radius-sm)",
              cursor: "pointer", fontFamily: "var(--font-mono)", fontSize: 10,
              fontWeight: 600, color: "var(--amber)", letterSpacing: "0.04em",
            }}
          >
            {drawTool === "freehand" ? "LINES" : "FREEHAND"}
          </button>
        </div>
      )}


      <DirectionArrow delta={directionDelta} />

      {/* ZAP overlay */}
      {zapFlash && (
        <div style={{
          position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
          background: "var(--red)", color: "white", padding: "14px 28px",
          borderRadius: "var(--radius-md)", fontFamily: "var(--font-display)", fontSize: 26,
          fontWeight: 800, textAlign: "center", animation: "zapPulse 0.3s ease-out",
          pointerEvents: "none", boxShadow: "0 0 60px var(--red-dim), 0 0 120px var(--red-glow)",
          letterSpacing: "0.1em",
        }}>
          ZAP
          <div style={{ fontSize: 12, fontFamily: "var(--font-mono)", fontWeight: 400, marginTop: 2, opacity: 0.85, letterSpacing: "0.02em" }}>
            {zapFlash.zone}
          </div>
        </div>
      )}

      {/* Halo 2 Multi-Kill Medal */}
      <MultiKillOverlay fired={firedState} />

      {/* Cat label prompts for uncertain detections */}
      {!drawMode && detections.map((det, i) => {
        const conf = det.cat_confidence ?? 0;
        if (conf < 0.3 || conf >= 0.6) return null;
        const [x1, , x2, y2] = det.bbox;
        const cx = ((x1 + x2) / 2) * 100;
        const cy = (y2) * 100;
        return (
          <CatLabelDropdown
            key={`label-${i}`}
            bbox={det.bbox}
            frameBase64={latestFrameRef.current}
            onLabeled={() => {}}
            style={{
              position: "absolute",
              left: `${cx}%`,
              top: `${cy}%`,
              transform: "translate(-50%, 4px)",
              zIndex: 15,
            }}
          />
        );
      })}

      {/* Top-left: state badge */}
      {!drawMode && (
        <div style={{ position: "absolute", top: 10, left: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <StateIndicator state={state} />
        </div>
      )}

      {/* Top-right: FPS */}
      {!drawMode && (
        <div style={{
          position: "absolute", top: 10, right: 10, display: "flex", alignItems: "center", gap: 6,
          padding: "3px 8px", background: "rgba(0,0,0,0.5)", backdropFilter: "blur(8px)", borderRadius: 4,
        }}>
          <span style={{
            width: 5, height: 5, borderRadius: "50%",
            background: fps > 0 ? "var(--green)" : "var(--text-ghost)", display: "inline-block",
            boxShadow: fps > 0 ? "0 0 6px var(--green-dim)" : "none",
          }} />
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: fps > 0 ? "var(--text-secondary)" : "var(--text-ghost)", fontVariantNumeric: "tabular-nums" }}>
            {fps > 0 ? `${fps} FPS` : "..."}
          </span>
        </div>
      )}

      {/* Bottom-right: servo + cats */}
      {!drawMode && (
        <div style={{
          position: "absolute", bottom: 10, right: 10, padding: "3px 8px",
          background: "rgba(0,0,0,0.5)", backdropFilter: "blur(8px)", borderRadius: 4,
          fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-tertiary)", fontVariantNumeric: "tabular-nums",
        }}>
          {detections.length > 0 && (
            <span style={{ color: "var(--amber)", marginRight: 8 }}>
              {detections.length} cat{detections.length !== 1 ? "s" : ""}
            </span>
          )}
          {servoPan.toFixed(0)}&deg; / {servoTilt.toFixed(0)}&deg;
        </div>
      )}

      {/* Halo 2 Motion Tracker Radar */}
      {!drawMode && (
        <RadarHUD
          detections={detections}
          servoPan={servoPan}
          servoTilt={servoTilt}
        />
      )}

    </div>
  );
}
