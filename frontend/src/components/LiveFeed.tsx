import { useEffect, useRef, useState, useCallback } from "react";
import type { Detection, Violation, Zone, FrameData, ZoneTransform } from "../types";
import { DEFAULT_TRANSFORM } from "../types";
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
  // Transform / editing props (same as PanoramaView)
  selectedZoneId?: string | null;
  onSelectZone?: (id: string | null) => void;
  onZonePolygonUpdate?: (zoneId: string, newPolygon: number[][]) => void;
  transform?: ZoneTransform;
  onTransformChange?: (t: ZoneTransform) => void;
  originalPolygon?: number[][] | null;
  pendingPolygon?: number[][] | null;
}

const FOV_H = 65;
const FOV_V = 50;

// Gizmo constants (matching PanoramaView)
type GizmoHandle = "none" | "move" | "x-axis" | "y-axis" | "z-axis"
  | "skew-top" | "skew-right" | "skew-bottom" | "skew-left" | "slant";

const ARROW_LEN = 50;
const HANDLE_SIZE = 7;
const SKEW_HANDLE_SIZE = 5;
const Z_DIR = { x: 0.15, y: -1.0 };
const Z_LEN = Math.sqrt(Z_DIR.x ** 2 + Z_DIR.y ** 2);
const Z_UNIT = { x: Z_DIR.x / Z_LEN, y: Z_DIR.y / Z_LEN };

export default function LiveFeed({
  zones, onFrameData, drawMode, onDrawComplete,
  selectedZoneId, onSelectZone, onZonePolygonUpdate,
  transform = { ...DEFAULT_TRANSFORM }, onTransformChange,
  originalPolygon, pendingPolygon,
}: LiveFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [fps, setFps] = useState(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [zapFlash, setZapFlash] = useState<FireTarget | null>(null);
  const [firedState, setFiredState] = useState(false);
  const [state, setState] = useState("SWEEPING");
  const [servoPan, setServoPan] = useState(90);
  const [servoTilt, setServoTilt] = useState(45);
  const [warningRemaining, setWarningRemaining] = useState(0);
  const [directionDelta, setDirectionDelta] = useState<{ pan: number; tilt: number } | null>(null);
  const [occludedCats, setOccludedCats] = useState<{ id: string; predicted: [number, number, number]; occluded_by: string }[]>([]);
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

  // Refs so drawFrame always reads current values (not stale closure)
  const transformRef = useRef(transform);
  transformRef.current = transform;
  const zonesRef = useRef(zones);
  zonesRef.current = zones;
  const selectedZoneIdRef = useRef(selectedZoneId);
  selectedZoneIdRef.current = selectedZoneId;
  const originalPolygonRef = useRef(originalPolygon);
  originalPolygonRef.current = originalPolygon;
  const pendingPolygonRef = useRef(pendingPolygon);
  pendingPolygonRef.current = pendingPolygon;
  const activeHandleRef = useRef<GizmoHandle>("none");
  const rightDragZoneIdRef = useRef<string | null>(null);

  // Gizmo drag state
  const [activeHandle, setActiveHandle] = useState<GizmoHandle>("none");
  const [dragStartNorm, setDragStartNorm] = useState<{ x: number; y: number } | null>(null);
  const [dragStartTransform, setDragStartTransform] = useState<ZoneTransform | null>(null);
  // Keep ref in sync
  useEffect(() => { activeHandleRef.current = activeHandle; }, [activeHandle]);

  // Right-click drag state
  const [rightDragZoneId, setRightDragZoneId] = useState<string | null>(null);
  const [rightDragStart, setRightDragStart] = useState<{ x: number; y: number } | null>(null);
  const [rightDragOrigPoly, setRightDragOrigPoly] = useState<number[][] | null>(null);
  useEffect(() => { rightDragZoneIdRef.current = rightDragZoneId; }, [rightDragZoneId]);

  // Which polygon the gizmo targets
  const targetPolygon = pendingPolygon || (selectedZoneId ? originalPolygon : null);
  const showGizmo = !!targetPolygon && targetPolygon.length >= 3;

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

  const getNormPos = useCallback((e: React.MouseEvent) => {
    if (!overlayRef.current) return null;
    const rect = overlayRef.current.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height)),
    };
  }, []);

  // ── Transform math (same as PanoramaView) ──────
  function getCentroid(poly: number[][]): [number, number] {
    let sx = 0, sy = 0;
    for (const [x, y] of poly) { sx += x; sy += y; }
    return [sx / poly.length, sy / poly.length];
  }

  function applyTransform(poly: number[][], t: ZoneTransform): number[][] {
    const [cx, cy] = getCentroid(poly);
    return poly.map(([x, y]) => {
      let dx = x - cx, dy = y - cy;
      dx *= t.scaleX; dy *= t.scaleY;
      return [cx + dx + dy * t.skewX, cy + dy + dx * t.skewY];
    });
  }

  function getTransformedPoly(): number[][] | null {
    if (!targetPolygon || targetPolygon.length < 3) return null;
    return applyTransform(targetPolygon, transform);
  }

  // Version that reads from refs (for drawFrame inside WebSocket closure)
  function getTransformedPolyFromRefs(): number[][] | null {
    const tp = pendingPolygonRef.current || (selectedZoneIdRef.current ? originalPolygonRef.current : null);
    if (!tp || tp.length < 3) return null;
    return applyTransform(tp, transformRef.current);
  }

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

  function dist(ax: number, ay: number, bx: number, by: number) {
    return Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2);
  }

  function hitTestGizmo(nx: number, ny: number): GizmoHandle {
    const poly = getTransformedPoly();
    if (!poly) return "none";
    const normPoly = poly.map(([pan, tilt]) => angleToPixelNorm(pan, tilt));
    const [cx, cy] = getCentroid(normPoly);

    const arrowScale = 0.08;
    const HIT_R = 0.025;

    const zEnd = { x: cx + Z_UNIT.x * arrowScale, y: cy + Z_UNIT.y * arrowScale };
    const xEnd = { x: cx + arrowScale, y: cy };
    const yEnd = { x: cx, y: cy + arrowScale };

    if (dist(nx, ny, zEnd.x, zEnd.y) < HIT_R) return "z-axis";
    if (dist(nx, ny, xEnd.x, xEnd.y) < HIT_R) return "x-axis";
    if (dist(nx, ny, yEnd.x, yEnd.y) < HIT_R) return "y-axis";

    // Skew handles at edge midpoints
    for (let i = 0; i < normPoly.length && i < 4; i++) {
      const ni = (i + 1) % normPoly.length;
      const mx = (normPoly[i][0] + normPoly[ni][0]) / 2;
      const my = (normPoly[i][1] + normPoly[ni][1]) / 2;
      if (dist(nx, ny, mx, my) < HIT_R * 0.8) {
        if (i === 0) return "skew-top";
        if (i === 1) return "skew-right";
        if (i === 2) return "skew-bottom";
        return "skew-left";
      }
    }

    if (pointInPoly(nx, ny, normPoly)) return "move";
    return "none";
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
        setWarningRemaining(data.warning_remaining ?? 0);
        setDirectionDelta(data.direction_delta ?? null);
        setOccludedCats((data.occluded_cats || []).map((c: any) => ({ id: c.id, predicted: c.predicted, occluded_by: c.occluded_by })));
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

    // Gizmo interaction
    if (showGizmo && onTransformChange && e.button === 0) {
      const pos = getNormPos(e);
      if (!pos) return;
      const handle = hitTestGizmo(pos.x, pos.y);
      if (handle !== "none") {
        e.preventDefault();
        setActiveHandle(handle);
        setDragStartNorm(pos);
        setDragStartTransform({ ...transform });
        return;
      }
      if (pendingPolygon) return; // don't deselect while configuring new zone
    }

    // Click zone to select (in edit mode)
    if (!drawMode && !selectedZoneId && onSelectZone && !pendingPolygon && e.button === 0) {
      const pos = getNormPos(e);
      if (!pos) return;
      for (const zone of zones) {
        if (!zone.enabled) continue;
        const normPoly = zone.polygon.map(([pan, tilt]) => angleToPixelNorm(pan, tilt));
        if (pointInPoly(pos.x, pos.y, normPoly)) {
          onSelectZone(zone.id);
          return;
        }
      }
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

    // Gizmo drag
    if (activeHandle !== "none" && dragStartNorm && dragStartTransform && onTransformChange) {
      const pos = getNormPos(e);
      if (!pos) return;
      const dx = pos.x - dragStartNorm.x;
      const dy = pos.y - dragStartNorm.y;
      const t = { ...dragStartTransform };
      const SCALE_SENS = 4.0;
      const SKEW_SENS = 3.0;
      const HEIGHT_SENS = 600;

      switch (activeHandle) {
        case "x-axis": t.scaleX = Math.max(0.1, dragStartTransform.scaleX + dx * SCALE_SENS); break;
        case "y-axis": t.scaleY = Math.max(0.1, dragStartTransform.scaleY + dy * SCALE_SENS); break;
        case "z-axis": {
          const proj = dx * Z_UNIT.x + dy * Z_UNIT.y;
          t.height = Math.max(0, dragStartTransform.height - proj * HEIGHT_SENS);
          break;
        }
        case "move": {
          if (targetPolygon && onZonePolygonUpdate && selectedZoneId) {
            const dPan = dx * FOV_H;
            const dTilt = dy * FOV_V;
            const newPoly = targetPolygon.map(([pan, tilt]) => [pan + dPan, tilt + dTilt]);
            const zone = zones.find((z) => z.id === selectedZoneId);
            if (zone) zone.polygon = newPoly;
          }
          setDragStartNorm(pos);
          return;
        }
        case "skew-top": case "skew-bottom": t.skewX = dragStartTransform.skewX + dx * SKEW_SENS; break;
        case "skew-left": case "skew-right": t.skewY = dragStartTransform.skewY + dy * SKEW_SENS; break;
        case "slant":
          t.slantX = Math.max(-1, Math.min(1, dragStartTransform.slantX + dx * 3));
          t.slantY = Math.max(-1, Math.min(1, dragStartTransform.slantY + dy * 3));
          break;
      }
      onTransformChange(t);
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

    // Commit gizmo drag
    if (activeHandle !== "none") {
      if (activeHandle === "move" && selectedZoneId && onZonePolygonUpdate) {
        const zone = zones.find((z) => z.id === selectedZoneId);
        if (zone) onZonePolygonUpdate(selectedZoneId, zone.polygon);
      }
      setActiveHandle("none"); setDragStartNorm(null); setDragStartTransform(null);
      return;
    }

    // Freehand completion (line mode completes on click near first vertex)
    if (!drawMode || !isDrawing || drawTool === "line") return;
    setIsDrawing(false);
    if (drawPoints.length > 2 && onDrawComplete) {
      const closed = [...drawPoints, drawPoints[0]];
      const anglePolygon = closed.map(([px, py]) => pixelNormToAngle(px, py));
      onDrawComplete(anglePolygon);
      setDrawPoints([]);
    }
  }

  // ── Draw gizmo on canvas ────────────────────────
  function drawGizmoOnCanvas(
    ctx: CanvasRenderingContext2D,
    basePixels: { x: number; y: number }[],
    w: number, h: number,
  ) {
    let cx = 0, cy = 0;
    basePixels.forEach((p) => { cx += p.x; cy += p.y; });
    cx /= basePixels.length; cy /= basePixels.length;
    const len = ARROW_LEN;

    function drawArrow(fx: number, fy: number, tx: number, ty: number, color: string, label: string, active: boolean) {
      const ddx = tx - fx, ddy = ty - fy;
      const mag = Math.sqrt(ddx * ddx + ddy * ddy);
      const ux = ddx / mag, uy = ddy / mag;
      ctx.strokeStyle = color; ctx.lineWidth = active ? 3 : 2; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(tx, ty); ctx.stroke();
      // Head
      ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(tx, ty);
      ctx.lineTo(tx - ux * 8 + uy * 4, ty - uy * 8 - ux * 4);
      ctx.lineTo(tx - ux * 8 - uy * 4, ty - uy * 8 + ux * 4);
      ctx.closePath(); ctx.fill();
      // Handle
      const hs = active ? HANDLE_SIZE + 2 : HANDLE_SIZE;
      ctx.fillStyle = active ? "white" : color;
      ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.lineWidth = 1;
      ctx.fillRect(tx - hs / 2, ty - hs / 2, hs, hs);
      ctx.strokeRect(tx - hs / 2, ty - hs / 2, hs, hs);
      // Label
      ctx.fillStyle = color; ctx.font = "bold 9px 'IBM Plex Mono', monospace";
      ctx.fillText(label, tx + 6, ty + 3);
    }

    // Center dot
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.4)"; ctx.lineWidth = 1; ctx.stroke();

    const ah = activeHandleRef.current;
    drawArrow(cx, cy, cx + len, cy, "rgba(239,68,68,0.9)", "W", ah === "x-axis");
    drawArrow(cx, cy, cx, cy + len, "rgba(34,197,94,0.9)", "L", ah === "y-axis");
    drawArrow(cx, cy, cx + Z_UNIT.x * len, cy + Z_UNIT.y * len, "rgba(59,130,246,0.9)", "H", ah === "z-axis");

    // Skew handles
    for (let i = 0; i < basePixels.length && i < 4; i++) {
      const ni = (i + 1) % basePixels.length;
      const mx = (basePixels[i].x + basePixels[ni].x) / 2;
      const my = (basePixels[i].y + basePixels[ni].y) / 2;
      const isActive = (i === 0 && ah === "skew-top") || (i === 1 && ah === "skew-right")
        || (i === 2 && ah === "skew-bottom") || (i === 3 && ah === "skew-left");
      const s = isActive ? SKEW_HANDLE_SIZE + 2 : SKEW_HANDLE_SIZE;
      ctx.fillStyle = isActive ? "white" : "rgba(245,158,11,0.8)";
      ctx.strokeStyle = "rgba(255,255,255,0.7)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(mx, my - s); ctx.lineTo(mx + s, my);
      ctx.lineTo(mx, my + s); ctx.lineTo(mx - s, my); ctx.closePath();
      ctx.fill(); ctx.stroke();
    }

    // Readout
    const t = transformRef.current;
    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.font = "500 9px 'IBM Plex Mono', monospace";
    ctx.fillText(`W:${t.scaleX.toFixed(1)}x  L:${t.scaleY.toFixed(1)}x  H:${t.height.toFixed(0)}cm`, cx - 60, cy - 14);
  }

  // ── Draw 3D prism ───────────────────────────────
  function drawPrism(
    ctx: CanvasRenderingContext2D,
    basePx: { x: number; y: number }[],
    hMin: number, hMax: number, w: number, h: number,
    slantX = 0, slantY = 0,
  ) {
    if (hMax <= hMin) return basePx;
    const heightNorm = (hMax - hMin) / 300;
    const offY = -heightNorm * h * 0.12;
    const offX = heightNorm * w * 0.015;

    let bcx = 0, bcy = 0;
    basePx.forEach((p) => { bcx += p.x; bcy += p.y; });
    bcx /= basePx.length; bcy /= basePx.length;

    const topPx = basePx.map((p) => {
      const relX = (p.x - bcx) / (w * 0.1 || 1);
      const relY = (p.y - bcy) / (h * 0.1 || 1);
      return { x: p.x + offX + slantX * relX * heightNorm * w * 0.02, y: p.y + offY + slantY * relY * heightNorm * h * 0.02 };
    });

    // Walls
    for (let si = 0; si < basePx.length; si++) {
      const ni = (si + 1) % basePx.length;
      ctx.beginPath(); ctx.moveTo(basePx[si].x, basePx[si].y);
      ctx.lineTo(topPx[si].x, topPx[si].y); ctx.lineTo(topPx[ni].x, topPx[ni].y);
      ctx.lineTo(basePx[ni].x, basePx[ni].y); ctx.closePath();
      ctx.fillStyle = "rgba(168,85,247,0.06)"; ctx.fill();
      ctx.strokeStyle = "rgba(245,158,11,0.3)"; ctx.lineWidth = 1; ctx.setLineDash([4, 3]); ctx.stroke();
    }
    // Base
    ctx.beginPath(); basePx.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); }); ctx.closePath();
    ctx.fillStyle = "rgba(245,158,11,0.12)"; ctx.fill();
    ctx.strokeStyle = "rgba(245,158,11,0.5)"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]); ctx.stroke();
    // Top
    ctx.beginPath(); topPx.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); }); ctx.closePath();
    ctx.fillStyle = "rgba(168,85,247,0.15)"; ctx.fill();
    ctx.strokeStyle = "rgba(168,85,247,0.5)"; ctx.lineWidth = 1.5; ctx.setLineDash([]); ctx.stroke();
    ctx.setLineDash([]);
    return topPx;
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
    const currentTransform = transformRef.current;
    const currentRightDrag = rightDragZoneIdRef.current;

    // ── Draw zones ──
    for (const zone of currentZones) {
      if (!zone.enabled) continue;
      if (zone.id === currentSelectedId) continue; // drawn separately with gizmo

      const projected = zone.polygon.map(([pan, tilt]) => ({
        px: 0.5 + (pan - sPan) / FOV_H, py: 0.5 + (tilt - sTilt) / FOV_V,
      }));
      if (!projected.some((p) => p.px >= -0.5 && p.px <= 1.5 && p.py >= -0.5 && p.py <= 1.5)) continue;

      const isDragging = zone.id === currentRightDrag;
      const is3d = zone.mode && zone.mode !== "2d";
      const basePx = projected.map((p) => ({ x: p.px * w, y: p.py * h }));

      if (is3d && zone.height_max > zone.height_min) {
        drawPrism(ctx, basePx, zone.height_min, zone.height_max, w, h);
        if (isDragging) {
          ctx.strokeStyle = "rgba(34,211,238,0.8)"; ctx.lineWidth = 2; ctx.setLineDash([]);
          ctx.beginPath(); basePx.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); }); ctx.closePath(); ctx.stroke();
        }
        if (basePx.length > 0) {
          ctx.fillStyle = isDragging ? "rgba(34,211,238,0.9)" : "rgba(168,85,247,0.9)";
          ctx.font = "500 10px 'IBM Plex Mono', monospace";
          ctx.fillText(`${zone.name} [${zone.height_min}-${zone.height_max}cm]`, basePx[0].x, basePx[0].y - 20);
        }
      } else {
        ctx.strokeStyle = isDragging ? "rgba(34,211,238,0.9)" : "rgba(239,68,68,0.7)";
        ctx.lineWidth = isDragging ? 2.5 : 1.5;
        ctx.setLineDash(isDragging ? [] : [6, 4]);
        ctx.beginPath(); basePx.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); }); ctx.closePath();
        ctx.stroke();
        ctx.fillStyle = isDragging ? "rgba(34,211,238,0.1)" : "rgba(239,68,68,0.06)";
        ctx.fill(); ctx.setLineDash([]);
        if (basePx.length > 0) {
          ctx.fillStyle = isDragging ? "rgba(34,211,238,0.9)" : "rgba(239,68,68,0.9)";
          ctx.font = "500 11px 'IBM Plex Mono', monospace";
          ctx.fillText(zone.name, basePx[0].x, basePx[0].y - 6);
        }
      }
    }

    // ── Draw gizmo target polygon (selected zone or pending) ──
    const transformedPoly = getTransformedPolyFromRefs();
    if (transformedPoly && transformedPoly.length >= 3) {
      const projected = transformedPoly.map(([pan, tilt]) => ({
        x: (0.5 + (pan - sPan) / FOV_H) * w,
        y: (0.5 + (tilt - sTilt) / FOV_V) * h,
      }));

      if (currentTransform.height > 0) {
        const topPx = drawPrism(ctx, projected, 0, currentTransform.height, w, h, currentTransform.slantX, currentTransform.slantY);
        if (topPx.length > 0) {
          ctx.fillStyle = "rgba(34,211,238,0.9)"; ctx.font = "600 10px 'IBM Plex Mono', monospace";
          const label = currentSelectedId ? currentZones.find((z) => z.id === currentSelectedId)?.name || "ZONE" : "NEW ZONE";
          ctx.fillText(`${label} [${currentTransform.height.toFixed(0)}cm]`, topPx[0].x, topPx[0].y - 4);
        }
      } else {
        ctx.strokeStyle = "rgba(34,211,238,0.9)"; ctx.lineWidth = 2; ctx.setLineDash([]);
        ctx.beginPath(); projected.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); }); ctx.closePath();
        ctx.stroke(); ctx.fillStyle = "rgba(34,211,238,0.12)"; ctx.fill();
        if (projected.length > 0) {
          ctx.fillStyle = "rgba(34,211,238,0.9)"; ctx.font = "600 10px 'IBM Plex Mono', monospace";
          const label = currentSelectedId ? currentZones.find((z) => z.id === currentSelectedId)?.name || "ZONE" : "NEW ZONE";
          ctx.fillText(label, projected[0].x, projected[0].y - 4);
        }
      }
      // Gizmo
      drawGizmoOnCanvas(ctx, projected, w, h);
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
  if (rightDragZoneId || activeHandle !== "none") cursor = "grabbing";
  else if (drawMode) cursor = (isDrawing && drawTool === "freehand") ? "none" : "crosshair";
  else if (showGizmo) cursor = "default";

  const drawPathData = drawMode && drawPoints.length > 1
    ? `M ${drawPoints.map(([x, y]) => `${x * 100},${y * 100}`).join(" L ")}${!isDrawing && drawPoints.length > 2 ? " Z" : ""}`
    : "";

  return (
    <div
      ref={overlayRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        if (isDrawing) handleMouseUp();
        if (activeHandle !== "none") handleMouseUp();
        if (rightDragZoneId) handleMouseUp();
      }}
      onContextMenu={(e) => e.preventDefault()}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        background: "var(--bg-deep)",
        cursor,
        userSelect: (drawMode || showGizmo || rightDragZoneId) ? "none" : "auto",
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block", objectFit: "contain", pointerEvents: "none" }}
      />

      {/* Drawing overlay SVG */}
      {drawMode && drawPathData && (
        <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
          viewBox="0 0 100 100" preserveAspectRatio="none">
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

      {/* Gizmo active banner */}
      {showGizmo && !drawMode && (
        <div style={{
          position: "absolute", top: 10, left: "50%", transform: "translateX(-50%)",
          padding: "4px 14px", background: "rgba(34,211,238,0.85)", color: "var(--bg-deep)",
          borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)", fontSize: 11,
          fontWeight: 600, pointerEvents: "none", letterSpacing: "0.04em",
        }}>
          TRANSFORM — drag W/L/H arrows
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
        const [x1, y1, x2, y2] = det.bbox;
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
      {!drawMode && !showGizmo && (
        <div style={{ position: "absolute", top: 10, left: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <StateIndicator state={state} warningRemaining={warningRemaining} />
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
          occludedCats={occludedCats}
          servoPan={servoPan}
          servoTilt={servoTilt}
        />
      )}

      {/* Occluded cat indicators — shifted above radar */}
      {!drawMode && occludedCats.length > 0 && (
        <div style={{ position: "absolute", bottom: 140, left: 10, display: "flex", flexDirection: "column", gap: 3 }}>
          {occludedCats.map((c) => (
            <div key={c.id} style={{
              padding: "2px 8px", background: "rgba(245,158,11,0.15)", border: "1px dashed var(--amber-dim)",
              borderRadius: 4, fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--amber)", animation: "pulse 2s infinite",
            }}>
              {c.id} — behind {c.occluded_by}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
