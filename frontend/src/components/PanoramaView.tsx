import { useEffect, useRef, useState, useCallback } from "react";
import type { Zone, Furniture, Detection, ZoneTransform } from "../types";
import { DEFAULT_TRANSFORM } from "../types";

interface PanoramaViewProps {
  panoramaBase64: string | null;
  zones: Zone[];
  furniture?: Furniture[];
  detections: Detection[];
  servoPan: number;
  servoTilt: number;
  sweepPanMin: number;
  sweepPanMax: number;
  sweepTiltMin: number;
  sweepTiltMax: number;
  fovH: number;
  fovV: number;
  onClickAngle?: (pan: number, tilt: number) => void;
  drawMode?: boolean;
  onDrawComplete?: (anglePolygon: number[][]) => void;
  pendingPolygon?: number[][] | null;
  selectedZoneId?: string | null;
  onSelectZone?: (id: string | null) => void;
  onZonePolygonUpdate?: (zoneId: string, newPolygon: number[][]) => void;
  transform?: ZoneTransform;
  onTransformChange?: (t: ZoneTransform) => void;
  originalPolygon?: number[][] | null;
  expanded?: boolean;
  onToggleExpand?: () => void;
}

// Axis handle identifiers
type GizmoHandle =
  | "none"
  | "move"       // drag inside polygon
  | "x-axis"     // red: width scale
  | "y-axis"     // green: length scale
  | "skew-top"   // skew from top edge midpoint
  | "skew-right" // skew from right edge midpoint
  | "skew-bottom"
  | "skew-left";

// Gizmo arrow length in pixels
const ARROW_LEN = 50;
const HANDLE_SIZE = 7;
const SKEW_HANDLE_SIZE = 5;

export default function PanoramaView({
  panoramaBase64, zones, furniture = [], detections,
  servoPan, servoTilt,
  sweepPanMin, sweepPanMax, sweepTiltMin, sweepTiltMax,
  fovH, fovV,
  onClickAngle, drawMode, onDrawComplete,
  pendingPolygon,
  selectedZoneId, onSelectZone, onZonePolygonUpdate,
  transform = { ...DEFAULT_TRANSFORM },
  onTransformChange,
  originalPolygon,
  expanded,
  onToggleExpand,
}: PanoramaViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [rawPoints, setRawPoints] = useState<number[][]>([]);
  const [isDrawing, setIsDrawing] = useState(false);

  // Drawing tool: "freehand" (click+drag) or "line" (click-to-place vertices)
  const [drawTool, setDrawTool] = useState<"freehand" | "line">("freehand");
  const [linePoints, setLinePoints] = useState<number[][]>([]); // vertices for line mode
  const [linePreview, setLinePreview] = useState<number[] | null>(null); // cursor pos for preview edge

  // Gizmo drag state
  const [activeHandle, setActiveHandle] = useState<GizmoHandle>("none");
  const [dragStartPos, setDragStartPos] = useState<{ x: number; y: number } | null>(null);
  const [dragStartTransform, setDragStartTransform] = useState<ZoneTransform | null>(null);

  // Right-click drag-to-rearrange (works on any zone, any mode)
  const [rightDragZoneId, setRightDragZoneId] = useState<string | null>(null);
  const [rightDragStart, setRightDragStart] = useState<{ x: number; y: number } | null>(null);
  const [rightDragOrigPoly, setRightDragOrigPoly] = useState<number[][] | null>(null);

  // Cached panorama Image to avoid creating new Image objects on every mouse move
  const panoramaImgRef = useRef<HTMLImageElement | null>(null);

  // Which polygon is being transformed: pending or selected zone
  const targetPolygon = pendingPolygon || (selectedZoneId ? originalPolygon : null);
  const showGizmo = !!targetPolygon && targetPolygon.length >= 3;

  // Decode panorama image only when the base64 data changes
  useEffect(() => {
    if (!panoramaBase64) return;
    const img = new Image();
    img.onload = () => { panoramaImgRef.current = img; drawPanorama(img); };
    img.src = `data:image/jpeg;base64,${panoramaBase64}`;
  }, [panoramaBase64]);

  // Redraw using cached image when anything else changes (no re-decode needed)
  useEffect(() => {
    if (panoramaImgRef.current) drawPanorama(panoramaImgRef.current);
  }, [zones, detections, servoPan, servoTilt,
      pendingPolygon,
      selectedZoneId, transform, originalPolygon, linePoints, linePreview,
      rawPoints, activeHandle, isDrawing]);

  useEffect(() => {
    if (!drawMode) { setRawPoints([]); setIsDrawing(false); setLinePoints([]); setLinePreview(null); }
  }, [drawMode]);

  // ── Coordinate helpers ──────────────────────────
  function angleToPx(pan: number, tilt: number, w: number, h: number) {
    return {
      x: ((pan - sweepPanMin) / (sweepPanMax - sweepPanMin)) * w,
      y: ((tilt - sweepTiltMin) / (sweepTiltMax - sweepTiltMin)) * h,
    };
  }

  function angleToNorm(pan: number, tilt: number): [number, number] {
    return [
      (pan - sweepPanMin) / (sweepPanMax - sweepPanMin),
      (tilt - sweepTiltMin) / (sweepTiltMax - sweepTiltMin),
    ];
  }

  function normToAngle(nx: number, ny: number): [number, number] {
    return [
      sweepPanMin + nx * (sweepPanMax - sweepPanMin),
      sweepTiltMin + ny * (sweepTiltMax - sweepTiltMin),
    ];
  }

  const getNormPos = useCallback((e: React.MouseEvent) => {
    if (!containerRef.current) return null;
    const rect = containerRef.current.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height)),
    };
  }, []);

  // ── Transform math ──────────────────────────────
  function getCentroid(poly: number[][]): [number, number] {
    let sx = 0, sy = 0;
    for (const [x, y] of poly) { sx += x; sy += y; }
    return [sx / poly.length, sy / poly.length];
  }

  function applyTransform(poly: number[][], t: ZoneTransform): number[][] {
    const [cx, cy] = getCentroid(poly);
    return poly.map(([x, y]) => {
      let dx = x - cx;
      let dy = y - cy;
      // Scale
      dx *= t.scaleX;
      dy *= t.scaleY;
      // Skew
      const skewedX = dx + dy * t.skewX;
      const skewedY = dy + dx * t.skewY;
      return [cx + skewedX, cy + skewedY];
    });
  }

  // Get the transformed polygon for rendering
  function getTransformedPoly(): number[][] | null {
    if (!targetPolygon || targetPolygon.length < 3) return null;
    return applyTransform(targetPolygon, transform);
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

  function distToPoint(ax: number, ay: number, bx: number, by: number): number {
    return Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2);
  }

  // Get gizmo geometry in normalized coords
  function getGizmoGeometry() {
    const poly = getTransformedPoly();
    if (!poly) return null;
    const normPoly = poly.map(([pan, tilt]) => angleToNorm(pan, tilt));
    const [cx, cy] = getCentroid(normPoly);

    // Arrow endpoints (in norm coords, scaled by container aspect ~3:1 so use pixel scaling)
    // We'll compute in a uniform space; actual pixel positions computed at draw time
    const arrowScale = 0.06; // normalized length of arrows

    return {
      center: { x: cx, y: cy },
      xEnd: { x: cx + arrowScale, y: cy },                         // red: right
      yEnd: { x: cx, y: cy + arrowScale },                         // green: down
      normPoly,
      // Edge midpoints for skew handles
      edgeMids: normPoly.map(([x, y], i) => {
        const ni = (i + 1) % normPoly.length;
        return { x: (x + normPoly[ni][0]) / 2, y: (y + normPoly[ni][1]) / 2 };
      }),
    };
  }

  function hitTestGizmo(nx: number, ny: number): GizmoHandle {
    const g = getGizmoGeometry();
    if (!g) return "none";
    const HIT_R = 0.02;

    // X axis handle
    if (distToPoint(nx, ny, g.xEnd.x, g.xEnd.y) < HIT_R) return "x-axis";
    // Y axis handle
    if (distToPoint(nx, ny, g.yEnd.x, g.yEnd.y) < HIT_R) return "y-axis";

    // Skew handles at edge midpoints
    for (let i = 0; i < g.edgeMids.length; i++) {
      if (distToPoint(nx, ny, g.edgeMids[i].x, g.edgeMids[i].y) < HIT_R * 0.8) {
        // Map edge index to skew direction
        if (i === 0) return "skew-top";
        if (i === 1) return "skew-right";
        if (i === 2) return "skew-bottom";
        return "skew-left";
      }
    }

    // Move: click inside polygon
    if (pointInPoly(nx, ny, g.normPoly)) return "move";

    return "none";
  }

  // ── Mouse handlers ──────────────────────────────
  function handlePointerDown(e: React.MouseEvent) {
    // Right-click: drag any zone to reposition
    if (e.button === 2 && onZonePolygonUpdate) {
      e.preventDefault();
      const pos = getNormPos(e);
      if (!pos) return;
      // Hit test all zones
      for (const zone of zones) {
        if (!zone.enabled) continue;
        const normPoly = zone.polygon.map(([pan, tilt]) => angleToNorm(pan, tilt));
        if (pointInPoly(pos.x, pos.y, normPoly)) {
          setRightDragZoneId(zone.id);
          setRightDragStart(pos);
          setRightDragOrigPoly([...zone.polygon]);
          return;
        }
      }
      return;
    }

    // If gizmo is showing, check gizmo hit first
    if (showGizmo && onTransformChange) {
      const pos = getNormPos(e);
      if (!pos) return;
      const handle = hitTestGizmo(pos.x, pos.y);
      if (handle !== "none") {
        e.preventDefault();
        setActiveHandle(handle);
        setDragStartPos(pos);
        setDragStartTransform({ ...transform });
        return;
      }
      // Click outside gizmo while pending polygon → do nothing (they need to use panel to clear)
      if (pendingPolygon) return;
    }

    // Click on zone to select it
    if (!drawMode && !selectedZoneId && onSelectZone && !pendingPolygon) {
      const pos = getNormPos(e);
      if (!pos) return;
      for (const zone of zones) {
        if (!zone.enabled) continue;
        const normPoly = zone.polygon.map(([pan, tilt]) => angleToNorm(pan, tilt));
        if (pointInPoly(pos.x, pos.y, normPoly)) {
          onSelectZone(zone.id);
          return;
        }
      }
    }

    // Normal click-to-aim
    if (!drawMode && !pendingPolygon) {
      if (!canvasRef.current || !onClickAngle) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const px = (e.clientX - rect.left) / rect.width;
      const py = (e.clientY - rect.top) / rect.height;
      const [pan, tilt] = normToAngle(px, py);
      onClickAngle(pan, tilt);
      return;
    }

    // Drawing
    if (drawMode) {
      e.preventDefault();
      const pos = getNormPos(e);
      if (!pos) return;

      if (drawTool === "line") {
        // Line mode: each click adds a vertex
        if (linePoints.length >= 3) {
          // Check if clicking near the first vertex to close
          const [fx, fy] = linePoints[0];
          if (Math.sqrt((pos.x - fx) ** 2 + (pos.y - fy) ** 2) < 0.025) {
            // Close the polygon
            const closed = [...linePoints, linePoints[0]];
            const anglePolygon = closed.map(([nx, ny]) => normToAngle(nx, ny));
            if (onDrawComplete) onDrawComplete(anglePolygon);
            setLinePoints([]);
            setLinePreview(null);
            return;
          }
        }
        setLinePoints((prev) => [...prev, [pos.x, pos.y]]);
      } else {
        // Freehand mode: click+drag
        setIsDrawing(true);
        setRawPoints([[pos.x, pos.y]]);
      }
    }
  }

  function handlePointerMove(e: React.MouseEvent) {
    // Middle-click drag: translate zone polygon
    if (rightDragZoneId && rightDragStart && rightDragOrigPoly) {
      const pos = getNormPos(e);
      if (!pos) return;
      const dx = pos.x - rightDragStart.x;
      const dy = pos.y - rightDragStart.y;
      const dPan = dx * (sweepPanMax - sweepPanMin);
      const dTilt = dy * (sweepTiltMax - sweepTiltMin);
      const newPoly = rightDragOrigPoly.map(([pan, tilt]) => [pan + dPan, tilt + dTilt]);
      // Mutate zone temporarily for live rendering
      const zone = zones.find((z) => z.id === rightDragZoneId);
      if (zone) zone.polygon = newPoly;
      // Trigger canvas redraw using cached image (avoids memory flood)
      if (canvasRef.current && panoramaImgRef.current) {
        drawPanorama(panoramaImgRef.current);
      }
      return;
    }

    // Gizmo drag
    if (activeHandle !== "none" && dragStartPos && dragStartTransform && onTransformChange) {
      const pos = getNormPos(e);
      if (!pos) return;
      const dx = pos.x - dragStartPos.x;
      const dy = pos.y - dragStartPos.y;
      const t = { ...dragStartTransform };

      // Scale factor: how much mouse movement maps to transform change
      const SCALE_SENS = 4.0;
      const SKEW_SENS = 3.0;

      switch (activeHandle) {
        case "x-axis":
          t.scaleX = Math.max(0.1, dragStartTransform.scaleX + dx * SCALE_SENS);
          break;
        case "y-axis":
          t.scaleY = Math.max(0.1, dragStartTransform.scaleY + dy * SCALE_SENS);
          break;
        case "move": {
          if (targetPolygon && onZonePolygonUpdate && selectedZoneId) {
            const dPan = dx * (sweepPanMax - sweepPanMin);
            const dTilt = dy * (sweepTiltMax - sweepTiltMin);
            const newPoly = targetPolygon.map(([pan, tilt]) => [pan + dPan, tilt + dTilt]);
            const zone = zones.find((z) => z.id === selectedZoneId);
            if (zone) zone.polygon = newPoly;
          }
          // Trigger redraw using cached image (avoids memory flood)
          if (canvasRef.current && panoramaImgRef.current) {
            drawPanorama(panoramaImgRef.current);
          }
          setDragStartPos(pos);
          return;
        }
        case "skew-top":
        case "skew-bottom":
          t.skewX = dragStartTransform.skewX + dx * SKEW_SENS;
          break;
        case "skew-left":
        case "skew-right":
          t.skewY = dragStartTransform.skewY + dy * SKEW_SENS;
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
      // Update preview line from last placed vertex to cursor
      if (linePoints.length > 0) {
        setLinePreview([pos.x, pos.y]);
      }
    } else {
      // Freehand
      if (!isDrawing) return;
      setRawPoints((prev) => [...prev, [pos.x, pos.y]]);
    }
  }

  function handlePointerUp() {
    // Commit middle-click drag
    if (rightDragZoneId && rightDragOrigPoly && onZonePolygonUpdate) {
      const zone = zones.find((z) => z.id === rightDragZoneId);
      if (zone) onZonePolygonUpdate(rightDragZoneId, zone.polygon);
      setRightDragZoneId(null);
      setRightDragStart(null);
      setRightDragOrigPoly(null);
      return;
    }

    if (activeHandle !== "none") {
      // Commit move for selected zones
      if (activeHandle === "move" && selectedZoneId && onZonePolygonUpdate) {
        const zone = zones.find((z) => z.id === selectedZoneId);
        if (zone) onZonePolygonUpdate(selectedZoneId, zone.polygon);
      }
      setActiveHandle("none");
      setDragStartPos(null);
      setDragStartTransform(null);
      return;
    }

    // Freehand completion (line mode completes on click near first vertex)
    if (!drawMode || !isDrawing || drawTool === "line") return;
    setIsDrawing(false);
    if (rawPoints.length > 2 && onDrawComplete) {
      const closed = [...rawPoints, rawPoints[0]];
      const anglePolygon = closed.map(([nx, ny]) => normToAngle(nx, ny));
      onDrawComplete(anglePolygon);
      setRawPoints([]);
    }
  }

  // ── Draw the transform gizmo ────────────────────
  function drawGizmo(ctx: CanvasRenderingContext2D, basePixels: { x: number; y: number }[], _w: number, _h: number) {
    // Centroid
    let cx = 0, cy = 0;
    basePixels.forEach((p) => { cx += p.x; cy += p.y; });
    cx /= basePixels.length; cy /= basePixels.length;

    const len = ARROW_LEN;

    // Helper: draw arrow line with triangle tip
    function drawArrow(
      fromX: number, fromY: number,
      toX: number, toY: number,
      color: string, label: string, active: boolean,
    ) {
      const dx = toX - fromX, dy = toY - fromY;
      const mag = Math.sqrt(dx * dx + dy * dy);
      const ux = dx / mag, uy = dy / mag;

      // Shaft
      ctx.strokeStyle = color;
      ctx.lineWidth = active ? 3 : 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(fromX, fromY);
      ctx.lineTo(toX, toY);
      ctx.stroke();

      // Triangle head
      const headLen = 8;
      const headW = 4;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(toX, toY);
      ctx.lineTo(toX - ux * headLen + uy * headW, toY - uy * headLen - ux * headW);
      ctx.lineTo(toX - ux * headLen - uy * headW, toY - uy * headLen + ux * headW);
      ctx.closePath();
      ctx.fill();

      // Handle square
      ctx.fillStyle = active ? "white" : color;
      ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.lineWidth = 1;
      const hs = active ? HANDLE_SIZE + 2 : HANDLE_SIZE;
      ctx.fillRect(toX - hs / 2, toY - hs / 2, hs, hs);
      ctx.strokeRect(toX - hs / 2, toY - hs / 2, hs, hs);

      // Label
      ctx.fillStyle = color;
      ctx.font = "bold 9px 'IBM Plex Mono', monospace";
      ctx.fillText(label, toX + 6, toY + 3);
    }

    // Center dot
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.4)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // X axis (red) — horizontal right
    drawArrow(cx, cy, cx + len, cy,
      "rgba(239, 68, 68, 0.9)", "W", activeHandle === "x-axis");

    // Y axis (green) — vertical down
    drawArrow(cx, cy, cx, cy + len,
      "rgba(34, 197, 94, 0.9)", "L", activeHandle === "y-axis");

    // Skew handles at edge midpoints (diamonds)
    const normPoly = basePixels;
    for (let i = 0; i < normPoly.length && i < 4; i++) {
      const ni = (i + 1) % normPoly.length;
      const mx = (normPoly[i].x + normPoly[ni].x) / 2;
      const my = (normPoly[i].y + normPoly[ni].y) / 2;
      const isActive = (
        (i === 0 && activeHandle === "skew-top") ||
        (i === 1 && activeHandle === "skew-right") ||
        (i === 2 && activeHandle === "skew-bottom") ||
        (i === 3 && activeHandle === "skew-left")
      );
      ctx.fillStyle = isActive ? "white" : "rgba(245, 158, 11, 0.8)";
      ctx.strokeStyle = "rgba(255,255,255,0.7)";
      ctx.lineWidth = 1;
      // Diamond shape
      const s = isActive ? SKEW_HANDLE_SIZE + 2 : SKEW_HANDLE_SIZE;
      ctx.beginPath();
      ctx.moveTo(mx, my - s);
      ctx.lineTo(mx + s, my);
      ctx.lineTo(mx, my + s);
      ctx.lineTo(mx - s, my);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    // Dimension readout
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.font = "500 9px 'IBM Plex Mono', monospace";
    const readout = [
      `W:${transform.scaleX.toFixed(1)}x`,
      `L:${transform.scaleY.toFixed(1)}x`,
    ].join("  ");
    ctx.fillText(readout, cx - 60, cy - 14);

    if (transform.skewX !== 0 || transform.skewY !== 0) {
      ctx.fillStyle = "rgba(245, 158, 11, 0.7)";
      ctx.fillText(
        `Skew: ${transform.skewX.toFixed(2)} / ${transform.skewY.toFixed(2)}`,
        cx - 60, cy - 26
      );
    }
  }

  // ── Main draw function ──────────────────────────
  function drawPanorama(img: HTMLImageElement) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const w = canvas.width;
    const h = canvas.height;

    // Draw existing zones
    for (const zone of zones) {
      if (!zone.enabled) continue;
      // Skip the selected zone — we'll draw it transformed
      if (zone.id === selectedZoneId) continue;

      const isDragging = zone.id === rightDragZoneId;
      const basePixels = zone.polygon.map(([pan, tilt]) => angleToPx(pan, tilt, w, h));

      ctx.strokeStyle = isDragging ? "rgba(34, 211, 238, 0.9)" : "rgba(239, 68, 68, 0.6)";
      ctx.lineWidth = isDragging ? 2.5 : 1.5;
      ctx.setLineDash(isDragging ? [] : [5, 3]);
      ctx.beginPath();
      basePixels.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); });
      ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = isDragging ? "rgba(34, 211, 238, 0.12)" : "rgba(239, 68, 68, 0.08)";
      ctx.fill();
      ctx.setLineDash([]);
      if (basePixels.length > 0) {
        ctx.fillStyle = isDragging ? "rgba(34, 211, 238, 0.9)" : "rgba(239, 68, 68, 0.8)";
        ctx.font = "500 10px 'IBM Plex Mono', monospace";
        ctx.fillText(zone.name, basePixels[0].x, basePixels[0].y - 4);
      }
    }

    // Draw furniture outlines
    for (const item of furniture) {
      const pts = item.polygon.map(([pan, tilt]) => angleToPx(pan, tilt, w, h));
      if (pts.length < 2) continue;
      ctx.strokeStyle = "rgba(168, 162, 158, 0.6)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      pts.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); });
      ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = "rgba(168, 162, 158, 0.06)";
      ctx.fill();
      ctx.setLineDash([]);
      if (pts.length > 0) {
        ctx.fillStyle = "rgba(168, 162, 158, 0.8)";
        ctx.font = "500 10px 'IBM Plex Mono', monospace";
        ctx.fillText(item.name, pts[0].x, pts[0].y - 4);
      }
    }

    // Draw the active transformed polygon (pending or selected zone being edited)
    const transformedPoly = getTransformedPoly();
    if (transformedPoly && transformedPoly.length >= 3) {
      const basePx = transformedPoly.map(([pan, tilt]) => angleToPx(pan, tilt, w, h));

      // 2D outline with cyan highlight
      ctx.strokeStyle = "rgba(34, 211, 238, 0.9)";
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      basePx.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); });
      ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = "rgba(34, 211, 238, 0.12)";
      ctx.fill();
      if (basePx.length > 0) {
        ctx.fillStyle = "rgba(34, 211, 238, 0.9)";
        ctx.font = "600 10px 'IBM Plex Mono', monospace";
        const label = selectedZoneId
          ? zones.find((z) => z.id === selectedZoneId)?.name || "ZONE"
          : "NEW ZONE";
        ctx.fillText(label, basePx[0].x, basePx[0].y - 4);
      }

      // Draw the transform gizmo
      drawGizmo(ctx, basePx, w, h);
    }

    // Draw in-progress freehand
    if (rawPoints.length > 1) {
      ctx.strokeStyle = "rgba(245, 158, 11, 0.9)";
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      rawPoints.forEach(([nx, ny], i) => {
        const px = nx * w, py = ny * h;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.fillStyle = "rgba(245, 158, 11, 0.9)";
      ctx.beginPath();
      ctx.arc(rawPoints[0][0] * w, rawPoints[0][1] * h, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw in-progress straight-line polygon
    if (linePoints.length > 0) {
      ctx.strokeStyle = "rgba(245, 158, 11, 0.9)";
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      linePoints.forEach(([nx, ny], i) => {
        const px = nx * w, py = ny * h;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      // Preview line to cursor
      if (linePreview) {
        ctx.lineTo(linePreview[0] * w, linePreview[1] * h);
      }
      ctx.stroke();

      // Dashed line from cursor back to first point (close preview)
      if (linePreview && linePoints.length >= 2) {
        ctx.strokeStyle = "rgba(245, 158, 11, 0.4)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(linePreview[0] * w, linePreview[1] * h);
        ctx.lineTo(linePoints[0][0] * w, linePoints[0][1] * h);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Vertex dots
      linePoints.forEach(([nx, ny], i) => {
        ctx.fillStyle = i === 0 ? "rgba(34, 211, 238, 0.9)" : "rgba(245, 158, 11, 0.9)";
        ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(nx * w, ny * h, i === 0 ? 6 : 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });

      // "Click start to close" label near first vertex
      if (linePoints.length >= 3) {
        ctx.fillStyle = "rgba(34, 211, 238, 0.8)";
        ctx.font = "500 9px 'IBM Plex Mono', monospace";
        ctx.fillText("click to close", linePoints[0][0] * w + 10, linePoints[0][1] * h - 4);
      }
    }

    // Camera viewport
    const vpLeft = angleToPx(servoPan - fovH / 2, servoTilt - fovV / 2, w, h);
    const vpRight = angleToPx(servoPan + fovH / 2, servoTilt + fovV / 2, w, h);
    ctx.fillStyle = "rgba(245, 158, 11, 0.06)";
    ctx.fillRect(vpLeft.x, vpLeft.y, vpRight.x - vpLeft.x, vpRight.y - vpLeft.y);
    ctx.strokeStyle = "rgba(245, 158, 11, 0.6)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.strokeRect(vpLeft.x, vpLeft.y, vpRight.x - vpLeft.x, vpRight.y - vpLeft.y);
    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    ctx.fillRect(vpLeft.x, vpLeft.y - 13, 52, 13);
    ctx.fillStyle = "rgba(245, 158, 11, 0.9)";
    ctx.font = "600 9px 'IBM Plex Mono', monospace";
    ctx.fillText("LIVE", vpLeft.x + 4, vpLeft.y - 3);

    // Cat markers
    for (const det of detections) {
      const cx = (det.bbox[0] + det.bbox[2]) / 2;
      const cy = (det.bbox[1] + det.bbox[3]) / 2;
      const catPan = servoPan + (cx - 0.5) * fovH;
      const catTilt = servoTilt + (cy - 0.5) * fovV;
      const p = angleToPx(catPan, catTilt, w, h);
      ctx.strokeStyle = "rgba(239, 68, 68, 0.6)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = "rgba(239, 68, 68, 0.9)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ── Cursor logic ────────────────────────────────
  let cursor = "default";
  if (rightDragZoneId) cursor = "grabbing";
  else if (activeHandle !== "none") cursor = "grabbing";
  else if (showGizmo) cursor = "default";
  else if (drawMode) cursor = (isDrawing && drawTool === "freehand") ? "none" : "crosshair";
  else if (onClickAngle) cursor = "crosshair";

  const borderColor = showGizmo ? "var(--cyan)" : drawMode ? "var(--amber)" : "transparent";

  return (
    <div
      ref={containerRef}
      onMouseDown={handlePointerDown}
      onMouseMove={handlePointerMove}
      onMouseUp={handlePointerUp}
      onMouseLeave={() => {
        if (isDrawing) handlePointerUp();
        if (activeHandle !== "none") handlePointerUp();
        if (rightDragZoneId) handlePointerUp();
      }}
      onContextMenu={(e) => e.preventDefault()}
      style={{
        position: "relative",
        background: "var(--bg-deep)",
        cursor,
        userSelect: drawMode || showGizmo || rightDragZoneId ? "none" : "auto",
        border: `2px solid ${borderColor}`,
        boxSizing: "border-box",
      }}
    >
      <div style={{
        position: "absolute",
        top: 6, left: 10, zIndex: 1,
        display: "flex", alignItems: "center", gap: 5,
      }}>
        <span style={{
          padding: "2px 7px",
          background: showGizmo ? "rgba(34, 211, 238, 0.85)"
            : drawMode ? "rgba(245, 158, 11, 0.85)"
            : "rgba(0, 0, 0, 0.55)",
          backdropFilter: "blur(8px)",
          borderRadius: 3,
          fontFamily: "var(--font-mono)",
          fontSize: 9,
          fontWeight: 600,
          color: (showGizmo || drawMode) ? "var(--bg-deep)" : "var(--text-tertiary)",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
        }}>
          {showGizmo
            ? "Transform — drag axis arrows to scale, diamonds to skew"
            : drawMode
              ? (drawTool === "line" ? "Line Draw — click to place vertices, click start to close" : "Freehand Draw — click and drag")
              : "Panorama"}
        </span>

        {/* Draw tool toggle */}
        {drawMode && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setDrawTool((t) => t === "freehand" ? "line" : "freehand");
              setRawPoints([]); setIsDrawing(false); setLinePoints([]); setLinePreview(null);
            }}
            style={{
              padding: "2px 8px",
              background: "rgba(0, 0, 0, 0.6)",
              backdropFilter: "blur(8px)",
              border: "1px solid var(--amber-dim)",
              borderRadius: 3,
              cursor: "pointer",
              fontFamily: "var(--font-mono)",
              fontSize: 9,
              fontWeight: 600,
              color: "var(--amber)",
              letterSpacing: "0.04em",
            }}
          >
            {drawTool === "freehand" ? "⊞ SWITCH TO LINES" : "〰 SWITCH TO FREEHAND"}
          </button>
        )}
      </div>

      {/* Expand / collapse button */}
      {onToggleExpand && (
        <button
          onClick={(e) => { e.stopPropagation(); onToggleExpand(); }}
          style={{
            position: "absolute",
            top: 6, right: 10, zIndex: 1,
            padding: "2px 8px",
            background: "rgba(0, 0, 0, 0.55)",
            backdropFilter: "blur(8px)",
            border: "1px solid var(--border-subtle)",
            borderRadius: 3,
            cursor: "pointer",
            fontFamily: "var(--font-mono)",
            fontSize: 9,
            fontWeight: 600,
            color: "var(--text-tertiary)",
            letterSpacing: "0.06em",
            transition: "all 0.15s",
          }}
          onMouseEnter={(e) => { (e.target as HTMLElement).style.color = "var(--amber)"; }}
          onMouseLeave={(e) => { (e.target as HTMLElement).style.color = "var(--text-tertiary)"; }}
        >
          {expanded ? "▾ COLLAPSE" : "▴ EXPAND"}
        </button>
      )}

      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          maxHeight: expanded ? "none" : "28vh",
          height: expanded ? "100%" : undefined,
          display: "block",
          pointerEvents: "none",
          objectFit: expanded ? "contain" : undefined,
        }}
      />
    </div>
  );
}
