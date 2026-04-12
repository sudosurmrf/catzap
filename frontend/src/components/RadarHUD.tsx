import { useEffect, useRef } from "react";
import type { Detection } from "../types";

interface RadarHUDProps {
  detections: Detection[];
  servoPan: number;
  servoTilt: number;
}

const SIZE = 360;
const HALF = SIZE / 2;
const DOT_RADIUS = 10;
const LERP = 0.3;
const FADE_DURATION_MS = 2000;

interface DotState {
  x: number;
  y: number;
  lastSeen: number;
  dimmed: boolean;
}

function clampToCircle(x: number, y: number): [number, number] {
  const maxR = HALF - DOT_RADIUS - 6;
  const dx = x - HALF;
  const dy = y - HALF;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist > maxR) {
    const scale = maxR / dist;
    return [HALF + dx * scale, HALF + dy * scale];
  }
  return [x, y];
}

export default function RadarHUD({
  detections,
  servoPan,
  servoTilt,
}: RadarHUDProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dotsRef = useRef<Map<string, DotState>>(new Map());
  const rafRef = useRef<number>(0);

  // Store latest props in refs so the animation loop reads current values
  const propsRef = useRef({ detections, servoPan, servoTilt });
  propsRef.current = { detections, servoPan, servoTilt };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const maybeCtx = canvas.getContext("2d");
    if (!maybeCtx) return;
    // Re-bind to a non-nullable local so the nested draw() closure below
    // doesn't re-widen back to CanvasRenderingContext2D | null.
    const ctx: CanvasRenderingContext2D = maybeCtx;

    let running = true;

    function draw() {
      if (!running) return;
      const { detections } = propsRef.current;
      const now = Date.now();
      const dots = dotsRef.current;

      ctx.clearRect(0, 0, SIZE, SIZE);

      // 1. Outer circle
      ctx.beginPath();
      ctx.arc(HALF, HALF, HALF - 2, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(34,211,238,0.05)";
      ctx.fill();
      ctx.strokeStyle = "rgba(34,211,238,0.4)";
      ctx.lineWidth = 2;
      ctx.stroke();

      // 2. Inner circle (slightly darker)
      const innerR = (HALF - 2) * 0.55;
      ctx.beginPath();
      ctx.arc(HALF, HALF, innerR, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(34,211,238,0.08)";
      ctx.fill();
      ctx.strokeStyle = "rgba(34,211,238,0.25)";
      ctx.lineWidth = 1;
      ctx.stroke();

      // 3. FOV cone — triangle from center fanning up, clipped to outer circle
      ctx.save();
      ctx.beginPath();
      ctx.arc(HALF, HALF, HALF - 2, 0, Math.PI * 2);
      ctx.clip();
      const coneSpread = HALF * 0.75; // half-width at the top edge
      ctx.beginPath();
      ctx.moveTo(HALF, HALF);                          // center
      ctx.lineTo(HALF - coneSpread, 0);                // top-left
      ctx.lineTo(HALF + coneSpread, 0);                // top-right
      ctx.closePath();
      ctx.fillStyle = "rgba(34,211,238,0.15)";
      ctx.fill();
      ctx.restore();

      // 4. Crosshair lines
      ctx.strokeStyle = "rgba(34,211,238,0.15)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(HALF, 8);
      ctx.lineTo(HALF, SIZE - 8);
      ctx.moveTo(8, HALF);
      ctx.lineTo(SIZE - 8, HALF);
      ctx.stroke();

      // 4. Player arrow — points straight up from center (our position + orientation)
      ctx.fillStyle = "rgba(34,211,238,0.8)";
      ctx.beginPath();
      ctx.moveTo(HALF, HALF - 14);      // tip (pointing up)
      ctx.lineTo(HALF - 7, HALF + 6);   // bottom-left
      ctx.lineTo(HALF, HALF + 2);       // notch
      ctx.lineTo(HALF + 7, HALF + 6);   // bottom-right
      ctx.closePath();
      ctx.fill();

      // Collect active dots this frame
      const activeKeys = new Set<string>();

      // Frame-relative: "in front" = up on radar
      // X in frame → X on radar (left/right preserved)
      // Y in frame → distance ahead (frame top = far ahead/up, frame bottom = close/down on radar)
      // Center of frame maps to directly above center on radar
      const radarR = HALF - 16;

      detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox;
        const cx = (x1 + x2) / 2;  // 0..1 normalized
        const cy = (y1 + y2) / 2;
        // X: left/right maps to left/right on radar
        const rx = HALF + (cx - 0.5) * radarR * 2;
        // Y: frame center (0.5) maps above center on radar, frame edges extend out
        const ry = HALF - (1 - cy) * radarR;
        // Use stable track_id from server so the same cat keeps the same dot
        const key = det.track_id != null ? `det-${det.track_id}` : `det-pos-${cx.toFixed(2)}-${cy.toFixed(2)}`;
        activeKeys.add(key);

        const prev = dots.get(key);
        const lx = prev ? prev.x + (rx - prev.x) * LERP : rx;
        const ly = prev ? prev.y + (ry - prev.y) * LERP : ry;
        dots.set(key, { x: lx, y: ly, lastSeen: now, dimmed: false });
      });

      // Draw all dots (active + fading)
      for (const [key, dot] of dots) {
        const age = now - dot.lastSeen;
        const isActive = activeKeys.has(key);

        // Remove fully faded dots
        if (!isActive && age > FADE_DURATION_MS) {
          dots.delete(key);
          continue;
        }

        // Opacity: full when active, fading when stale
        const fadeAlpha = isActive ? 1 : Math.max(0, 1 - age / FADE_DURATION_MS);
        const baseAlpha = dot.dimmed ? 0.4 : 0.9;
        const alpha = baseAlpha * fadeAlpha;

        const [cx, cy] = clampToCircle(dot.x, dot.y);

        ctx.beginPath();
        if (!dot.dimmed && fadeAlpha > 0.5) {
          ctx.shadowColor = `rgba(239,68,68,${0.6 * fadeAlpha})`;
          ctx.shadowBlur = 12;
        }
        ctx.fillStyle = `rgba(239,68,68,${alpha})`;
        ctx.arc(cx, cy, DOT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowColor = "transparent";
        ctx.shadowBlur = 0;
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={SIZE}
      height={SIZE}
      style={{
        position: "absolute",
        bottom: 10,
        left: 10,
        width: SIZE,
        height: SIZE,
        pointerEvents: "none",
        zIndex: 10,
      }}
    />
  );
}
