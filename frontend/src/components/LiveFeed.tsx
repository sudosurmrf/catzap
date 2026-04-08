import { useEffect, useRef, useState } from "react";
import type { Detection, Violation, Zone } from "../types";

interface FireTarget {
  x: number;
  y: number;
  zone: string;
}

interface LiveFeedProps {
  zones: Zone[];
  canvasRefCallback?: (canvas: HTMLCanvasElement | null) => void;
  onClickPoint?: (x: number, y: number) => void;
}

export default function LiveFeed({ zones, canvasRefCallback, onClickPoint }: LiveFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fps, setFps] = useState(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [zapFlash, setZapFlash] = useState<FireTarget | null>(null);
  const zapTimeoutRef = useRef<number | null>(null);

  // Expose canvas ref to parent
  useEffect(() => {
    canvasRefCallback?.(canvasRef.current);
    return () => canvasRefCallback?.(null);
  }, [canvasRefCallback]);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/feed`);
    let frameCount = 0;
    let lastFpsTime = Date.now();

    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);

      if (data.frame) {
        const img = new Image();
        img.onload = () => {
          drawFrame(img, data.detections || [], data.violations || [], data.fired, data.fire_target);
          frameCount++;
          const now = Date.now();
          if (now - lastFpsTime >= 1000) {
            setFps(frameCount);
            frameCount = 0;
            lastFpsTime = now;
          }
        };
        img.src = `data:image/jpeg;base64,${data.frame}`;
        setDetections(data.detections || []);
        setViolations(data.violations || []);

        if (data.fired && data.fire_target) {
          setZapFlash(data.fire_target);
          if (zapTimeoutRef.current) clearTimeout(zapTimeoutRef.current);
          zapTimeoutRef.current = window.setTimeout(() => setZapFlash(null), 1500);
        }
      }
    };

    return () => {
      ws.close();
      if (zapTimeoutRef.current) clearTimeout(zapTimeoutRef.current);
    };
  }, [zones]);

  function drawFrame(
    img: HTMLImageElement,
    dets: Detection[],
    viols: Violation[],
    fired: boolean,
    fireTarget: FireTarget | null,
  ) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const w = canvas.width;
    const h = canvas.height;

    // Draw zones
    for (const zone of zones) {
      if (!zone.enabled) continue;
      ctx.strokeStyle = "#f94144";
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 4]);
      ctx.beginPath();
      zone.polygon.forEach(([x, y], i) => {
        if (i === 0) ctx.moveTo(x * w, y * h);
        else ctx.lineTo(x * w, y * h);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = "rgba(249, 65, 68, 0.08)";
      ctx.fill();
      ctx.setLineDash([]);

      if (zone.polygon.length > 0) {
        const [lx, ly] = zone.polygon[0];
        ctx.fillStyle = "#f94144";
        ctx.font = "12px monospace";
        ctx.fillText(zone.name, lx * w, ly * h - 4);
      }
    }

    // Draw detections
    for (const det of dets) {
      const [x1, y1, x2, y2] = det.bbox;
      const isViolating = viols.some((v) =>
        zones.some((z) => z.id === v.zone_id)
      );

      ctx.strokeStyle = isViolating ? "#f94144" : "#4cc9f0";
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);

      const label = det.cat_name
        ? `${det.cat_name} ${Math.round(det.confidence * 100)}%`
        : `Cat ${Math.round(det.confidence * 100)}%`;
      const labelColor = isViolating ? "rgba(249, 65, 68, 0.4)" : "rgba(76, 201, 240, 0.3)";
      ctx.fillStyle = labelColor;
      ctx.fillRect(x1 * w, y1 * h - 18, ctx.measureText(label).width + 8, 18);
      ctx.fillStyle = isViolating ? "#f94144" : "#4cc9f0";
      ctx.font = "12px monospace";
      ctx.fillText(label, x1 * w + 4, y1 * h - 4);

      if (isViolating) {
        ctx.fillStyle = "#f94144";
        ctx.font = "bold 11px monospace";
        ctx.fillText("IN ZONE!", x1 * w, y2 * h + 14);
      }
    }

    // Draw ZAP crosshair on fire target
    if (fired && fireTarget) {
      const tx = fireTarget.x * w;
      const ty = fireTarget.y * h;
      const r = 30;

      ctx.fillStyle = "rgba(249, 65, 68, 0.15)";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#f94144";
      ctx.lineWidth = 3;
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.arc(tx, ty, r, 0, Math.PI * 2);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(tx, ty, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#f94144";
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(tx - r - 10, ty);
      ctx.lineTo(tx - 8, ty);
      ctx.moveTo(tx + 8, ty);
      ctx.lineTo(tx + r + 10, ty);
      ctx.moveTo(tx, ty - r - 10);
      ctx.lineTo(tx, ty - 8);
      ctx.moveTo(tx, ty + 8);
      ctx.lineTo(tx, ty + r + 10);
      ctx.stroke();
    }
  }

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!canvasRef.current || !onClickPoint) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    onClickPoint(x, y);
  }

  return (
    <div style={{ position: "relative" }}>
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        style={{
          width: "100%",
          display: "block",
          cursor: onClickPoint ? "crosshair" : "default",
          background: "#1a1a2e",
          borderRadius: 8,
          minHeight: 300,
        }}
      />

      {/* ZAP overlay banner */}
      {zapFlash && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            background: "rgba(249, 65, 68, 0.9)",
            color: "white",
            padding: "16px 32px",
            borderRadius: 12,
            fontFamily: "monospace",
            fontSize: 28,
            fontWeight: "bold",
            textAlign: "center",
            animation: "zapPulse 0.3s ease-out",
            pointerEvents: "none",
            boxShadow: "0 0 40px rgba(249, 65, 68, 0.6)",
          }}
        >
          ZAP!
          <div style={{ fontSize: 14, fontWeight: "normal", marginTop: 4 }}>
            Cat in {zapFlash.zone}
          </div>
        </div>
      )}

      {/* Status bar */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          padding: "8px 12px",
          background: "rgba(0,0,0,0.7)",
          display: "flex",
          justifyContent: "space-between",
          fontFamily: "monospace",
          fontSize: "12px",
          color: "#ccc",
          borderRadius: "0 0 8px 8px",
        }}
      >
        <span>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: fps > 0 ? "#4cc9f0" : "#888",
              display: "inline-block",
              marginRight: 8,
            }}
          />
          {fps > 0 ? `LIVE — ${fps} FPS` : "Connecting..."}
        </span>
        <span>
          Cats: {detections.length} | Violations: {violations.length}
        </span>
      </div>

      <style>{`
        @keyframes zapPulse {
          0% { transform: translate(-50%, -50%) scale(0.5); opacity: 0; }
          50% { transform: translate(-50%, -50%) scale(1.1); opacity: 1; }
          100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
