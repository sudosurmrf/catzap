import { useEffect, useState, useCallback, useMemo } from "react";
import LiveFeed from "./components/LiveFeed";
import PanoramaView from "./components/PanoramaView";
import ZoneConfigPanel from "./components/ZoneConfigPanel";
import SweepControls from "./components/SweepControls";
import type { Zone, Detection, FrameData, ZoneTransform } from "./types";
import { DEFAULT_TRANSFORM } from "./types";
import { getZones, setVirtualAngle, updateZone } from "./api/client";

const EMPTY_DETECTIONS: Detection[] = [];

const SWEEP_CONFIG = { panMin: 30, panMax: 150, tiltMin: 20, tiltMax: 70 };

export default function App() {
  const [zones, setZones] = useState<Zone[]>([]);
  const [editingZones, setEditingZones] = useState(false);
  const [latestPanorama, setLatestPanorama] = useState<string | null>(null);
  const [latestDetections, setLatestDetections] = useState<Detection[]>([]);
  const [servoPan, setServoPan] = useState(90);
  const [servoTilt, setServoTilt] = useState(45);
  const [panoExpanded, setPanoExpanded] = useState(false);

  // Pending polygon drawn from either surface (angle-space)
  const [pendingPolygon, setPendingPolygon] = useState<number[][] | null>(null);
  // Selected zone for editing (move/resize)
  const [selectedZoneId, setSelectedZoneId] = useState<string | null>(null);
  // Transform state for the active polygon (pending or selected zone)
  const [transform, setTransform] = useState<ZoneTransform>({ ...DEFAULT_TRANSFORM });
  // The original (untransformed) polygon — set when drawing completes or zone is selected
  const [originalPolygon, setOriginalPolygon] = useState<number[][] | null>(null);

  useEffect(() => {
    getZones().then(setZones).catch(console.error);
  }, []);

  const refreshZones = useCallback(() => getZones().then(setZones).catch(console.error), []);

  const handleFrameData = useCallback((data: FrameData) => {
    if (data.panorama) setLatestPanorama(data.panorama);
    setLatestDetections(data.detections?.length ? data.detections : EMPTY_DETECTIONS);
    setServoPan(data.servo_pan ?? 90);
    setServoTilt(data.servo_tilt ?? 45);
  }, []);

  const handleDrawComplete = useCallback((anglePolygon: number[][]) => {
    setPendingPolygon(anglePolygon);
    setOriginalPolygon(anglePolygon);
    setTransform({ ...DEFAULT_TRANSFORM });
  }, []);

  function startEditing() {
    setEditingZones(true);
    setPendingPolygon(null);
    setSelectedZoneId(null);
    setTransform({ ...DEFAULT_TRANSFORM });
    setOriginalPolygon(null);
  }

  function stopEditing() {
    setEditingZones(false);
    setPendingPolygon(null);
    setSelectedZoneId(null);
    setTransform({ ...DEFAULT_TRANSFORM });
    setOriginalPolygon(null);
  }

  const handleZoneSaved = useCallback(() => {
    setPendingPolygon(null);
    setSelectedZoneId(null);
    setTransform({ ...DEFAULT_TRANSFORM });
    setOriginalPolygon(null);
    refreshZones();
  }, [refreshZones]);

  const handleSelectZone = useCallback((id: string | null) => {
    setSelectedZoneId(id);
    if (id) {
      setZones((prev) => {
        const zone = prev.find((z) => z.id === id);
        if (zone) {
          setOriginalPolygon([...zone.polygon]);
          setTransform({ ...DEFAULT_TRANSFORM });
        }
        return prev;
      });
    } else {
      setOriginalPolygon(null);
      setTransform({ ...DEFAULT_TRANSFORM });
    }
  }, []);

  const onClickAngle = useMemo(
    () => (editingZones ? undefined : (pan: number, tilt: number) => setVirtualAngle(pan, tilt)),
    [editingZones],
  );

  const handleZonePolygonUpdate = useCallback(async (zoneId: string, newPolygon: number[][]) => {
    await updateZone(zoneId, { polygon: newPolygon });
    refreshZones();
  }, [refreshZones]);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "Escape" && editingZones) { stopEditing(); return; }
      if (e.key === "ArrowLeft") setVirtualAngle(servoPan - 5, servoTilt);
      else if (e.key === "ArrowRight") setVirtualAngle(servoPan + 5, servoTilt);
      else if (e.key === "ArrowUp") setVirtualAngle(servoPan, servoTilt - 5);
      else if (e.key === "ArrowDown") setVirtualAngle(servoPan, servoTilt + 5);
      else return;
      e.preventDefault();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [servoPan, servoTilt, editingZones]);

  return (
    <div style={{
      height: "100%",
      display: "flex",
      overflow: "hidden",
      background: "var(--bg-deep)",
    }}>
      {/* ── Left sidebar ────────────────────────── */}
      <aside style={{
        width: "var(--sidebar-width)",
        flexShrink: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "12px 8px",
        gap: 4,
        borderRight: "1px solid var(--border-subtle)",
        background: "var(--bg-base)",
      }}>
        {/* Logo mark */}
        <div style={{
          width: 32, height: 32,
          display: "flex", alignItems: "center", justifyContent: "center",
          marginBottom: 12,
        }}>
          <div style={{
            width: 10, height: 10,
            background: "var(--amber)",
            borderRadius: 2,
            transform: "rotate(45deg)",
            boxShadow: "0 0 12px var(--amber-dim)",
          }} />
        </div>

        <SweepControls />

        <div style={{ flex: 1 }} />

        {/* Zone edit toggle */}
        <div className="tooltip-wrapper" data-tooltip={editingZones ? "Exit zones (Esc)" : "Edit zones"}>
          <button
            className={`nav-btn ${editingZones ? "active" : ""}`}
            onClick={() => editingZones ? stopEditing() : startEditing()}
            style={editingZones ? {
              background: "var(--amber-glow)",
              color: "var(--amber)",
              animation: "breathe 2s infinite",
            } : {}}
          >
            ⬡
          </button>
        </div>

        <CompactClock />
      </aside>

      {/* ── Center: feed area ───────────────────── */}
      <main style={{
        flex: 1,
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        gap: 1,
        overflow: "hidden",
        position: "relative",
      }}>
        {/* Panorama strip — always visible, supports drawing in edit mode */}
        <div style={{
          flexShrink: 0,
          flex: panoExpanded ? 1 : undefined,
          minHeight: panoExpanded ? 0 : undefined,
          borderBottom: "1px solid var(--border-subtle)",
          overflow: "hidden",
        }}>
          <PanoramaView
            panoramaBase64={latestPanorama}
            zones={zones}
            detections={latestDetections}
            servoPan={servoPan}
            servoTilt={servoTilt}
            sweepPanMin={SWEEP_CONFIG.panMin}
            sweepPanMax={SWEEP_CONFIG.panMax}
            sweepTiltMin={SWEEP_CONFIG.tiltMin}
            sweepTiltMax={SWEEP_CONFIG.tiltMax}
            fovH={65}
            fovV={50}
            onClickAngle={onClickAngle}
            drawMode={editingZones && !selectedZoneId && !pendingPolygon}
            onDrawComplete={handleDrawComplete}
            pendingPolygon={pendingPolygon}
            selectedZoneId={editingZones ? selectedZoneId : null}
            onSelectZone={editingZones ? handleSelectZone : undefined}
            onZonePolygonUpdate={handleZonePolygonUpdate}
            transform={transform}
            onTransformChange={setTransform}
            originalPolygon={originalPolygon}
            expanded={panoExpanded}
            onToggleExpand={() => setPanoExpanded((v) => !v)}
          />
        </div>

        {/* Live feed — primary draw surface in edit mode */}
        <div style={{ flex: panoExpanded ? undefined : 1, minHeight: 0, height: panoExpanded ? 120 : undefined, flexShrink: 0 }}>
          <LiveFeed
            zones={zones}
            onFrameData={handleFrameData}
            drawMode={editingZones && !selectedZoneId && !pendingPolygon}
            onDrawComplete={handleDrawComplete}
            selectedZoneId={editingZones ? selectedZoneId : null}
            onSelectZone={editingZones ? handleSelectZone : undefined}
            onZonePolygonUpdate={handleZonePolygonUpdate}
            pendingPolygon={pendingPolygon}
          />
        </div>

        {/* Bottom info bar */}
        <div style={{
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          gap: 16,
          padding: "6px 14px",
          background: "var(--bg-base)",
          borderTop: "1px solid var(--border-subtle)",
        }}>
          <span style={{
            fontFamily: "var(--font-display)",
            fontSize: 13,
            fontWeight: 700,
            color: "var(--amber)",
            letterSpacing: "0.06em",
          }}>
            CATZAP
          </span>

          <div style={{ height: 12, width: 1, background: "var(--border-base)" }} />

          <span style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--text-tertiary)",
            letterSpacing: "0.02em",
            fontVariantNumeric: "tabular-nums",
          }}>
            PAN {servoPan.toFixed(0)}&deg; &middot; TILT {servoTilt.toFixed(0)}&deg;
          </span>

          <div style={{ flex: 1 }} />

          {editingZones && (
            <span style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--amber)",
              animation: "pulse 2s infinite",
            }}>
              Zone editing — draw on feed or panorama
            </span>
          )}

          <span style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--text-ghost)",
          }}>
            {zones.length} zone{zones.length !== 1 ? "s" : ""} active
          </span>
        </div>

        {/* Floating zone editor — shown in edit mode */}
        {editingZones && (
          <div style={{
            position: "absolute",
            right: 12,
            bottom: 48,
            width: 300,
            maxHeight: "60vh",
            overflow: "auto",
            zIndex: 50,
            background: "var(--bg-base)",
            border: "1px solid var(--border-base)",
            borderRadius: "var(--radius-md)",
            boxShadow: "var(--shadow-lg)",
            padding: 12,
          }}>
            <div style={{
              paddingBottom: 10,
              marginBottom: 8,
              borderBottom: "1px solid var(--border-subtle)",
              fontFamily: "var(--font-display)",
              fontSize: 12,
              fontWeight: 700,
              color: "var(--amber)",
              letterSpacing: "0.04em",
              textTransform: "uppercase",
            }}>
              Zones — editing
            </div>
            <ZoneConfigPanel
              zones={zones}
              pendingPolygon={pendingPolygon}
              onSaved={handleZoneSaved}
              onClearPending={() => { setPendingPolygon(null); setOriginalPolygon(null); setTransform({ ...DEFAULT_TRANSFORM }); }}
              onExit={stopEditing}
              selectedZoneId={selectedZoneId}
              onSelectZone={handleSelectZone}
              onZoneUpdated={refreshZones}
            />
          </div>
        )}
      </main>
    </div>
  );
}

function CompactClock() {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  const h = time.getHours().toString().padStart(2, "0");
  const m = time.getMinutes().toString().padStart(2, "0");
  return (
    <div style={{
      fontFamily: "var(--font-mono)",
      fontSize: 10,
      color: "var(--text-ghost)",
      letterSpacing: "0.04em",
      fontVariantNumeric: "tabular-nums",
      textAlign: "center",
      marginTop: 8,
    }}>
      {h}
      <span style={{ opacity: 0.4 }}>:</span>
      {m}
    </div>
  );
}
