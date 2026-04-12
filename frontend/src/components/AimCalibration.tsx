import { useEffect, useRef, useState, useCallback } from "react";
import type { CSSProperties } from "react";
import {
  startAimCalibration,
  jogCalibration,
  setAimCalibrationHome,
  beginExtentCapture,
  recordExtentCorner,
  computeTileGrid,
  startVerification,
  confirmVerification,
  skipVerification,
  finalizeAimCalibration,
  cancelAimCalibration,
  getRigSettings,
  updateRigSettings,
  autoLevelPanAxis,
  testLevelSweep,
} from "../api/client";
import type {
  CalibrationPhase,
  CalibrationPose,
  ExtentCorner,
  ExtentCornerLabel,
  ExtentBounds,
  JogDirection,
  JogStep,
  RigSettings,
} from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
}

interface VerificationState {
  currentIndex: number;
  total: number;
  expectedPan: number;
  expectedTilt: number;
  tileCol: number;
  tileRow: number;
  lastResidual?: number;
  maxResidual?: number;
  meanResidual?: number;
  passed?: boolean;
  threshold?: number;
}

interface TileGridResult {
  bounds: ExtentBounds;
  tile_cols: number;
  tile_rows: number;
  total_tiles: number;
  fov_h: number;
  fov_v: number;
}

const EXTENT_LABELS: ExtentCornerLabel[] = ["bl", "tl", "tr", "br"];
const CORNER_LABEL_TEXT: Record<ExtentCornerLabel, string> = {
  bl: "Bottom-Left",
  tl: "Top-Left",
  tr: "Top-Right",
  br: "Bottom-Right",
};

export default function AimCalibration({ open, onClose }: Props) {
  const [phase, setPhase] = useState<CalibrationPhase | null>(null);
  const [currentPose, setCurrentPose] = useState<CalibrationPose>({ pan: 0, tilt: 0 });
  const [referenceFrameB64, setReferenceFrameB64] = useState<string | null>(null);
  const [verification, setVerification] = useState<VerificationState | null>(null);
  const [jogStep, setJogStep] = useState<JogStep>("coarse");
  const [rigSettings, setRigSettings] = useState<RigSettings>({
    tilt_jog_inverted: false,
    pan_jog_inverted: false,
    pan_tilt_poly: [0, 0, 0],
    pan_min: null,
    pan_max: null,
    tilt_min: null,
    tilt_max: null,
  });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Auto-level N-point recorder state (used during LEVELING phase).
  // Keyed by label so each button can be re-clicked to overwrite its own
  // recording. Minimum 3 labels needed to solve (quadratic has 3 unknowns);
  // 5 is recommended to average out the ±1° servo-snap noise on each point.
  const [recordedLevelPoints, setRecordedLevelPoints] = useState<
    Record<string, CalibrationPose>
  >({});
  const [autoLevelResult, setAutoLevelResult] = useState<{
    maxResidual: number;
    rmsResidual: number;
    numPoints: number;
  } | null>(null);
  const [sweepRunning, setSweepRunning] = useState(false);

  // Extent capture state (used during CAPTURING_EXTENT phase).
  // Keyed by corner label so each of the 4 buttons can be re-clicked to
  // overwrite its own recorded pose until the user is happy with it.
  const [recordedCorners, setRecordedCorners] = useState<
    Record<string, ExtentCorner>
  >({});
  const [tileGridResult, setTileGridResult] = useState<TileGridResult | null>(null);

  // Live feed — own WebSocket to /ws/feed, independent from the main App feed
  const [liveFrame, setLiveFrame] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // ── Live feed subscription ─────────────────────────
  useEffect(() => {
    if (!open) return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/feed`);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.frame) setLiveFrame(data.frame);
        if (typeof data.servo_pan === "number" && typeof data.servo_tilt === "number") {
          setCurrentPose({ pan: data.servo_pan, tilt: data.servo_tilt });
        }
      } catch {
        // ignore malformed frames
      }
    };
    ws.onerror = () => { /* ignore — main app will reconnect */ };
    return () => { ws.close(); wsRef.current = null; };
  }, [open]);

  // ── Session lifecycle ──────────────────────────────
  const startSession = useCallback(async () => {
    setError(null);
    setBusy(true);
    try {
      const [r, rs] = await Promise.all([startAimCalibration(), getRigSettings()]);
      setPhase(r.phase);
      setCurrentPose(r.current_pose);
      setReferenceFrameB64(null);
      setVerification(null);
      setRigSettings(rs);
      setRecordedCorners({});
      setTileGridResult(null);
    } catch (e: any) {
      setError(e.message ?? "Failed to start calibration");
    } finally {
      setBusy(false);
    }
  }, []);

  // Auto-start when opened
  useEffect(() => {
    if (open && phase === null) startSession();
  }, [open, phase, startSession]);

  // Reset internal state on close (so reopening starts fresh)
  useEffect(() => {
    if (!open) {
      setPhase(null);
      setReferenceFrameB64(null);
      setVerification(null);
      setRecordedLevelPoints({});
      setAutoLevelResult(null);
      setRecordedCorners({});
      setTileGridResult(null);
    }
  }, [open]);

  // ── Actions ────────────────────────────────────────
  const withBusy = async <T,>(fn: () => Promise<T>): Promise<T | null> => {
    setBusy(true);
    setError(null);
    try {
      return await fn();
    } catch (e: any) {
      setError(e.message ?? "Request failed");
      return null;
    } finally {
      setBusy(false);
    }
  };

  const doJog = useCallback(async (direction: JogDirection, stepOverride?: JogStep) => {
    const effectiveStep = stepOverride ?? jogStep;
    const result = await withBusy(() => jogCalibration(direction, effectiveStep));
    if (result) setCurrentPose({ pan: result.pan, tilt: result.tilt });
  }, [jogStep]);

  const doSetHome = async () => {
    const r = await withBusy(() => setAimCalibrationHome());
    if (!r) return;
    setPhase(r.phase);
    setReferenceFrameB64(r.reference_frame_b64);
  };

  const doBeginExtent = async () => {
    const r = await withBusy(() => beginExtentCapture());
    if (!r) return;
    setPhase(r.phase);
    setRecordedCorners(r.recorded_corners);
  };

  const doRecordCorner = async (label: ExtentCornerLabel) => {
    const r = await withBusy(() => recordExtentCorner(label));
    if (!r) return;
    setRecordedCorners(r.recorded_corners);
  };

  const doComputeTileGrid = async () => {
    const r = await withBusy(() => computeTileGrid());
    if (!r) return;
    setPhase(r.phase);
    setTileGridResult({
      bounds: r.bounds,
      tile_cols: r.tile_cols,
      tile_rows: r.tile_rows,
      total_tiles: r.total_tiles,
      fov_h: r.fov_h,
      fov_v: r.fov_v,
    });
  };

  const doStartVerification = async () => {
    const r = await withBusy(() => startVerification());
    if (!r) return;
    setPhase(r.phase ?? "verifying");
    setVerification({
      currentIndex: r.current_index,
      total: r.total,
      expectedPan: r.expected_pan,
      expectedTilt: r.expected_tilt,
      tileCol: r.tile_col,
      tileRow: r.tile_row,
    });
  };

  const doConfirmVerification = async () => {
    const r = await withBusy(() => confirmVerification());
    if (!r) return;
    if (r.complete) {
      setPhase("complete");
      setVerification((v) => v ? {
        ...v,
        lastResidual: r.last_residual,
        maxResidual: r.max_residual,
        meanResidual: r.mean_residual,
        passed: r.passed,
        threshold: r.threshold,
      } : null);
    } else {
      setVerification({
        currentIndex: r.current_index ?? 0,
        total: r.total ?? 0,
        expectedPan: r.expected_pan ?? 0,
        expectedTilt: r.expected_tilt ?? 0,
        tileCol: r.tile_col ?? 0,
        tileRow: r.tile_row ?? 0,
        lastResidual: r.last_residual,
      });
    }
  };

  const doSkipVerification = async () => {
    const r = await withBusy(() => skipVerification());
    if (r) setPhase("complete");
  };

  const doFinalize = async () => {
    await withBusy(() => finalizeAimCalibration());
    onClose();
  };

  const doCancel = async () => {
    await withBusy(() => cancelAimCalibration());
    onClose();
  };

  const toggleTiltInvert = async () => {
    const next = !rigSettings.tilt_jog_inverted;
    const r = await withBusy(() => updateRigSettings({ tilt_jog_inverted: next }));
    if (r) setRigSettings(r);
  };

  const togglePanInvert = async () => {
    const next = !rigSettings.pan_jog_inverted;
    const r = await withBusy(() => updateRigSettings({ pan_jog_inverted: next }));
    if (r) setRigSettings(r);
  };

  const recordLevelPoint = (label: string) => {
    setRecordedLevelPoints((prev) => ({
      ...prev,
      [label]: { pan: currentPose.pan, tilt: currentPose.tilt },
    }));
    setError(null);
  };

  const clearLevelPoints = () => {
    setRecordedLevelPoints({});
    setAutoLevelResult(null);
  };

  const computeAutoLevel = async () => {
    const points = Object.values(recordedLevelPoints);
    if (points.length < 3) {
      setError("Need at least 3 recorded points for the quadratic fit (5 recommended)");
      return;
    }
    const r = await withBusy(() => autoLevelPanAxis(
      points.map((p) => ({ pan: p.pan, tilt: p.tilt })),
    ));
    if (!r) return;
    setRigSettings(r.rig_settings);
    setAutoLevelResult({
      maxResidual: r.max_residual,
      rmsResidual: r.rms_residual,
      numPoints: r.num_points,
    });
  };

  const runTestSweep = async () => {
    setSweepRunning(true);
    setError(null);
    try {
      await testLevelSweep();
    } catch (e: any) {
      setError(e.message ?? "Test sweep failed");
    } finally {
      setSweepRunning(false);
    }
  };

  const allCornersRecorded = Object.keys(recordedCorners).length === 4;

  // ── Keyboard shortcuts ─────────────────────────────
  useEffect(() => {
    if (!open) return;
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "Escape") { doCancel(); return; }
      if (e.key === "Shift") return;  // modifier only
      const stepForThisKey: JogStep = e.shiftKey ? "fine" : "coarse";

      if (e.key === "ArrowLeft") { doJog("left", stepForThisKey); e.preventDefault(); }
      else if (e.key === "ArrowRight") { doJog("right", stepForThisKey); e.preventDefault(); }
      else if (e.key === "ArrowUp") { doJog("up", stepForThisKey); e.preventDefault(); }
      else if (e.key === "ArrowDown") { doJog("down", stepForThisKey); e.preventDefault(); }
      else if (e.key === "Enter") {
        // Phase-appropriate primary action
        if (phase === "jogging_to_home") doSetHome();
        else if (phase === "leveling") doBeginExtent();
        else if (phase === "capturing_extent" && allCornersRecorded) doComputeTileGrid();
        else if (phase === "extent_ready") doStartVerification();
        else if (phase === "verifying") doConfirmVerification();
        else if (phase === "complete") doFinalize();
        e.preventDefault();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [open, phase, allCornersRecorded, doJog]);

  if (!open) return null;

  // ── Render helpers ─────────────────────────────────
  const cornerCount = Object.keys(recordedCorners).length;

  const phaseLabel = (() => {
    if (!phase) return "Starting...";
    if (phase === "jogging_to_home") return "Step 1 — Jog to chosen home pose";
    if (phase === "leveling") return "Step 2 — Auto-level pan-axis compensation";
    if (phase === "capturing_extent") {
      return `Step 3 — Record 4 safe-envelope corners (${cornerCount}/4)`;
    }
    if (phase === "extent_ready") return "Step 4 — Review tile grid, begin verification";
    if (phase === "verifying") return `Step 5 — Verify tile ${(verification?.currentIndex ?? 0) + 1} / ${verification?.total ?? 0}`;
    if (phase === "complete") return "Done — review results and finalize";
    return phase;
  })();

  // Format a corner pose for button display. Shown as absolute (pan, tilt)
  // since the frontend doesn't track the home pose separately across phase
  // transitions; if you want to see the offset from home, hover over the
  // button for the title.
  const formatCorner = (corner: ExtentCorner): string =>
    `${corner.servo_pan.toFixed(0)},${corner.servo_tilt.toFixed(0)}`;

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      background: "rgba(0,0,0,0.92)",
      display: "flex", flexDirection: "column",
      fontFamily: "var(--font-mono)",
    }}>
      {/* ── Header ───────────────────────────────── */}
      <div style={{
        padding: "14px 20px",
        borderBottom: "1px solid var(--border-subtle)",
        display: "flex", alignItems: "center", gap: 16,
        background: "var(--bg-base)",
      }}>
        <span style={{
          fontFamily: "var(--font-display)",
          fontSize: 14, fontWeight: 700,
          color: "var(--amber)",
          letterSpacing: "0.06em",
        }}>
          AIM CALIBRATION
        </span>
        <span style={{ height: 14, width: 1, background: "var(--border-base)" }} />
        <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{phaseLabel}</span>
        <div style={{ flex: 1 }} />
        <span style={{
          fontSize: 11, color: "var(--text-tertiary)",
          fontVariantNumeric: "tabular-nums",
        }}>
          PAN {currentPose.pan.toFixed(1)}° · TILT {currentPose.tilt.toFixed(1)}°
        </span>
        <button className="btn" onClick={doCancel} disabled={busy}>
          Cancel (Esc)
        </button>
      </div>

      {error && (
        <div style={{
          padding: "8px 20px",
          background: "var(--red-dim, rgba(255,60,60,0.2))",
          color: "var(--red, #ff6060)",
          fontSize: 11,
        }}>
          {error}
        </div>
      )}

      {/* ── Main split ───────────────────────────── */}
      <div style={{
        flex: 1, display: "flex", gap: 2,
        minHeight: 0, padding: 2,
      }}>
        {/* Reference frame (frozen) */}
        <div style={{
          flex: 1, minWidth: 0,
          background: "var(--bg-deep)",
          display: "flex", flexDirection: "column",
          border: "1px solid var(--border-subtle)",
        }}>
          <div style={{
            padding: "6px 12px",
            borderBottom: "1px solid var(--border-subtle)",
            fontSize: 10, color: "var(--text-tertiary)",
            textTransform: "uppercase", letterSpacing: "0.06em",
          }}>
            Reference frame (frozen at home)
          </div>
          <div style={{
            flex: 1, position: "relative", minHeight: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            {referenceFrameB64 ? (
              <div style={{ position: "relative", maxWidth: "100%", maxHeight: "100%" }}>
                <img
                  src={`data:image/jpeg;base64,${referenceFrameB64}`}
                  style={{ display: "block", maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                />
                {phase === "leveling" && <HorizontalReferenceLine />}
              </div>
            ) : (
              <div style={{ color: "var(--text-ghost)", fontSize: 11 }}>
                {phase === "jogging_to_home"
                  ? "Waiting for you to set home pose…"
                  : "No reference frame"}
              </div>
            )}
          </div>
        </div>

        {/* Live feed */}
        <div style={{
          flex: 1, minWidth: 0,
          background: "var(--bg-deep)",
          display: "flex", flexDirection: "column",
          border: "1px solid var(--border-subtle)",
        }}>
          <div style={{
            padding: "6px 12px",
            borderBottom: "1px solid var(--border-subtle)",
            fontSize: 10, color: "var(--text-tertiary)",
            textTransform: "uppercase", letterSpacing: "0.06em",
          }}>
            Live feed (gun-aimed)
          </div>
          <div style={{
            flex: 1, position: "relative", minHeight: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            {liveFrame ? (
              <div style={{ position: "relative", maxWidth: "100%", maxHeight: "100%" }}>
                <img
                  src={`data:image/jpeg;base64,${liveFrame}`}
                  style={{ display: "block", maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                />
                {phase === "leveling" && <HorizontalReferenceLine />}
                {/* Center reticle */}
                <div style={{
                  position: "absolute", left: "50%", top: "50%",
                  transform: "translate(-50%, -50%)",
                  pointerEvents: "none",
                }}>
                  <CenterReticle />
                </div>
              </div>
            ) : (
              <div style={{ color: "var(--text-ghost)", fontSize: 11 }}>
                Connecting to feed…
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Controls ─────────────────────────────── */}
      <div style={{
        padding: "14px 20px",
        borderTop: "1px solid var(--border-subtle)",
        background: "var(--bg-base)",
        display: "flex", gap: 20, alignItems: "center",
      }}>
        {/* Jog pad */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
          <button className="btn" style={jogBtnStyle} onClick={() => doJog("up")} disabled={busy}>↑</button>
          <div style={{ display: "flex", gap: 4 }}>
            <button className="btn" style={jogBtnStyle} onClick={() => doJog("left")} disabled={busy}>←</button>
            <button className="btn" style={jogBtnStyle} onClick={() => doJog("right")} disabled={busy}>→</button>
          </div>
          <button className="btn" style={jogBtnStyle} onClick={() => doJog("down")} disabled={busy}>↓</button>
        </div>

        {/* Step selector */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ fontSize: 9, color: "var(--text-ghost)", textTransform: "uppercase" }}>Jog step</div>
          <div style={{ display: "flex", gap: 4 }}>
            <button
              className={`btn ${jogStep === "coarse" ? "btn-primary" : ""}`}
              onClick={() => setJogStep("coarse")}
              style={{ fontSize: 10, padding: "4px 10px" }}
            >
              Coarse (6°)
            </button>
            <button
              className={`btn ${jogStep === "fine" ? "btn-primary" : ""}`}
              onClick={() => setJogStep("fine")}
              style={{ fontSize: 10, padding: "4px 10px" }}
            >
              Fine (2°)
            </button>
          </div>
        </div>

        {/* Invert toggles */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ fontSize: 9, color: "var(--text-ghost)", textTransform: "uppercase" }}>Invert</div>
          <div style={{ display: "flex", gap: 4 }}>
            <button
              className={`btn ${rigSettings.tilt_jog_inverted ? "btn-primary" : ""}`}
              onClick={toggleTiltInvert}
              disabled={busy}
              style={{ fontSize: 10, padding: "4px 10px" }}
            >
              Tilt {rigSettings.tilt_jog_inverted ? "⇅" : "⇵"}
            </button>
            <button
              className={`btn ${rigSettings.pan_jog_inverted ? "btn-primary" : ""}`}
              onClick={togglePanInvert}
              disabled={busy}
              style={{ fontSize: 10, padding: "4px 10px" }}
            >
              Pan {rigSettings.pan_jog_inverted ? "⇆" : "⇄"}
            </button>
          </div>
        </div>

        {/* Tilt-compensation polynomial — read-only display of the coefficients
            that /auto-level most recently fit. */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div
            style={{ fontSize: 9, color: "var(--text-ghost)", textTransform: "uppercase" }}
            title="Quadratic tilt correction: delta(pan) = a + b*dp + c*dp² where dp = pan − home. Fit by auto-level."
          >
            Tilt comp
          </div>
          <div style={{
            fontSize: 10, color: "var(--text-secondary)",
            fontVariantNumeric: "tabular-nums",
            lineHeight: 1.4,
          }}>
            <div>a {(rigSettings.pan_tilt_poly[0] ?? 0).toFixed(3)}</div>
            <div>b {(rigSettings.pan_tilt_poly[1] ?? 0).toFixed(4)}</div>
            <div>c {(rigSettings.pan_tilt_poly[2] ?? 0).toFixed(5)}</div>
          </div>
        </div>

        {/* Auto-level N-point recorder — visible during LEVELING. */}
        {phase === "leveling" && (() => {
          const left = recordedLevelPoints.left;
          const midLeft = recordedLevelPoints.mid_left;
          const center = recordedLevelPoints.center;
          const midRight = recordedLevelPoints.mid_right;
          const right = recordedLevelPoints.right;
          const count = Object.keys(recordedLevelPoints).length;
          const fmt = (p?: CalibrationPose) =>
            p ? `${p.pan.toFixed(0)},${p.tilt.toFixed(0)}` : "";
          return (
            <div style={{
              display: "flex", flexDirection: "column", gap: 4,
              paddingLeft: 12, borderLeft: "1px solid var(--border-subtle)",
            }}>
              <div style={{ fontSize: 9, color: "var(--text-ghost)", textTransform: "uppercase" }}>
                Auto-level ({count}/5)
              </div>
              <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
                <button className={`btn ${left ? "btn-primary" : ""}`}
                  onClick={() => recordLevelPoint("left")} disabled={busy}
                  style={{ fontSize: 10, padding: "4px 8px" }}>
                  {left ? `L ${fmt(left)}` : "Rec L"}
                </button>
                <button className={`btn ${midLeft ? "btn-primary" : ""}`}
                  onClick={() => recordLevelPoint("mid_left")} disabled={busy}
                  style={{ fontSize: 10, padding: "4px 8px" }}>
                  {midLeft ? `ML ${fmt(midLeft)}` : "Rec ML"}
                </button>
                <button className={`btn ${center ? "btn-primary" : ""}`}
                  onClick={() => recordLevelPoint("center")} disabled={busy}
                  style={{ fontSize: 10, padding: "4px 8px" }}>
                  {center ? `C ${fmt(center)}` : "Rec C"}
                </button>
                <button className={`btn ${midRight ? "btn-primary" : ""}`}
                  onClick={() => recordLevelPoint("mid_right")} disabled={busy}
                  style={{ fontSize: 10, padding: "4px 8px" }}>
                  {midRight ? `MR ${fmt(midRight)}` : "Rec MR"}
                </button>
                <button className={`btn ${right ? "btn-primary" : ""}`}
                  onClick={() => recordLevelPoint("right")} disabled={busy}
                  style={{ fontSize: 10, padding: "4px 8px" }}>
                  {right ? `R ${fmt(right)}` : "Rec R"}
                </button>
              </div>
              <div style={{ display: "flex", gap: 4 }}>
                <button className="btn btn-primary" onClick={computeAutoLevel}
                  disabled={busy || count < 3}
                  style={{ fontSize: 10, padding: "4px 10px", opacity: count >= 3 ? 1 : 0.4, flex: 1 }}>
                  Compute poly
                </button>
                <button className="btn" onClick={clearLevelPoints}
                  disabled={busy || count === 0}
                  style={{ fontSize: 10, padding: "4px 8px" }}>
                  Clear
                </button>
              </div>
              <button className="btn" onClick={runTestSweep} disabled={busy || sweepRunning}
                style={{ fontSize: 10, padding: "4px 10px", opacity: sweepRunning ? 0.6 : 1 }}>
                {sweepRunning ? "Sweeping…" : "Test Sweep"}
              </button>
              {autoLevelResult && (
                <div style={{
                  fontSize: 9,
                  color: autoLevelResult.maxResidual < 1.0
                    ? "var(--green, #4ade80)"
                    : autoLevelResult.maxResidual < 2.5
                    ? "var(--amber, #fbbf24)"
                    : "var(--red, #f87171)",
                  fontVariantNumeric: "tabular-nums",
                }}>
                  fit: n={autoLevelResult.numPoints} · max={autoLevelResult.maxResidual.toFixed(2)}° · rms={autoLevelResult.rmsResidual.toFixed(2)}°
                </div>
              )}
            </div>
          );
        })()}

        {/* Extent capture panel — visible during CAPTURING_EXTENT.
            User jogs to each safe-envelope corner (bl → tl → tr → br) and
            clicks record. When all 4 are captured, clicks Compute Grid to
            derive the bounding box and tile dimensions. */}
        {phase === "capturing_extent" && (
          <div style={{
            display: "flex", flexDirection: "column", gap: 4,
            paddingLeft: 12, borderLeft: "1px solid var(--border-subtle)",
          }}>
            <div style={{ fontSize: 9, color: "var(--text-ghost)", textTransform: "uppercase" }}>
              Extent corners ({cornerCount}/4)
            </div>
            <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
              {EXTENT_LABELS.map((label) => {
                const rec = recordedCorners[label];
                return (
                  <button
                    key={label}
                    className={`btn ${rec ? "btn-primary" : ""}`}
                    onClick={() => doRecordCorner(label)}
                    disabled={busy}
                    style={{ fontSize: 10, padding: "4px 8px" }}
                    title={`Jog to the furthest safe ${CORNER_LABEL_TEXT[label].toLowerCase()} position, then click to record.`}
                  >
                    {rec ? `${label.toUpperCase()} ${formatCorner(rec)}` : `Rec ${label.toUpperCase()}`}
                  </button>
                );
              })}
            </div>
            <button
              className="btn btn-primary"
              onClick={doComputeTileGrid}
              disabled={busy || !allCornersRecorded}
              style={{
                fontSize: 10, padding: "4px 10px",
                opacity: allCornersRecorded ? 1 : 0.4,
              }}
              title="Derive bounding box and tile grid from the 4 recorded corners."
            >
              Compute Tile Grid
            </button>
          </div>
        )}

        {/* Tile grid preview — visible during EXTENT_READY.
            Shows the computed bounds and tile count before starting the
            verification sweep. */}
        {phase === "extent_ready" && tileGridResult && (
          <div style={{
            display: "flex", flexDirection: "column", gap: 4,
            paddingLeft: 12, borderLeft: "1px solid var(--border-subtle)",
            fontSize: 10, color: "var(--text-secondary)",
            fontVariantNumeric: "tabular-nums",
          }}>
            <div style={{ fontSize: 9, color: "var(--text-ghost)", textTransform: "uppercase" }}>
              Tile grid preview
            </div>
            <div>
              pan [{tileGridResult.bounds.pan_min.toFixed(0)}..{tileGridResult.bounds.pan_max.toFixed(0)}]°
            </div>
            <div>
              tilt [{tileGridResult.bounds.tilt_min.toFixed(0)}..{tileGridResult.bounds.tilt_max.toFixed(0)}]°
            </div>
            <div style={{ color: "var(--amber, #fbbf24)", fontWeight: 600 }}>
              {tileGridResult.tile_cols} × {tileGridResult.tile_rows} = {tileGridResult.total_tiles} tiles
            </div>
            <div style={{ fontSize: 9, color: "var(--text-ghost)" }}>
              fov {tileGridResult.fov_h.toFixed(0)}°×{tileGridResult.fov_v.toFixed(0)}°
            </div>
          </div>
        )}

        <div style={{ flex: 1 }} />

        {/* Primary action button (phase-dependent) */}
        <PrimaryAction
          phase={phase}
          allCornersRecorded={allCornersRecorded}
          busy={busy}
          verification={verification}
          onSetHome={doSetHome}
          onBeginExtent={doBeginExtent}
          onComputeTileGrid={doComputeTileGrid}
          onStartVerification={doStartVerification}
          onConfirmVerification={doConfirmVerification}
          onSkipVerification={doSkipVerification}
          onFinalize={doFinalize}
        />
      </div>

      {/* ── Verification result overlay ───── */}
      {phase === "complete" && verification && verification.passed !== undefined && (
        <div style={{
          position: "absolute",
          top: 60, right: 20,
          padding: 14,
          background: "var(--bg-base)",
          border: `1px solid ${verification.passed ? "var(--green, #4ade80)" : "var(--amber)"}`,
          borderRadius: 6,
          fontSize: 11,
          color: "var(--text-secondary)",
          minWidth: 260,
        }}>
          <div style={{
            fontSize: 10, textTransform: "uppercase", marginBottom: 6,
            color: verification.passed ? "var(--green, #4ade80)" : "var(--amber)",
          }}>
            Verification {verification.passed ? "PASSED" : "MARGINAL"}
          </div>
          <div>Max residual: {verification.maxResidual?.toFixed(2)}° / threshold {verification.threshold?.toFixed(1)}°</div>
          <div>Mean residual: {verification.meanResidual?.toFixed(2)}°</div>
          {tileGridResult && (
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid var(--border-subtle)", fontSize: 10 }}>
              <div>Bounds: pan [{tileGridResult.bounds.pan_min.toFixed(0)}..{tileGridResult.bounds.pan_max.toFixed(0)}]°</div>
              <div>        tilt [{tileGridResult.bounds.tilt_min.toFixed(0)}..{tileGridResult.bounds.tilt_max.toFixed(0)}]°</div>
              <div>Tiles: {tileGridResult.tile_cols} × {tileGridResult.tile_rows}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Primary action button ─────────────────────────────
interface PrimaryActionProps {
  phase: CalibrationPhase | null;
  allCornersRecorded: boolean;
  busy: boolean;
  verification: VerificationState | null;
  onSetHome: () => void;
  onBeginExtent: () => void;
  onComputeTileGrid: () => void;
  onStartVerification: () => void;
  onConfirmVerification: () => void;
  onSkipVerification: () => void;
  onFinalize: () => void;
}

function PrimaryAction(p: PrimaryActionProps) {
  const btnStyle = {
    fontSize: 12, padding: "8px 18px", minWidth: 180,
  };
  const btn = (label: string, onClick: () => void, primary = true, disabled = false) => (
    <button
      className={`btn ${primary ? "btn-primary" : ""}`}
      onClick={onClick}
      disabled={p.busy || disabled}
      style={btnStyle}
    >
      {label}
    </button>
  );

  if (p.phase === null) return <div style={{ fontSize: 11, color: "var(--text-ghost)" }}>Loading…</div>;
  if (p.phase === "jogging_to_home") return btn("Set Home Here", p.onSetHome);
  if (p.phase === "leveling") return btn("Continue to Extent Capture", p.onBeginExtent);
  if (p.phase === "capturing_extent") {
    return btn(
      p.allCornersRecorded ? "Compute Tile Grid" : "Record all 4 corners",
      p.onComputeTileGrid,
      true,
      !p.allCornersRecorded,
    );
  }
  if (p.phase === "extent_ready") {
    return (
      <div style={{ display: "flex", gap: 8 }}>
        {btn("Skip Verification", p.onSkipVerification, false)}
        {btn("Run Verification Sweep", p.onStartVerification)}
      </div>
    );
  }
  if (p.phase === "verifying") return btn("Confirm This Tile", p.onConfirmVerification);
  if (p.phase === "complete") return btn("Finalize & Resume", p.onFinalize);
  return null;
}

function HorizontalReferenceLine() {
  // A bright horizontal line overlaid on an image at its vertical midpoint.
  // Used during LEVELING phase — as the user pans, real-world horizontal
  // features should slide along this line if pan-axis compensation is tuned
  // correctly. If they drift above/below, the polynomial needs adjustment.
  return (
    <div style={{
      position: "absolute",
      left: 0,
      right: 0,
      top: "50%",
      height: 0,
      borderTop: "1.5px dashed var(--amber, #fbbf24)",
      pointerEvents: "none",
      boxShadow: "0 0 6px rgba(251, 191, 36, 0.5)",
    }} />
  );
}

function CenterReticle() {
  return (
    <svg width={40} height={40} style={{ display: "block" }}>
      <circle cx={20} cy={20} r={14} stroke="var(--amber, #fbbf24)" strokeWidth={1.5} fill="none" opacity={0.8} />
      <line x1={20} y1={0} x2={20} y2={14} stroke="var(--amber, #fbbf24)" strokeWidth={1} />
      <line x1={20} y1={26} x2={20} y2={40} stroke="var(--amber, #fbbf24)" strokeWidth={1} />
      <line x1={0} y1={20} x2={14} y2={20} stroke="var(--amber, #fbbf24)" strokeWidth={1} />
      <line x1={26} y1={20} x2={40} y2={20} stroke="var(--amber, #fbbf24)" strokeWidth={1} />
      <circle cx={20} cy={20} r={2} fill="var(--amber, #fbbf24)" />
    </svg>
  );
}

const jogBtnStyle: CSSProperties = {
  width: 36, height: 36,
  padding: 0,
  fontSize: 14,
  display: "flex", alignItems: "center", justifyContent: "center",
};
