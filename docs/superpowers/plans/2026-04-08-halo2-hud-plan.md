# Halo 2 HUD — Multi-Kill Medals & Motion Tracker Radar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Halo 2-style multi-kill medal overlays with announcer voice lines and a motion tracker radar HUD to the LiveFeed camera view.

**Architecture:** Two new frontend-only components (`MultiKillOverlay` and `RadarHUD`) rendered as absolutely-positioned overlays inside `LiveFeed.tsx`. Multi-kill tracking uses a streak ref comparing zap timestamps. Radar maps cat detections from pixel-space to servo angle-space on a small canvas. No backend changes.

**Tech Stack:** React, TypeScript, HTML5 Canvas, CSS keyframe animations, Web Audio (HTMLAudioElement)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `frontend/src/components/MultiKillOverlay.tsx` | Streak tracking, medal display, audio queue |
| Create | `frontend/src/components/RadarHUD.tsx` | Canvas-based motion tracker radar |
| Modify | `frontend/src/components/LiveFeed.tsx:779-848` | Mount both overlay components |
| Modify | `frontend/src/index.css:308-338` | Add medal fade-in/fade-out keyframes |

Assets already exist in `frontend/public/`: 9 PNGs (medal icons) + 9 MP3s (voice lines).

---

### Task 1: MultiKillOverlay Component

**Files:**
- Create: `frontend/src/components/MultiKillOverlay.tsx`

- [ ] **Step 1: Create the medal tier config**

Create `frontend/src/components/MultiKillOverlay.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";

interface MedalTier {
  name: string;
  icon: string;
  audio: string;
}

const MEDAL_TIERS: MedalTier[] = [
  { name: "Double Kill",   icon: "/dbl.png",      audio: "/dblVoice.mp3" },
  { name: "Triple Kill",   icon: "/triple.png",    audio: "/tripleVoice.mp3" },
  { name: "Overkill",      icon: "/overki.png",    audio: "/overkill.mp3" },
  { name: "Killtacular",   icon: "/killtac.png",   audio: "/killtac.mp3" },
  { name: "Killtrocity",   icon: "/killtroc.png",  audio: "/killtrocVoice.mp3" },
  { name: "Killamanjaro",  icon: "/killaman.png",  audio: "/killamanVoice.mp3" },
  { name: "Killtastrophy", icon: "/killtast.png",  audio: "/killtastVoice.mp3" },
  { name: "Killpocalypse", icon: "/killpoc.png",   audio: "/killpocVoice.mp3" },
  { name: "Killionaire",   icon: "/killion.png",   audio: "/killionVoice.mp3" },
];

const STREAK_WINDOW_MS = 6000;
const MEDAL_DISPLAY_MS = 2800; // 300ms fade-in + 2000ms hold + 500ms fade-out
```

- [ ] **Step 2: Build the streak tracker and audio queue**

Add below the constants:

```tsx
interface MultiKillOverlayProps {
  fired: boolean;
}

export default function MultiKillOverlay({ fired }: MultiKillOverlayProps) {
  const streakRef = useRef({ lastZapTime: 0, count: 0 });
  const audioQueueRef = useRef<string[]>([]);
  const isPlayingRef = useRef(false);
  const [activeMedal, setActiveMedal] = useState<MedalTier | null>(null);
  const medalTimeoutRef = useRef<number | null>(null);
  const prevFiredRef = useRef(false);

  // Audio queue processor — plays next queued voice line when current finishes
  function playNext() {
    if (audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }
    isPlayingRef.current = true;
    const src = audioQueueRef.current.shift()!;
    const audio = new Audio(src);
    audio.onended = playNext;
    audio.onerror = playNext; // skip broken clips
    audio.play().catch(playNext);
  }

  function enqueueAudio(src: string) {
    audioQueueRef.current.push(src);
    if (!isPlayingRef.current) playNext();
  }

  // Detect rising edge of fired (false → true) to count zaps
  useEffect(() => {
    const wasFired = prevFiredRef.current;
    prevFiredRef.current = fired;
    if (!fired || wasFired) return; // only trigger on rising edge

    const now = Date.now();
    const streak = streakRef.current;

    if (now - streak.lastZapTime < STREAK_WINDOW_MS) {
      streak.count += 1;
    } else {
      streak.count = 1;
    }
    streak.lastZapTime = now;

    if (streak.count < 2) return;

    // Tier index: count 2 = index 0 (Double Kill), capped at last tier
    const tierIndex = Math.min(streak.count - 2, MEDAL_TIERS.length - 1);
    const medal = MEDAL_TIERS[tierIndex];

    // Show medal overlay
    setActiveMedal(medal);
    if (medalTimeoutRef.current) clearTimeout(medalTimeoutRef.current);
    medalTimeoutRef.current = window.setTimeout(() => setActiveMedal(null), MEDAL_DISPLAY_MS);

    // Queue voice line
    enqueueAudio(medal.audio);
  }, [fired]);

  useEffect(() => {
    return () => {
      if (medalTimeoutRef.current) clearTimeout(medalTimeoutRef.current);
    };
  }, []);

  if (!activeMedal) return null;

  return (
    <div style={{
      position: "absolute",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 8,
      pointerEvents: "none",
      zIndex: 20,
      animation: "medalFadeIn 0.3s ease-out forwards, medalFadeOut 0.5s ease-in 2.3s forwards",
    }}>
      <img
        src={activeMedal.icon}
        alt={activeMedal.name}
        style={{ width: 96, height: 96, filter: "drop-shadow(0 0 20px rgba(255, 200, 50, 0.6))" }}
      />
      <div style={{
        fontFamily: "var(--font-display)",
        fontSize: 22,
        fontWeight: 800,
        color: "#fff",
        textTransform: "uppercase",
        letterSpacing: "0.12em",
        textShadow: "0 0 12px rgba(255, 200, 50, 0.8), 0 2px 8px rgba(0,0,0,0.7)",
      }}>
        {activeMedal.name}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify component compiles**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors related to `MultiKillOverlay.tsx`

---

### Task 2: RadarHUD Component

**Files:**
- Create: `frontend/src/components/RadarHUD.tsx`

- [ ] **Step 1: Create the radar component with canvas setup**

Create `frontend/src/components/RadarHUD.tsx`:

```tsx
import { useEffect, useRef } from "react";
import type { Detection, OccludedCat } from "../types";

// Must add OccludedCat to types/index.ts exports if not already there
// OccludedCat: { id: string; predicted: [number, number, number]; occluded_by: string }

interface RadarHUDProps {
  detections: Detection[];
  occludedCats: OccludedCat[];
  servoPan: number;
  servoTilt: number;
}

const SIZE = 120;
const HALF = SIZE / 2;
const DOT_RADIUS = 4;
const FOV_H = 65;
const FOV_V = 50;
// Full servo range for radar scale
const PAN_RANGE = 180;
const TILT_RANGE = 90;

export default function RadarHUD({ detections, occludedCats, servoPan, servoTilt }: RadarHUDProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smoothDotsRef = useRef<Map<string, { x: number; y: number }>>(new Map());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, SIZE, SIZE);

    // ── Background circle ──
    ctx.beginPath();
    ctx.arc(HALF, HALF, HALF - 2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fill();
    ctx.strokeStyle = "rgba(34, 211, 238, 0.4)"; // cyan/teal border
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // ── Crosshair lines ──
    ctx.strokeStyle = "rgba(34, 211, 238, 0.15)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(HALF, 4);
    ctx.lineTo(HALF, SIZE - 4);
    ctx.moveTo(4, HALF);
    ctx.lineTo(SIZE - 4, HALF);
    ctx.stroke();

    // ── FOV cone (shows current camera view area) ──
    const fovW = (FOV_H / PAN_RANGE) * (SIZE - 8);
    const fovH = (FOV_V / TILT_RANGE) * (SIZE - 8);
    ctx.strokeStyle = "rgba(34, 211, 238, 0.2)";
    ctx.lineWidth = 1;
    ctx.strokeRect(HALF - fovW / 2, HALF - fovH / 2, fovW, fovH);

    // ── Collect all cat dots ──
    const dots: { key: string; targetX: number; targetY: number; dimmed: boolean }[] = [];

    // Detections: convert bbox center from pixel-space to angle offset
    for (let i = 0; i < detections.length; i++) {
      const d = detections[i];
      const [x1, y1, x2, y2] = d.bbox;
      const cx = (x1 + x2) / 2; // 0..1 normalized
      const cy = (y1 + y2) / 2;
      // Convert pixel-norm to absolute angle
      const catPan = servoPan + (cx - 0.5) * FOV_H;
      const catTilt = servoTilt + (cy - 0.5) * FOV_V;
      // Map angle to radar position (centered on mid-servo-range)
      const rx = HALF + ((catPan - 90) / PAN_RANGE) * (SIZE - 8);
      const ry = HALF + ((catTilt - 45) / TILT_RANGE) * (SIZE - 8);
      dots.push({ key: `det-${i}`, targetX: rx, targetY: ry, dimmed: false });
    }

    // Occluded cats: already have predicted [pan, tilt, ...]
    for (const oc of occludedCats) {
      const [pan, tilt] = oc.predicted;
      const rx = HALF + ((pan - 90) / PAN_RANGE) * (SIZE - 8);
      const ry = HALF + ((tilt - 45) / TILT_RANGE) * (SIZE - 8);
      dots.push({ key: `occ-${oc.id}`, targetX: rx, targetY: ry, dimmed: true });
    }

    // ── Lerp smoothing ──
    const smoothDots = smoothDotsRef.current;
    const LERP = 0.3;
    const activeDotKeys = new Set<string>();

    for (const dot of dots) {
      activeDotKeys.add(dot.key);
      const prev = smoothDots.get(dot.key);
      let sx: number, sy: number;
      if (prev) {
        sx = prev.x + (dot.targetX - prev.x) * LERP;
        sy = prev.y + (dot.targetY - prev.y) * LERP;
      } else {
        sx = dot.targetX;
        sy = dot.targetY;
      }
      smoothDots.set(dot.key, { x: sx, y: sy });

      // Clamp to radar circle
      const dx = sx - HALF;
      const dy = sy - HALF;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const maxR = HALF - DOT_RADIUS - 3;
      if (dist > maxR) {
        sx = HALF + (dx / dist) * maxR;
        sy = HALF + (dy / dist) * maxR;
      }

      // Draw dot
      ctx.beginPath();
      ctx.arc(sx, sy, DOT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = dot.dimmed ? "rgba(239, 68, 68, 0.4)" : "rgba(239, 68, 68, 0.9)";
      ctx.fill();
      if (!dot.dimmed) {
        ctx.shadowColor = "rgba(239, 68, 68, 0.6)";
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // Clean up stale dots
    for (const key of smoothDots.keys()) {
      if (!activeDotKeys.has(key)) smoothDots.delete(key);
    }
  }, [detections, occludedCats, servoPan, servoTilt]);

  return (
    <canvas
      ref={canvasRef}
      width={SIZE}
      height={SIZE}
      style={{
        position: "absolute",
        bottom: 10,
        left: 10,
        borderRadius: "50%",
        pointerEvents: "none",
        zIndex: 10,
      }}
    />
  );
}
```

- [ ] **Step 2: Verify component compiles**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors related to `RadarHUD.tsx`. If `OccludedCat` is not exported from types, proceed to Task 3 which handles that.

---

### Task 3: Update Types (if needed)

**Files:**
- Modify: `frontend/src/types/index.ts:54-58`

- [ ] **Step 1: Check OccludedCat is exported**

The `OccludedCat` interface already exists at `frontend/src/types/index.ts:54-58`. Verify it is exported. If it is not (i.e. it's only used as an inline type), add the `export` keyword:

```typescript
export interface OccludedCat {
  id: string;
  predicted: [number, number, number];
  occluded_by: string;
}
```

- [ ] **Step 2: Verify types compile**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: PASS — no type errors

---

### Task 4: Add CSS Keyframe Animations

**Files:**
- Modify: `frontend/src/index.css:338` (after the existing `zapPulse` keyframe)

- [ ] **Step 1: Add medal animations**

Add after the `zapPulse` keyframe block (after line 338 in `index.css`):

```css
@keyframes medalFadeIn {
  0% {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.8);
  }
  100% {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes medalFadeOut {
  0% {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
  100% {
    opacity: 0;
    transform: translate(-50%, -50%) scale(1.05);
  }
}
```

- [ ] **Step 2: Verify CSS parses**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -10`
Expected: PASS (CSS doesn't affect TS compilation, but this confirms nothing broke)

---

### Task 5: Integrate Components into LiveFeed

**Files:**
- Modify: `frontend/src/components/LiveFeed.tsx:1-2` (imports)
- Modify: `frontend/src/components/LiveFeed.tsx:779-848` (overlay section)

- [ ] **Step 1: Add imports**

At the top of `LiveFeed.tsx`, after line 4 (`import DirectionArrow`), add:

```typescript
import MultiKillOverlay from "./MultiKillOverlay";
import RadarHUD from "./RadarHUD";
```

- [ ] **Step 2: Add fired state tracking**

In the state declarations area (around line 53, near `zapFlash`), add:

```typescript
const [firedState, setFiredState] = useState(false);
```

In the WebSocket `onmessage` handler (around line 230, after `setOccludedCats`), add:

```typescript
setFiredState(!!data.fired);
```

- [ ] **Step 3: Mount RadarHUD in the overlay**

In the JSX return, replace the occluded cats block (lines 836-848):

```tsx
      {/* Occluded cat indicators */}
      {!drawMode && occludedCats.length > 0 && (
        <div style={{ position: "absolute", bottom: 10, left: 10, display: "flex", flexDirection: "column", gap: 3 }}>
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
```

with:

```tsx
      {/* Halo 2 Motion Tracker Radar */}
      {!drawMode && (
        <RadarHUD
          detections={detections}
          occludedCats={(occludedCats as any)} 
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
```

Note: The `as any` cast for occludedCats is because LiveFeed stores a reduced `{ id, occluded_by }` shape. If the full `OccludedCat` type (with `predicted`) is needed, we need to update the `setOccludedCats` call to preserve the `predicted` field. See Step 4.

- [ ] **Step 4: Preserve predicted field for occluded cats**

In `LiveFeed.tsx` around line 237, the current code strips `predicted`:

```typescript
setOccludedCats((data.occluded_cats || []).map((c: any) => ({ id: c.id, occluded_by: c.occluded_by })));
```

Replace with:

```typescript
setOccludedCats((data.occluded_cats || []).map((c: any) => ({ id: c.id, predicted: c.predicted, occluded_by: c.occluded_by })));
```

And update the `occludedCats` state type (around line 59):

```typescript
const [occludedCats, setOccludedCats] = useState<{ id: string; predicted: [number, number, number]; occluded_by: string }[]>([]);
```

Now remove the `as any` cast on RadarHUD's `occludedCats` prop from Step 3.

- [ ] **Step 5: Mount MultiKillOverlay**

In the JSX return, after the ZAP overlay block (after line 793), add:

```tsx
      {/* Halo 2 Multi-Kill Medal */}
      <MultiKillOverlay fired={firedState} />
```

- [ ] **Step 6: Verify everything compiles**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -30`
Expected: PASS — no type errors

---

### Task 6: Manual Testing Checklist

- [ ] **Step 1: Start the frontend dev server**

Run: `cd frontend && npm run dev`

- [ ] **Step 2: Verify radar renders**

Open the app in the browser. The radar should appear in the bottom-left corner of the LiveFeed as a dark circle with cyan crosshairs and a FOV rectangle. When cats are detected, red dots should appear and track their positions.

- [ ] **Step 3: Verify multi-kill medals**

Trigger two zaps within 6 seconds. The "Double Kill" medal icon should fade in centered on the feed, hold for ~2 seconds, then fade out. The announcer voice line should play. A third zap within 6s should show "Triple Kill" and queue the voice line after the previous finishes.

- [ ] **Step 4: Verify no overlap with existing HUD**

Confirm the radar doesn't overlap the occluded cat indicators (they should sit above the radar). Confirm the medal overlay doesn't conflict with the existing ZAP flash (they can coexist — ZAP flash lasts 1.5s, medal lasts 2.8s).

---
