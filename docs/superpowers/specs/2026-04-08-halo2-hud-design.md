# Halo 2 HUD — Multi-Kill Medals & Motion Tracker Radar

**Date:** 2026-04-08
**Scope:** Frontend-only (no backend changes)

## Overview

Add two Halo 2-inspired HUD elements to the LiveFeed camera view:

1. **Multi-Kill Medal Overlay** — When zaps land within 6 seconds of each other, show escalating Halo 2 multi-kill medals with announcer voice lines.
2. **Motion Tracker Radar** — A circular radar in the bottom-left showing red dots for detected cats, mapped to pan/tilt servo space.

## Multi-Kill Streak System

### Tracking Logic

- A `useRef` stores `{ lastZapTime: number, streak: number }`.
- On every frame where `fired=true`:
  - If `Date.now() - lastZapTime < 6000` → increment streak.
  - Otherwise → reset streak to 1.
  - Update `lastZapTime`.
- Medals trigger at streak >= 2.

### Medal Tiers

| Streak | Name         | Icon           | Audio              |
|--------|--------------|----------------|--------------------|
| 2      | Double Kill  | `dbl.png`      | `dblVoice.mp3`     |
| 3      | Triple Kill  | `triple.png`   | `tripleVoice.mp3`  |
| 4      | Overkill     | `overki.png`   | `overkill.mp3`     |
| 5      | Killtacular  | `killtac.png`  | `killtac.mp3`      |
| 6      | Killtrocity  | `killtroc.png` | `killtrocVoice.mp3`|
| 7      | Killamanjaro | `killaman.png` | `killamanVoice.mp3`|
| 8      | Killtastrophy| `killtast.png` | `killtastVoice.mp3`|
| 9      | Killpocalypse| `killpoc.png`  | `killpocVoice.mp3` |
| 10+    | Killionaire  | `killion.png`  | `killionVoice.mp3` |

### Visual Presentation

- Medal icon + text centered over the LiveFeed.
- Fade in: scale 80% → 100%, opacity 0 → 1 over ~300ms.
- Hold for ~2 seconds.
- Fade out: opacity 1 → 0 over ~500ms.
- CSS keyframe animations in `index.css`.

### Audio Queue

- Voice lines are queued sequentially, never overlapping.
- Maintain a `useRef<string[]>` audio queue.
- When a medal triggers, push the voice line URL onto the queue.
- Playback loop: if nothing playing and queue non-empty, play next.
- Each clip's `onended` triggers the next queued clip.
- If streak resets mid-queue, queued lines still finish playing.

## Motion Tracker Radar

### Appearance

- Bottom-left corner of LiveFeed, ~120px diameter circle.
- Semi-transparent dark background with green/teal border.
- Faint crosshair lines through center.
- Classic Halo 2 motion tracker aesthetic.

### Coordinate Mapping

- Center of radar = camera's current `servo_pan` / `servo_tilt`.
- Detection bbox centers converted from pixel-space to angle-space using camera FOV + servo angles.
- `occluded_cats` already carry predicted `[pan, tilt, ...]` — map directly.
- Radar covers full servo range (~180° pan, ~90° tilt).
- Dots plotted as offset from camera aim point.

### Dot Behavior

- Red dots, ~6px diameter, for each detected cat.
- Smooth movement via lerp between frames.
- Occluded cat dots rendered slightly dimmer.
- Empty radar still visible when no cats detected.

### Implementation

- Small `<canvas>` element overlaid on the feed.
- Redraws each frame with current detection data.
- Performant at ~30fps since it's lightweight draw calls.

## Component Structure

### New Components

- `MultiKillOverlay.tsx` — Streak tracking, medal display, audio queue. Receives `fired` from frame data.
- `RadarHUD.tsx` — Radar canvas rendering. Receives `detections`, `occludedCats`, `servoPan`, `servoTilt`.

### Integration

- Both rendered inside `LiveFeed.tsx` as absolutely-positioned overlay children.
- LiveFeed already tracks all required state.

### No Changes To

- Backend / server code
- WebSocket protocol
- Types (`FrameData` already carries everything needed)
- Any other existing components

### CSS

- Medal animations defined in `index.css` alongside existing `zapPulse`.
