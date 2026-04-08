import { useEffect, useRef, useState } from "react";

/* ── Medal tier config ──────────────────────────── */

interface MedalTier {
  name: string;
  icon: string;
  audio: string;
}

const MEDAL_TIERS: MedalTier[] = [
  { name: "Double Kill",   icon: "/dbl.png",      audio: "/dblVoice.mp3" },
  { name: "Triple Kill",   icon: "/triple.png",   audio: "/tripleVoice.mp3" },
  { name: "Overkill",      icon: "/overki.png",   audio: "/overkill.mp3" },
  { name: "Killtacular",   icon: "/killtac.png",  audio: "/killtac.mp3" },
  { name: "Killtrocity",   icon: "/killtroc.png", audio: "/killtrocVoice.mp3" },
  { name: "Killamanjaro",  icon: "/killaman.png",  audio: "/killamanVoice.mp3" },
  { name: "Killtastrophy", icon: "/killtast.png",  audio: "/killtastVoice.mp3" },
  { name: "Killpocalypse", icon: "/killpoc.png",   audio: "/killpocVoice.mp3" },
  { name: "Killionaire",   icon: "/killion.png",   audio: "/killionVoice.mp3" },
];

const STREAK_WINDOW_MS = 6000;
const MEDAL_DISPLAY_MS = 2800;

/* ── Component ──────────────────────────────────── */

interface MultiKillOverlayProps {
  fired: boolean;
}

export default function MultiKillOverlay({ fired }: MultiKillOverlayProps) {
  const [activeMedal, setActiveMedal] = useState<MedalTier | null>(null);
  const [medalKey, setMedalKey] = useState(0);

  // Streak tracking
  const streakRef = useRef({ lastZapTime: 0, count: 0 });
  const prevFiredRef = useRef(false);
  const medalTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Audio queue — reuse a single Audio element to respect browser autoplay policy
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioQueueRef = useRef<string[]>([]);
  const isPlayingRef = useRef(false);

  // Unlock audio on first user interaction
  useEffect(() => {
    const audio = new Audio();
    audioRef.current = audio;

    const unlock = () => {
      audio.play().then(() => audio.pause()).catch(() => {});
      window.removeEventListener("click", unlock);
      window.removeEventListener("keydown", unlock);
      window.removeEventListener("touchstart", unlock);
    };
    window.addEventListener("click", unlock);
    window.addEventListener("keydown", unlock);
    window.addEventListener("touchstart", unlock);

    return () => {
      window.removeEventListener("click", unlock);
      window.removeEventListener("keydown", unlock);
      window.removeEventListener("touchstart", unlock);
    };
  }, []);

  const playNext = () => {
    const audio = audioRef.current;
    if (!audio || audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }
    isPlayingRef.current = true;
    const src = audioQueueRef.current.shift()!;
    audio.src = src;
    audio.onended = playNext;
    audio.onerror = playNext;
    audio.play().catch(playNext);
  };

  useEffect(() => {
    // Detect rising edge: false → true
    const wasFired = prevFiredRef.current;
    prevFiredRef.current = fired;

    if (!fired || wasFired) return;

    const now = Date.now();
    const streak = streakRef.current;

    if (now - streak.lastZapTime < STREAK_WINDOW_MS) {
      streak.count += 1;
    } else {
      streak.count = 1;
    }
    streak.lastZapTime = now;

    // Medal triggers at streak >= 2
    if (streak.count < 2) return;

    const tierIdx = Math.min(streak.count - 2, MEDAL_TIERS.length - 1);
    const tier = MEDAL_TIERS[tierIdx];

    // Show medal — increment key to restart CSS animation
    if (medalTimeoutRef.current) clearTimeout(medalTimeoutRef.current);
    setActiveMedal(tier);
    setMedalKey((k) => k + 1);
    medalTimeoutRef.current = setTimeout(() => setActiveMedal(null), MEDAL_DISPLAY_MS);

    // Queue voice line
    audioQueueRef.current.push(tier.audio);
    if (!isPlayingRef.current) playNext();
  }, [fired]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (medalTimeoutRef.current) clearTimeout(medalTimeoutRef.current);
    };
  }, []);

  if (!activeMedal) return null;

  return (
    <div
      key={medalKey}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        pointerEvents: "none",
        zIndex: 20,
        animation:
          "medalFadeIn 0.3s ease-out forwards, medalFadeOut 0.5s ease-in 2.3s forwards",
      }}
    >
      <img
        src={activeMedal.icon}
        alt={activeMedal.name}
        width={96}
        height={96}
        style={{
          filter: "drop-shadow(0 0 12px rgba(255, 215, 0, 0.7))",
        }}
      />
      <span
        style={{
          fontFamily: "var(--font-display)",
          fontSize: 22,
          fontWeight: 800,
          textTransform: "uppercase",
          color: "#fff",
          textShadow: "0 0 8px rgba(255, 215, 0, 0.8), 0 2px 4px rgba(0,0,0,0.6)",
          marginTop: 8,
          letterSpacing: 1,
        }}
      >
        {activeMedal.name}
      </span>
    </div>
  );
}
