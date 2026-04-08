const STATE_CONFIG: Record<string, { color: string; bg: string; label: string }> = {
  SWEEPING: { color: "var(--cyan)", bg: "var(--cyan-glow)", label: "SWEEP" },
  WARNING: { color: "var(--amber)", bg: "var(--amber-glow)", label: "WARN" },
  FIRING: { color: "var(--red)", bg: "var(--red-glow)", label: "FIRE" },
  TRACKING: { color: "var(--purple)", bg: "var(--purple-dim)", label: "TRACK" },
  PAUSED: { color: "var(--text-secondary)", bg: "var(--bg-elevated)", label: "PAUSED" },
  STOPPED: { color: "var(--red)", bg: "var(--red-glow)", label: "E-STOP" },
};

interface StateIndicatorProps {
  state: string;
  warningRemaining: number;
}

export default function StateIndicator({ state, warningRemaining }: StateIndicatorProps) {
  const cfg = STATE_CONFIG[state] || { color: "var(--text-tertiary)", bg: "var(--bg-elevated)", label: state };

  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        padding: "3px 10px",
        background: cfg.bg,
        borderRadius: 4,
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        fontWeight: 600,
        color: cfg.color,
        animation: state === "FIRING" ? "pulse 0.4s infinite" : undefined,
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: cfg.color,
          display: "inline-block",
          animation: state === "WARNING" ? "blink 0.5s infinite" : undefined,
          boxShadow: `0 0 6px ${cfg.color}`,
        }}
      />
      {cfg.label}
      {state === "WARNING" && warningRemaining > 0 && (
        <span style={{ fontWeight: 400, opacity: 0.8 }}>
          {warningRemaining.toFixed(1)}s
        </span>
      )}
    </div>
  );
}
