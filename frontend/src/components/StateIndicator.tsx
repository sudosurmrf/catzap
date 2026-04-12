const STATE_CONFIG: Record<string, { color: string; bg: string; label: string }> = {
  SWEEPING: { color: "var(--cyan)", bg: "var(--cyan-glow)", label: "SWEEP" },
  TRACKING: { color: "var(--cyan)", bg: "var(--cyan-glow)", label: "TRACK" },
  ENGAGING: { color: "var(--amber)", bg: "var(--amber-glow)", label: "ENGAGE" },
  PAUSED: { color: "var(--text-secondary)", bg: "var(--bg-elevated)", label: "PAUSED" },
  STOPPED: { color: "var(--red)", bg: "var(--red-glow)", label: "E-STOP" },
};

interface StateIndicatorProps {
  state: string;
}

export default function StateIndicator({ state }: StateIndicatorProps) {
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
        animation: state === "ENGAGING" ? "pulse 0.4s infinite" : undefined,
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: cfg.color,
          display: "inline-block",
          boxShadow: `0 0 6px ${cfg.color}`,
        }}
      />
      {cfg.label}
    </div>
  );
}
