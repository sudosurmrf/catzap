import { useEffect, useState } from "react";

interface DirectionArrowProps {
  delta: { pan: number; tilt: number } | null;
}

export default function DirectionArrow({ delta }: DirectionArrowProps) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (!delta) return;
    const interval = setInterval(() => setVisible((v) => !v), 250);
    return () => clearInterval(interval);
  }, [delta]);

  if (!delta || (Math.abs(delta.pan) < 10 && Math.abs(delta.tilt) < 10)) return null;

  const panAbs = Math.abs(delta.pan);
  const arrowSize = Math.min(60, 18 + panAbs * 0.8);
  const opacity = visible ? Math.min(0.9, 0.3 + panAbs / 80) : 0.08;

  const isRight = delta.pan > 0;

  return (
    <div
      style={{
        position: "absolute",
        top: "50%",
        [isRight ? "right" : "left"]: 12,
        transform: "translateY(-50%)",
        fontSize: arrowSize,
        color: "var(--amber)",
        opacity,
        pointerEvents: "none",
        fontWeight: "bold",
        textShadow: "0 0 16px var(--amber-dim)",
        transition: "opacity 0.15s",
        zIndex: 20,
      }}
    >
      {isRight ? "\u25B6" : "\u25C0"}
    </div>
  );
}
