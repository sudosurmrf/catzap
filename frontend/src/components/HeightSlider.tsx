interface HeightSliderProps {
  heightMin: number;
  heightMax: number;
  onChangeMin: (v: number) => void;
  onChangeMax: (v: number) => void;
  maxRange?: number;
}

export default function HeightSlider({
  heightMin, heightMax, onChangeMin, onChangeMax, maxRange = 300,
}: HeightSliderProps) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 6,
      padding: "10px 12px",
      background: "var(--bg-deep)",
      borderRadius: "var(--radius-sm)",
      border: "1px solid var(--border-subtle)",
    }}>
      <div style={{
        fontFamily: "var(--font-mono)", fontSize: 10,
        color: "var(--text-tertiary)", letterSpacing: "0.05em",
        textTransform: "uppercase",
      }}>
        Height extrusion (cm)
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ flex: 1 }}>
          <div style={{
            display: "flex", justifyContent: "space-between",
            fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-tertiary)",
          }}>
            <span>Min</span>
            <span style={{ color: "var(--amber)" }}>{heightMin} cm</span>
          </div>
          <input type="range" min={0} max={maxRange} value={heightMin}
            onChange={(e) => onChangeMin(Math.min(Number(e.target.value), heightMax))} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{
            display: "flex", justifyContent: "space-between",
            fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-tertiary)",
          }}>
            <span>Max</span>
            <span style={{ color: "var(--amber)" }}>{heightMax} cm</span>
          </div>
          <input type="range" min={0} max={maxRange} value={heightMax}
            onChange={(e) => onChangeMax(Math.max(Number(e.target.value), heightMin))} />
        </div>
      </div>
      <div style={{
        height: 40, position: "relative",
        background: "var(--bg-surface)", borderRadius: 3, overflow: "hidden",
      }}>
        <div style={{
          position: "absolute",
          bottom: `${(heightMin / maxRange) * 100}%`,
          height: `${((heightMax - heightMin) / maxRange) * 100}%`,
          left: 0, right: 0,
          background: "var(--amber-glow)",
          border: "1px solid var(--amber-dim)",
          borderRadius: 2,
        }} />
      </div>
    </div>
  );
}
