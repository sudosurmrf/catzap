import { useState, useEffect } from "react";
import type { Cat } from "../types";
import { getCats, capturePhoto } from "../api/client";

interface CatLabelDropdownProps {
  bbox: number[];
  frameBase64: string;
  onLabeled: () => void;
  style?: React.CSSProperties;
}

export default function CatLabelDropdown({ bbox, frameBase64, onLabeled, style }: CatLabelDropdownProps) {
  const [cats, setCats] = useState<Cat[]>([]);
  const [open, setOpen] = useState(false);
  const [labeling, setLabeling] = useState(false);

  useEffect(() => {
    getCats().then(setCats).catch(console.error);
  }, []);

  async function handleSelect(catId: string) {
    setLabeling(true);
    try {
      await capturePhoto(catId, frameBase64, bbox);
      onLabeled();
    } catch (e) {
      console.error("Failed to label:", e);
    }
    setLabeling(false);
    setOpen(false);
  }

  if (!open) {
    return (
      <button
        onClick={(e) => { e.stopPropagation(); setOpen(true); }}
        style={{
          padding: "2px 6px",
          fontSize: 9,
          fontFamily: "var(--font-mono)",
          background: "rgba(245, 158, 11, 0.2)",
          border: "1px solid var(--amber-dim)",
          borderRadius: 3,
          color: "var(--amber)",
          cursor: "pointer",
          animation: "pulse 2s infinite",
          ...style,
        }}
      >
        Label?
      </button>
    );
  }

  return (
    <div
      onClick={(e) => e.stopPropagation()}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 2,
        background: "var(--bg-panel)",
        border: "1px solid var(--border-base)",
        borderRadius: "var(--radius-sm)",
        padding: 4,
        minWidth: 100,
        ...style,
      }}
    >
      {cats.map((cat) => (
        <button
          key={cat.id}
          onClick={() => handleSelect(cat.id)}
          disabled={labeling}
          style={{
            padding: "3px 8px",
            fontSize: 10,
            fontFamily: "var(--font-mono)",
            background: "var(--bg-deep)",
            border: "1px solid var(--border-subtle)",
            borderRadius: 3,
            color: "var(--text-secondary)",
            cursor: labeling ? "wait" : "pointer",
            textAlign: "left",
          }}
        >
          {cat.name}
        </button>
      ))}
      <button
        onClick={() => setOpen(false)}
        style={{
          padding: "2px 6px",
          fontSize: 9,
          fontFamily: "var(--font-mono)",
          background: "transparent",
          border: "none",
          color: "var(--text-ghost)",
          cursor: "pointer",
          textAlign: "center",
        }}
      >
        cancel
      </button>
    </div>
  );
}
