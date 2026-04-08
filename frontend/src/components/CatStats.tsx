import { useEffect, useState } from "react";
import type { Cat } from "../types";
import { getCats, getEvents } from "../api/client";

interface CatStat {
  cat: Cat;
  zapCount: number;
  detectCount: number;
  zapRate: number;
  favoriteZone: string;
  peakHour: string;
}

const ACCENT_COLORS = ["var(--amber)", "var(--cyan)", "var(--purple)", "var(--red)", "var(--green)"];

export default function CatStats() {
  const [stats, setStats] = useState<CatStat[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  async function loadStats() {
    setLoading(true);
    const [cats, events] = await Promise.all([getCats(), getEvents({ limit: 1000 })]);

    const catStats: CatStat[] = cats.map((cat) => {
      const catEvents = events.filter((e) => e.cat_name === cat.name);
      const zaps = catEvents.filter((e) => e.type === "ZAP");
      const detects = catEvents.filter((e) => e.type === "ZAP" || e.type === "DETECT_ENTER");

      const zoneCounts: Record<string, number> = {};
      zaps.forEach((e) => {
        if (e.zone_name) zoneCounts[e.zone_name] = (zoneCounts[e.zone_name] || 0) + 1;
      });
      const favoriteZone = Object.entries(zoneCounts).sort(([, a], [, b]) => b - a)[0]?.[0] || "None";

      const hourCounts: Record<number, number> = {};
      catEvents.forEach((e) => {
        const hour = new Date(e.timestamp).getHours();
        hourCounts[hour] = (hourCounts[hour] || 0) + 1;
      });
      const peakHourNum = Object.entries(hourCounts).sort(([, a], [, b]) => b - a)[0]?.[0];
      const peakHour = peakHourNum ? `${peakHourNum}:00` : "N/A";

      return { cat, zapCount: zaps.length, detectCount: detects.length, zapRate: detects.length > 0 ? zaps.length / detects.length : 0, favoriteZone, peakHour };
    });

    setStats(catStats);
    setLoading(false);
  }

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: 32, color: "var(--text-ghost)", fontFamily: "var(--font-mono)", fontSize: 12 }}>
        Loading...
      </div>
    );
  }

  if (stats.length === 0) {
    return (
      <div style={{ textAlign: "center", padding: 32, color: "var(--text-ghost)", fontFamily: "var(--font-mono)", fontSize: 12 }}>
        No cats registered yet
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {stats.map((stat, i) => {
        const accent = ACCENT_COLORS[i % ACCENT_COLORS.length];
        return (
          <div key={stat.cat.id} className="glass-panel-solid" style={{ padding: 14, overflow: "hidden" }}>
            {/* Cat name */}
            <div style={{
              fontFamily: "var(--font-display)",
              fontSize: 14,
              fontWeight: 700,
              color: accent,
              marginBottom: 10,
              letterSpacing: "0.02em",
            }}>
              {stat.cat.name}
            </div>

            {/* Stat row */}
            <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
              <div className="stat-card" style={{ flex: 1 }}>
                <div className="stat-value" style={{ color: accent }}>{stat.zapCount}</div>
                <div className="stat-label">Zaps</div>
              </div>
              <div className="stat-card" style={{ flex: 1 }}>
                <div className="stat-value" style={{ color: "var(--cyan)" }}>{stat.detectCount}</div>
                <div className="stat-label">Detects</div>
              </div>
              <div className="stat-card" style={{ flex: 1 }}>
                <div className="stat-value" style={{ color: "var(--text-secondary)" }}>{Math.round(stat.zapRate * 100)}%</div>
                <div className="stat-label">Rate</div>
              </div>
            </div>

            {/* Meta */}
            <div style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--text-ghost)",
              display: "flex",
              gap: 12,
            }}>
              <span>Peak: {stat.peakHour}</span>
              <span>Zone: {stat.favoriteZone}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
