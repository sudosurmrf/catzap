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
      const peakHour = peakHourNum ? `${peakHourNum}:00-${(parseInt(peakHourNum) + 1) % 24}:00` : "N/A";

      return {
        cat,
        zapCount: zaps.length,
        detectCount: detects.length,
        zapRate: detects.length > 0 ? zaps.length / detects.length : 0,
        favoriteZone,
        peakHour,
      };
    });

    setStats(catStats);
    setLoading(false);
  }

  if (loading) return <p style={{ color: "#888", fontFamily: "monospace", textAlign: "center" }}>Loading stats...</p>;
  if (stats.length === 0) return <p style={{ color: "#888", fontFamily: "monospace", textAlign: "center" }}>No cats registered yet. Add cats in Settings.</p>;

  const colors = ["#f72585", "#4361ee", "#4cc9f0", "#7209b7", "#f94144"];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {stats.map((stat, i) => (
        <div key={stat.cat.id} style={{ padding: 16, background: "#222", borderRadius: 8, borderLeft: `4px solid ${colors[i % colors.length]}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <span style={{ color: colors[i % colors.length], fontWeight: "bold", fontSize: 16, fontFamily: "monospace" }}>{stat.cat.name}</span>
          </div>
          <div style={{ display: "flex", gap: 12, marginBottom: 8 }}>
            <div style={{ textAlign: "center", padding: "8px 16px", background: `${colors[i % colors.length]}20`, borderRadius: 6 }}>
              <div style={{ color: colors[i % colors.length], fontSize: 24, fontWeight: "bold", fontFamily: "monospace" }}>{stat.zapCount}</div>
              <div style={{ color: "#888", fontSize: 11, fontFamily: "monospace" }}>Zaps</div>
            </div>
            <div style={{ textAlign: "center", padding: "8px 16px", background: "rgba(76, 201, 240, 0.1)", borderRadius: 6 }}>
              <div style={{ color: "#4cc9f0", fontSize: 24, fontWeight: "bold", fontFamily: "monospace" }}>{stat.detectCount}</div>
              <div style={{ color: "#888", fontSize: 11, fontFamily: "monospace" }}>Detections</div>
            </div>
            <div style={{ textAlign: "center", padding: "8px 16px", background: "rgba(255,255,255,0.05)", borderRadius: 6 }}>
              <div style={{ color: "#ccc", fontSize: 24, fontWeight: "bold", fontFamily: "monospace" }}>{Math.round(stat.zapRate * 100)}%</div>
              <div style={{ color: "#888", fontSize: 11, fontFamily: "monospace" }}>Zap Rate</div>
            </div>
          </div>
          <div style={{ color: "#888", fontSize: 12, fontFamily: "monospace" }}>
            Peak mischief: {stat.peakHour} | Favorite zone: {stat.favoriteZone}
          </div>
        </div>
      ))}
    </div>
  );
}
