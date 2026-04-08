import { useEffect, useState } from "react";
import type { CatEvent } from "../types";
import { getEvents, connectEventSocket } from "../api/client";

const EVENT_COLORS: Record<string, string> = {
  ZAP: "#f94144",
  DETECT_ENTER: "#4cc9f0",
  DETECT_EXIT: "#4cc9f0",
  SYSTEM: "#7209b7",
};

export default function EventLog() {
  const [events, setEvents] = useState<CatEvent[]>([]);
  const [filter, setFilter] = useState({ type: "", cat_name: "" });

  useEffect(() => {
    loadEvents();
    const ws = connectEventSocket((event) => {
      setEvents((prev) => [event, ...prev].slice(0, 200));
    });
    return () => ws.close();
  }, []);

  useEffect(() => {
    loadEvents();
  }, [filter]);

  async function loadEvents() {
    const params: Record<string, string> = {};
    if (filter.type) params.type = filter.type;
    if (filter.cat_name) params.cat_name = filter.cat_name;
    const data = await getEvents(params);
    setEvents(data);
  }

  function formatTime(ts: string) {
    return new Date(ts).toLocaleTimeString();
  }

  function eventMessage(event: CatEvent): string {
    switch (event.type) {
      case "ZAP":
        return `${event.cat_name || "Cat"} on ${event.zone_name || "zone"} — ZAPPED!`;
      case "DETECT_ENTER":
        return `${event.cat_name || "Cat"} entered ${event.zone_name || "zone"}`;
      case "DETECT_EXIT":
        return `${event.cat_name || "Cat"} left ${event.zone_name || "zone"}`;
      case "SYSTEM":
        return event.zone_name || "System event";
      default:
        return "Unknown event";
    }
  }

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <select
          value={filter.type}
          onChange={(e) => setFilter({ ...filter, type: e.target.value })}
          style={{ padding: "8px 12px", background: "#333", border: "1px solid #555", borderRadius: 6, color: "#ccc", fontFamily: "monospace" }}
        >
          <option value="">All Types</option>
          <option value="ZAP">Zaps</option>
          <option value="DETECT_ENTER">Detections</option>
          <option value="SYSTEM">System</option>
        </select>
        <input
          type="text"
          placeholder="Filter by cat name..."
          value={filter.cat_name}
          onChange={(e) => setFilter({ ...filter, cat_name: e.target.value })}
          style={{ padding: "8px 12px", background: "#333", border: "1px solid #555", borderRadius: 6, color: "#ccc", fontFamily: "monospace", flex: 1 }}
        />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {events.length === 0 && (
          <p style={{ color: "#888", fontFamily: "monospace", textAlign: "center", padding: 32 }}>
            No events yet. Waiting for cat activity...
          </p>
        )}
        {events.map((event) => (
          <div
            key={event.id}
            style={{
              display: "flex",
              gap: 12,
              padding: "8px 12px",
              background: `${EVENT_COLORS[event.type] || "#333"}15`,
              borderLeft: `3px solid ${EVENT_COLORS[event.type] || "#333"}`,
              borderRadius: 4,
              fontFamily: "monospace",
              fontSize: 12,
              alignItems: "center",
            }}
          >
            <span style={{ color: "#888", minWidth: 70 }}>{formatTime(event.timestamp)}</span>
            <span style={{ color: EVENT_COLORS[event.type] || "#ccc", minWidth: 50, fontWeight: "bold" }}>
              {event.type}
            </span>
            <span style={{ color: "#ccc", flex: 1 }}>{eventMessage(event)}</span>
            {event.confidence && (
              <span style={{ color: "#888" }}>{Math.round(event.confidence * 100)}%</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
