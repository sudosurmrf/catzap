import { useEffect, useState } from "react";
import type { CatEvent } from "../types";
import { getEvents, connectEventSocket } from "../api/client";

const EVENT_STYLES: Record<string, { color: string; icon: string }> = {
  ZAP: { color: "var(--red)", icon: "⚡" },
  DETECT_ENTER: { color: "var(--cyan)", icon: "→" },
  DETECT_EXIT: { color: "var(--text-tertiary)", icon: "←" },
  SYSTEM: { color: "var(--purple)", icon: "●" },
};

export default function EventLog() {
  const [events, setEvents] = useState<CatEvent[]>([]);
  const [filter, setFilter] = useState({ type: "", cat_name: "" });

  useEffect(() => {
    loadEvents();
    const ws = connectEventSocket((event) => {
      setEvents((prev) => [event, ...prev].slice(0, 25));
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
    setEvents(data.slice(0, 25));
  }

  function formatTime(ts: string) {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function eventMessage(event: CatEvent): string {
    switch (event.type) {
      case "ZAP":
        return `${event.cat_name || "Cat"} zapped in ${event.zone_name || "zone"}`;
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
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {/* Filters */}
      <div style={{ display: "flex", gap: 6 }}>
        <select
          value={filter.type}
          onChange={(e) => setFilter({ ...filter, type: e.target.value })}
          style={{
            padding: "5px 8px",
            background: "var(--bg-deep)",
            border: "1px solid var(--border-base)",
            borderRadius: "var(--radius-sm)",
            color: "var(--text-secondary)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
          }}
        >
          <option value="">All</option>
          <option value="ZAP">Zaps</option>
          <option value="DETECT_ENTER">Enter</option>
          <option value="SYSTEM">System</option>
        </select>
        <input
          type="text"
          placeholder="Filter cat..."
          value={filter.cat_name}
          onChange={(e) => setFilter({ ...filter, cat_name: e.target.value })}
          style={{ flex: 1, fontSize: 11 }}
        />
      </div>

      {/* Event list */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {events.length === 0 && (
          <div style={{
            textAlign: "center",
            padding: "32px 16px",
            color: "var(--text-ghost)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
          }}>
            Waiting for activity...
          </div>
        )}

        {events.map((event, i) => {
          const style = EVENT_STYLES[event.type] || { color: "var(--text-tertiary)", icon: "·" };
          return (
            <div
              key={event.id || `ws-${event.timestamp}-${i}`}
              style={{
                display: "flex",
                alignItems: "flex-start",
                gap: 8,
                padding: "7px 10px",
                background: i === 0 ? "var(--bg-elevated)" : "transparent",
                borderRadius: "var(--radius-sm)",
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                animation: i === 0 ? "fadeIn 0.3s ease" : undefined,
                transition: "background 0.2s",
              }}
            >
              <span style={{
                color: style.color,
                flexShrink: 0,
                width: 14,
                textAlign: "center",
                fontSize: 12,
              }}>
                {style.icon}
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  color: "var(--text-secondary)",
                  lineHeight: 1.4,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}>
                  {eventMessage(event)}
                </div>
                <div style={{
                  display: "flex",
                  gap: 8,
                  marginTop: 1,
                  color: "var(--text-ghost)",
                  fontSize: 10,
                }}>
                  <span>{formatTime(event.timestamp)}</span>
                  {event.confidence != null && (
                    <span>{Math.round(event.confidence * 100)}%</span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
