import type { Zone, Cat, CatEvent, ControlStatus } from "../types";

const API_BASE = "/api";

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

// Zones
export const getZones = () => fetchJSON<Zone[]>("/zones");

export const createZone = (zone: {
  name: string;
  polygon: number[][];
  overlap_threshold?: number;
  cooldown_seconds?: number;
}) => fetchJSON<Zone>("/zones", { method: "POST", body: JSON.stringify(zone) });

export const updateZone = (id: string, updates: Partial<Zone>) =>
  fetchJSON<Zone>(`/zones/${id}`, {
    method: "PUT",
    body: JSON.stringify(updates),
  });

export const deleteZone = (id: string) =>
  fetchJSON(`/zones/${id}`, { method: "DELETE" });

// Cats
export const getCats = () => fetchJSON<Cat[]>("/cats");

export const createCat = (name: string) =>
  fetchJSON<Cat>("/cats", {
    method: "POST",
    body: JSON.stringify({ name }),
  });

export const deleteCat = (id: string) =>
  fetchJSON(`/cats/${id}`, { method: "DELETE" });

// Events
export const getEvents = (params?: {
  type?: string;
  cat_name?: string;
  zone_name?: string;
  limit?: number;
  offset?: number;
}) => {
  const searchParams = new URLSearchParams();
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) searchParams.set(key, String(value));
    });
  }
  const query = searchParams.toString();
  return fetchJSON<CatEvent[]>(`/events${query ? `?${query}` : ""}`);
};

// Control
export const getControlStatus = () =>
  fetchJSON<ControlStatus>("/control/status");

export const setArmed = (armed: boolean) =>
  fetchJSON<{ armed: boolean }>("/control/arm", {
    method: "POST",
    body: JSON.stringify({ armed }),
  });

export const manualFire = (pan: number, tilt: number, duration_ms?: number) =>
  fetchJSON("/control/fire", {
    method: "POST",
    body: JSON.stringify({ pan, tilt, duration_ms: duration_ms ?? 200 }),
  });

export const addCalibrationPoint = (
  pixel_x: number,
  pixel_y: number,
  pan_angle: number,
  tilt_angle: number
) =>
  fetchJSON<{ points_count: number }>("/control/calibrate", {
    method: "POST",
    body: JSON.stringify({ pixel_x, pixel_y, pan_angle, tilt_angle }),
  });

export const clearCalibration = () =>
  fetchJSON("/control/calibrate", { method: "DELETE" });

// WebSocket
export function connectEventSocket(
  onEvent: (event: CatEvent) => void
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/events`);
  ws.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    onEvent(event);
  };
  return ws;
}
