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
  mode?: string;
  room_polygon?: number[][];
  height_min?: number;
  height_max?: number;
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

// Cat Photos
export const getCatPhotos = (catId: string) =>
  fetchJSON<{ id: string; file_path: string; source: string; created_at: string }[]>(
    `/cats/${catId}/photos`
  );

export const capturePhoto = (catId: string, frameBase64: string, bbox: number[]) =>
  fetch(`${API_BASE}/cats/${catId}/photos/capture`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frame_base64: frameBase64, bbox }),
  }).then((r) => r.json());

export const uploadPhotos = (catId: string, files: FileList) => {
  const form = new FormData();
  Array.from(files).forEach((f) => form.append("files", f));
  return fetch(`${API_BASE}/cats/${catId}/photos/upload`, {
    method: "POST",
    body: form,
  }).then((r) => r.json());
};

export const deleteCatPhoto = (catId: string, photoId: string) =>
  fetchJSON(`/cats/${catId}/photos/${photoId}`, { method: "DELETE" });

// Classifier
export const startTraining = () =>
  fetchJSON<{ status: string }>("/classifier/train", { method: "POST" });

export const getTrainingStatus = () =>
  fetchJSON<{ state: string; progress: number; accuracy: number; error: string | null }>(
    "/classifier/status"
  );

export const getClassifierInfo = () =>
  fetchJSON<{
    model_exists: boolean;
    per_cat: { name: string; photo_count: number }[];
    min_photos_required: number;
  }>("/classifier/info");

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

export const setVirtualAngle = (pan: number, tilt: number) =>
  fetchJSON("/control/virtual-angle", {
    method: "POST",
    body: JSON.stringify({ pan, tilt }),
  });

export const startCalibrationSweep = () =>
  fetchJSON("/control/calibration-sweep", { method: "POST" });

export const togglePause = () =>
  fetchJSON<{ paused: boolean }>("/control/pause", { method: "POST" });

export const emergencyStop = () =>
  fetchJSON<{ stopped: boolean }>("/control/emergency-stop", { method: "POST" });

export const clearEmergencyStop = () =>
  fetchJSON<{ stopped: boolean; armed: boolean }>("/control/clear-estop", { method: "POST" });

// Spatial / Calibration
export const getFurniture = () => fetchJSON<any[]>("/spatial/furniture");

export const createFurniture = (furniture: {
  name: string;
  base_polygon: number[][];
  height_min: number;
  height_max: number;
}) => fetchJSON("/spatial/furniture", { method: "POST", body: JSON.stringify(furniture) });

export const updateFurniture = (id: string, updates: {
  name?: string;
  base_polygon?: number[][];
  height_min?: number;
  height_max?: number;
}) => fetchJSON(`/spatial/furniture/${id}`, { method: "PUT", body: JSON.stringify(updates) });

export const deleteFurniture = (id: string) =>
  fetchJSON(`/spatial/furniture/${id}`, { method: "DELETE" });

export const estimateHeight = (polygon: number[][]) =>
  fetchJSON<{ height_min: number; height_max: number; estimated: boolean; reason?: string }>(
    "/spatial/estimate-height", { method: "POST", body: JSON.stringify({ polygon }) }
  );

export const getRoomModelStatus = () =>
  fetchJSON<{ initialized: boolean; width_cm?: number; depth_cm?: number; furniture_count?: number; depth_scale?: number }>("/spatial/room-model/status");

export const calibrateDepthScale = (realDistanceCm: number) =>
  fetchJSON("/spatial/calibrate-scale", {
    method: "POST",
    body: JSON.stringify({ real_distance_cm: realDistanceCm }),
  });

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
