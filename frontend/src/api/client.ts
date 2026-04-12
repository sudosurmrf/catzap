import type {
  Zone, Furniture, Cat, CatEvent, ControlStatus,
  CalibrationStartResponse, CalibrationJogResponse, CalibrationSetHomeResponse,
  BeginExtentResponse, RecordExtentCornerResponse, ComputeTileGridResponse,
  ExtentCornerLabel,
  VerificationTargetResponse, VerificationConfirmResponse,
  RigSettings, JogDirection, JogStep,
} from "../types";

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
  enabled?: boolean;
}) => fetchJSON<Zone>("/zones", { method: "POST", body: JSON.stringify(zone) });

export const updateZone = (id: string, updates: {
  name?: string;
  polygon?: number[][];
  overlap_threshold?: number;
  enabled?: boolean;
}) =>
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

// Furniture
export const getFurniture = () => fetchJSON<Furniture[]>("/furniture");

export const createFurniture = (name: string, polygon: number[][]) =>
  fetchJSON<Furniture>("/furniture", { method: "POST", body: JSON.stringify({ name, polygon }) });

export const deleteFurniture = (id: string) =>
  fetchJSON(`/furniture/${id}`, { method: "DELETE" });

// ── Aim Calibration (9-point pixel→servo) ──

export const startAimCalibration = () =>
  fetchJSON<CalibrationStartResponse>("/calibration/start", { method: "POST" });

export const jogCalibration = (direction: JogDirection, step: JogStep = "coarse") =>
  fetchJSON<CalibrationJogResponse>("/calibration/jog", {
    method: "POST",
    body: JSON.stringify({ direction, step }),
  });

export const setAimCalibrationHome = () =>
  fetchJSON<CalibrationSetHomeResponse>("/calibration/set-home", { method: "POST" });

export const beginExtentCapture = () =>
  fetchJSON<BeginExtentResponse>("/calibration/begin-extent", { method: "POST" });

export const recordExtentCorner = (label: ExtentCornerLabel) =>
  fetchJSON<RecordExtentCornerResponse>("/calibration/record-extent-corner", {
    method: "POST",
    body: JSON.stringify({ label }),
  });

export const computeTileGrid = () =>
  fetchJSON<ComputeTileGridResponse>("/calibration/compute-tile-grid", {
    method: "POST",
  });

export const startVerification = () =>
  fetchJSON<VerificationTargetResponse>("/calibration/start-verification", {
    method: "POST",
  });

export const confirmVerification = () =>
  fetchJSON<VerificationConfirmResponse>("/calibration/confirm-verification", {
    method: "POST",
  });

export const skipVerification = () =>
  fetchJSON<{ phase: string }>("/calibration/skip-verification", { method: "POST" });

export const finalizeAimCalibration = () =>
  fetchJSON<{ status: string }>("/calibration/finalize", { method: "POST" });

export const cancelAimCalibration = () =>
  fetchJSON<{ cancelled: boolean }>("/calibration/cancel", { method: "POST" });

export const getAimCalibrationStatus = () =>
  fetchJSON<any>("/calibration/status");

export const deleteAimCalibration = () =>
  fetchJSON<{ deleted: boolean }>("/calibration", { method: "DELETE" });

export const getRigSettings = () =>
  fetchJSON<RigSettings>("/calibration/rig-settings");

export const updateRigSettings = (updates: Partial<RigSettings>) =>
  fetchJSON<RigSettings>("/calibration/rig-settings", {
    method: "POST",
    body: JSON.stringify(updates),
  });

export const autoLevelPanAxis = (points: { pan: number; tilt: number }[]) =>
  fetchJSON<{
    pan_tilt_poly: number[];
    home_pan: number;
    home_tilt: number;
    num_points: number;
    per_point_residuals: number[];
    max_residual: number;
    rms_residual: number;
    rig_settings: RigSettings;
  }>("/calibration/auto-level", {
    method: "POST",
    body: JSON.stringify({ points }),
  });

export const testLevelSweep = () =>
  fetchJSON<{
    status: string;
    span: number;
    home_pan: number;
    home_tilt: number;
  }>("/calibration/test-level-sweep", { method: "POST" });

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
