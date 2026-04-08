export interface Zone {
  id: string;
  name: string;
  polygon: number[][];
  overlap_threshold: number;
  cooldown_seconds: number;
  enabled: boolean;
  created_at: string;
}

export interface Cat {
  id: string;
  name: string;
  model_version: number;
  created_at: string;
}

export interface CatEvent {
  id: string;
  type: "ZAP" | "DETECT_ENTER" | "DETECT_EXIT" | "SYSTEM";
  cat_id: string | null;
  cat_name: string | null;
  zone_id: string | null;
  zone_name: string | null;
  confidence: number | null;
  overlap: number | null;
  servo_pan: number | null;
  servo_tilt: number | null;
  timestamp: string;
}

export interface Detection {
  bbox: number[];
  confidence: number;
  cat_name?: string;
}

export interface Violation {
  zone_id: string;
  zone_name: string;
  overlap: number;
}

export interface FrameResult {
  detections: Detection[];
  violations: Violation[];
  fired: boolean;
}

export interface ControlStatus {
  armed: boolean;
  calibration_points: number;
}
