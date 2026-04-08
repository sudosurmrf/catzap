export interface Zone {
  id: string;
  name: string;
  polygon: number[][];
  overlap_threshold: number;
  cooldown_seconds: number;
  enabled: boolean;
  created_at: string;
  mode: "2d" | "auto_3d" | "manual_3d";
  room_polygon: number[][] | null;
  height_min: number;
  height_max: number;
  furniture_id: string | null;
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
  cat_confidence?: number;
  track_id?: number;
}

export interface Violation {
  zone_id: string;
  zone_name: string;
  overlap: number;
}

export interface DirectionDelta {
  pan: number;
  tilt: number;
}

export interface OccludedCat {
  id: string;
  predicted: [number, number, number];
  occluded_by: string;
}

export interface FrameData {
  frame: string;
  panorama: string | null;
  detections: Detection[];
  violations: Violation[];
  fired: boolean;
  fire_target: { x: number; y: number; zone: string } | null;
  state: string;
  servo_pan: number;
  servo_tilt: number;
  warning_remaining: number;
  direction_delta: DirectionDelta | null;
  occluded_cats: OccludedCat[];
}

export interface FrameResult {
  detections: Detection[];
  violations: Violation[];
  fired: boolean;
}

export interface ZoneTransform {
  scaleX: number;   // width multiplier (1.0 = original)
  scaleY: number;   // length multiplier
  height: number;   // extrusion height in cm
  skewX: number;    // horizontal shear factor
  skewY: number;    // vertical shear factor
  slantX: number;   // top face X tilt (-1 to 1)
  slantY: number;   // top face Y tilt (-1 to 1)
}

export const DEFAULT_TRANSFORM: ZoneTransform = {
  scaleX: 1, scaleY: 1, height: 0,
  skewX: 0, skewY: 0, slantX: 0, slantY: 0,
};

export interface ControlStatus {
  armed: boolean;
  state: string;
  servo_pan: number;
  servo_tilt: number;
  dev_mode: boolean;
  paused: boolean;
  stopped: boolean;
  pause_queued: boolean;
}
