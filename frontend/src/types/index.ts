export interface Zone {
  id: string;
  name: string;
  polygon: number[][];  // [pan, tilt] pairs in servo degrees
  overlap_threshold: number;
  enabled: boolean;
  created_at: string;
}

export interface Furniture {
  id: string;
  name: string;
  polygon: number[][];  // [pan, tilt] pairs in servo degrees
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
  direction_delta: DirectionDelta | null;
}

export interface FrameResult {
  detections: Detection[];
  violations: Violation[];
  fired: boolean;
}

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

// ── Aim calibration (extent-based) ──
export type CalibrationPhase =
  | "jogging_to_home"
  | "leveling"
  | "capturing_extent"
  | "extent_ready"
  | "verifying"
  | "complete";

export type JogDirection = "left" | "right" | "up" | "down";
export type JogStep = "coarse" | "fine";
export type ExtentCornerLabel = "bl" | "tl" | "tr" | "br";

export interface CalibrationPose {
  pan: number;
  tilt: number;
}

export interface ExtentCorner {
  label: ExtentCornerLabel;
  servo_pan: number;
  servo_tilt: number;
}

export interface ExtentBounds {
  pan_min: number;
  pan_max: number;
  tilt_min: number;
  tilt_max: number;
}

export interface CalibrationStartResponse {
  phase: CalibrationPhase;
  current_pose: CalibrationPose;
}

export interface CalibrationJogResponse {
  pan: number;
  tilt: number;
}

export interface CalibrationSetHomeResponse {
  phase: CalibrationPhase;
  home_pose: CalibrationPose;
  reference_frame_b64: string;
}

export interface BeginExtentResponse {
  phase: CalibrationPhase;
  corner_labels: ExtentCornerLabel[];
  recorded_corners: Record<string, ExtentCorner>;
}

export interface RecordExtentCornerResponse {
  recorded: ExtentCorner;
  recorded_corners: Record<string, ExtentCorner>;
  all_recorded: boolean;
}

export interface ComputeTileGridResponse {
  phase: CalibrationPhase;
  bounds: ExtentBounds;
  tile_cols: number;
  tile_rows: number;
  total_tiles: number;
  fov_h: number;
  fov_v: number;
}

export interface VerificationTargetResponse {
  phase?: CalibrationPhase;
  complete: boolean;
  current_index: number;
  total: number;
  tile_col: number;
  tile_row: number;
  expected_pan: number;
  expected_tilt: number;
}

export interface VerificationConfirmResponse {
  complete: boolean;
  last_residual: number;
  current_index?: number;
  total?: number;
  tile_col?: number;
  tile_row?: number;
  expected_pan?: number;
  expected_tilt?: number;
  max_residual?: number;
  mean_residual?: number;
  threshold?: number;
  passed?: boolean;
}

// ── Zone polygon transform (used by the 2D gizmo in the zone editor) ──
export interface ZoneTransform {
  scaleX: number;
  scaleY: number;
  skewX: number;
  skewY: number;
}

export const DEFAULT_TRANSFORM: ZoneTransform = {
  scaleX: 1,
  scaleY: 1,
  skewX: 0,
  skewY: 0,
};

export interface RigSettings {
  tilt_jog_inverted: boolean;
  pan_jog_inverted: boolean;
  // Quadratic tilt-compensation polynomial coefficients [a, b, c]:
  //     delta_tilt(pan) = a + b*dp + c*dp²    where dp = pan - CALIBRATION_HOME_PAN
  // Populated by /auto-level; all-zero means compensation disabled.
  pan_tilt_poly: number[];
  // Extent bounds set by the 4-corner capture phase. null until the first
  // successful calibration run that writes them. The ActuatorClient clamps
  // all outgoing commands to these bounds.
  pan_min: number | null;
  pan_max: number | null;
  tilt_min: number | null;
  tilt_max: number | null;
}
