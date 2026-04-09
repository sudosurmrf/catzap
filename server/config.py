from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Dev mode — uses local webcam instead of ESP32-CAM, simulates fire instead of sending HTTP
    dev_mode: bool = False

    # ESP32 addresses (used when dev_mode=False)
    esp32_cam_url: str = "http://192.168.0.143:81/stream"
    esp32_actuator_url: str = "http://192.168.0.158"

    # Vision
    detection_model: str = "yolov8n.pt"  # yolov8n.pt (fast) | yolov8s.pt (balanced) | yolov8m.pt (accurate)
    detection_imgsz: int = 640  # input resolution for YOLO — higher catches distant cats
    confidence_threshold: float = 0.35
    classifier_confidence_threshold: float = 0.6
    classifier_uncertain_min: float = 0.3
    classify_every_n_frames: int = 30
    overlap_threshold: float = 0.3
    frame_skip_n: int = 2
    vision_loop_interval: float = 0.1  # seconds between vision loop iterations (~10 FPS)

    # Actuation
    cooldown_default: int = 3

    # Sweep / Panorama
    sweep_pan_min: float = 30.0
    sweep_pan_max: float = 150.0
    sweep_tilt_min: float = 20.0
    sweep_tilt_max: float = 70.0
    sweep_speed: float = 2.5
    fov_horizontal: float = 65.0
    fov_vertical: float = 50.0
    tile_overlap: float = 10.0
    tile_refresh_threshold: int = 15
    warning_duration: float = 1.5
    tracking_duration: float = 3.0
    reentry_warning: float = 0.5
    lock_on_grace: float = 1.0  # seconds to keep tracking after losing detection during WARNING

    # Depth / Spatial
    midas_model: str = "MiDaS_small"  # MiDaS model variant
    depth_run_interval: int = 5  # run depth every Nth tile refresh
    depth_blend_alpha: float = 0.2  # EMA blend factor for heightmap
    depth_change_threshold: float = 20.0  # cm change to flag furniture move
    heightmap_resolution: float = 5.0  # cm per cell
    room_width_cm: float = 500.0  # room dimensions for model
    room_depth_cm: float = 500.0
    room_height_cm: float = 300.0
    camera_height_cm: float = 150.0  # camera mount height from floor
    occlusion_timeout: float = 10.0  # seconds before giving up on occluded cat
    occlusion_grace_frames: int = 3  # frames before declaring cat lost (no occluder)

    # Database (PostgreSQL)
    database_url: str = "postgresql://localhost:5432/catzap"

    # Notifications
    notification_webhook_url: str = ""

    # Cat photos storage
    cat_photos_dir: Path = Path("data/cat_photos")

    # Model weights
    classifier_weights_dir: Path = Path("models/weights")

    model_config = {"env_prefix": "CATZAP_", "env_file": ".env"}


settings = Settings()
