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
    confidence_threshold: float = 0.25 
    classifier_confidence_threshold: float = 0.6
    classifier_uncertain_min: float = 0.3
    classify_every_n_frames: int = 30
    overlap_threshold: float = 0.3
    frame_skip_n: int = 2  # submit every 2nd frame; detection smoother bridges the gaps
    vision_loop_interval: float = 0.1  # seconds between vision loop iterations (~10 FPS)

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

    # Engagement
    min_shot_interval_ms: int = 2000  # minimum time between successive shots while in ENGAGING
    engagement_grace_ms: int = 3000   # time with no cat-in-zone before ENGAGING → SWEEPING

    # Tracking deadband: if the cat's bbox center is within this radius of the
    # frame center (normalized [0..1] pixel space), the camera holds position
    # instead of issuing a new tracking jump. Absorbs small overshoots and
    # YOLO bbox jitter — without it, the servo would buzz on every frame as
    # the bbox center wobbles a few pixels. 0.15 ≈ ±15% of frame radius from
    # center (about ±10° in pan with a 65° FOV). Engagement (zone violations)
    # bypasses the deadband — we always lock on tightly before firing.
    tracking_deadband_frac: float = 0.15

    # Database (PostgreSQL)
    database_url: str = "postgresql://localhost:5432/catzap"

    # Notifications
    notification_webhook_url: str = ""

    # Cat photos storage
    cat_photos_dir: Path = Path("data/cat_photos")

    # Model weights
    classifier_weights_dir: Path = Path("models/weights")

    model_config = {"env_prefix": "CATZAP_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
