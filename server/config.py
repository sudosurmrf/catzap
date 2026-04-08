from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Dev mode — uses local webcam instead of ESP32-CAM, simulates fire instead of sending HTTP
    dev_mode: bool = True

    # ESP32 addresses (used when dev_mode=False)
    esp32_cam_url: str = "http://192.168.1.100:81/stream"
    esp32_actuator_url: str = "http://192.168.1.101"

    # Vision
    confidence_threshold: float = 0.5
    overlap_threshold: float = 0.3
    frame_skip_n: int = 2

    # Actuation
    cooldown_default: int = 3

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
