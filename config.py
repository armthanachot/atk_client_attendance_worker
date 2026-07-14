from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def read_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def read_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def read_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


@dataclass(frozen=True)
class DistanceConfig:
    gate_enabled: bool
    known_face_width_cm: float
    camera_focal_length_px: float
    min_distance_cm: float
    max_distance_cm: float

    @property
    def can_estimate(self) -> bool:
        return self.known_face_width_cm > 0 and self.camera_focal_length_px > 0


@dataclass(frozen=True)
class FaceDetectorConfig:
    detector_type: str
    yunet_model_path: Path
    yunet_score_threshold: float
    yunet_nms_threshold: float
    yunet_top_k: int


@dataclass(frozen=True)
class Config:
    backend_url: str
    api_key: str
    camera_id: str
    direction: str
    camera_index: int
    request_interval_seconds: float
    min_face_size: int
    crop_padding_ratio: float
    jpeg_quality: int
    display_preview: bool
    face_detector: FaceDetectorConfig
    distance: DistanceConfig

    @property
    def recognize_url(self) -> str:
        return f"{self.backend_url.rstrip('/')}/api/client-attendance/recognize"


def load_config() -> Config:
    load_env_file(ROOT / ".env")
    direction = os.environ.get("ATTENDANCE_DIRECTION", "entry").strip()
    if direction not in {"entry", "exit", "sighting"}:
        raise RuntimeError("ATTENDANCE_DIRECTION must be entry, exit, or sighting")

    distance = DistanceConfig(
        gate_enabled=read_bool("ATTENDANCE_DISTANCE_GATE_ENABLED", True),
        known_face_width_cm=read_float("ATTENDANCE_FACE_KNOWN_WIDTH_CM", 15.0),
        camera_focal_length_px=read_float("ATTENDANCE_CAMERA_FOCAL_LENGTH_PX", 600.0),
        min_distance_cm=read_float("ATTENDANCE_MIN_DISTANCE_CM", 40.0),
        max_distance_cm=read_float("ATTENDANCE_MAX_DISTANCE_CM", 120.0),
    )
    if distance.gate_enabled and not distance.can_estimate:
        raise RuntimeError(
            "Distance gate needs positive ATTENDANCE_FACE_KNOWN_WIDTH_CM "
            "and ATTENDANCE_CAMERA_FOCAL_LENGTH_PX",
        )
    if distance.min_distance_cm >= distance.max_distance_cm:
        raise RuntimeError(
            "ATTENDANCE_MIN_DISTANCE_CM must be less than ATTENDANCE_MAX_DISTANCE_CM",
        )

    face_detector_type = os.environ.get("ATTENDANCE_FACE_DETECTOR", "yunet").strip()
    if face_detector_type not in {"yunet", "haar"}:
        raise RuntimeError("ATTENDANCE_FACE_DETECTOR must be yunet or haar")

    yunet_model_path = Path(
        os.environ.get(
            "ATTENDANCE_YUNET_MODEL_PATH",
            "models/face_detection_yunet_2023mar.onnx",
        ),
    )
    if not yunet_model_path.is_absolute():
        yunet_model_path = ROOT / yunet_model_path

    return Config(
        backend_url=read_required("ATTENDANCE_BACKEND_URL"),
        api_key=read_required("ATTENDANCE_API_KEY"),
        camera_id=read_required("ATTENDANCE_CAMERA_ID"),
        direction=direction,
        camera_index=read_int("ATTENDANCE_CAMERA_INDEX", 0),
        request_interval_seconds=read_float(
            "ATTENDANCE_REQUEST_INTERVAL_SECONDS",
            3.0,
        ),
        min_face_size=read_int("ATTENDANCE_MIN_FACE_SIZE", 90),
        crop_padding_ratio=read_float("ATTENDANCE_CROP_PADDING_RATIO", 0.35),
        jpeg_quality=read_int("ATTENDANCE_JPEG_QUALITY", 85),
        display_preview=read_bool("ATTENDANCE_DISPLAY_PREVIEW", True),
        face_detector=FaceDetectorConfig(
            detector_type=face_detector_type,
            yunet_model_path=yunet_model_path,
            yunet_score_threshold=read_float("ATTENDANCE_YUNET_SCORE_THRESHOLD", 0.8),
            yunet_nms_threshold=read_float("ATTENDANCE_YUNET_NMS_THRESHOLD", 0.3),
            yunet_top_k=read_int("ATTENDANCE_YUNET_TOP_K", 5000),
        ),
        distance=distance,
    )
