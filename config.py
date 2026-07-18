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


def validate_liveness_config(config: "LivenessConfig") -> None:
    if not 0.0 <= config.threshold <= 1.0:
        raise RuntimeError("ATTENDANCE_LIVENESS_THRESHOLD must be between 0 and 1")
    if config.min_frames < 1:
        raise RuntimeError("ATTENDANCE_LIVENESS_MIN_FRAMES must be at least 1")
    if config.pass_ttl_seconds <= 0:
        raise RuntimeError("ATTENDANCE_LIVENESS_PASS_TTL_SECONDS must be positive")
    if config.precheck_extra_cm < 0:
        raise RuntimeError("ATTENDANCE_LIVENESS_PRECHECK_EXTRA_CM cannot be negative")
    if config.min_face_size < 1:
        raise RuntimeError("ATTENDANCE_LIVENESS_MIN_FACE_SIZE must be positive")
    if config.crop_padding_ratio < 0:
        raise RuntimeError("ATTENDANCE_LIVENESS_CROP_PADDING_RATIO cannot be negative")
    if config.input_width < 1 or config.input_height < 1:
        raise RuntimeError("ATTENDANCE_LIVENESS_INPUT_WIDTH/HEIGHT must be positive")
    if not config.torch_device:
        raise RuntimeError("ATTENDANCE_LIVENESS_TORCH_DEVICE cannot be empty")
    if config.live_class_index != 1:
        raise RuntimeError(
            "MiniFASNet real-face class is 1; "
            "ATTENDANCE_LIVENESS_LIVE_CLASS_INDEX must be 1",
        )
    if config.track_max_gap_seconds <= 0:
        raise RuntimeError("ATTENDANCE_LIVENESS_TRACK_MAX_GAP_SECONDS must be positive")
    if not 0.0 <= config.screen_risk_threshold <= 1.0:
        raise RuntimeError(
            "ATTENDANCE_LIVENESS_SCREEN_RISK_THRESHOLD must be between 0 and 1",
        )
    if config.motion_min_observations < 1:
        raise RuntimeError(
            "ATTENDANCE_LIVENESS_MOTION_MIN_OBSERVATIONS must be at least 1",
        )


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
class LivenessConfig:
    enabled: bool
    precheck_enabled: bool
    precheck_extra_cm: float
    pass_ttl_seconds: float
    min_frames: int
    threshold: float
    min_face_size: int
    crop_padding_ratio: float
    screen_check_enabled: bool
    model_path: Path
    v1se_model_path: Path
    model_proto_path: Path | None
    input_width: int
    input_height: int
    scale: float
    mean_b: float
    mean_g: float
    mean_r: float
    swap_rb: bool
    live_class_index: int
    torch_device: str
    track_max_gap_seconds: float
    screen_risk_threshold: float
    motion_check_enabled: bool
    motion_min_observations: int

    @property
    def input_size(self) -> tuple[int, int]:
        return (self.input_width, self.input_height)


@dataclass(frozen=True)
class Config:
    backend_url: str
    api_key: str
    camera_id: str
    direction: str
    camera_index: int
    camera_width: int
    camera_height: int
    camera_fps: int
    request_interval_seconds: float
    min_face_size: int
    crop_padding_ratio: float
    jpeg_quality: int
    display_preview: bool
    face_detector: FaceDetectorConfig
    distance: DistanceConfig
    liveness: LivenessConfig

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

    liveness_model_path = Path(
        os.environ.get(
            "ATTENDANCE_LIVENESS_MODEL_PATH",
            "models/2.7_80x80_MiniFASNetV2.pth",
        ),
    )
    if not liveness_model_path.is_absolute():
        liveness_model_path = ROOT / liveness_model_path

    liveness_v1se_model_path = Path(
        os.environ.get(
            "ATTENDANCE_LIVENESS_V1SE_MODEL_PATH",
            "models/4_0_0_80x80_MiniFASNetV1SE.pth",
        ),
    )
    if not liveness_v1se_model_path.is_absolute():
        liveness_v1se_model_path = ROOT / liveness_v1se_model_path

    raw_liveness_proto_path = os.environ.get("ATTENDANCE_LIVENESS_MODEL_PROTO_PATH")
    liveness_model_proto_path: Path | None = None
    if raw_liveness_proto_path:
        liveness_model_proto_path = Path(raw_liveness_proto_path)
        if not liveness_model_proto_path.is_absolute():
            liveness_model_proto_path = ROOT / liveness_model_proto_path

    liveness = LivenessConfig(
        enabled=read_bool("ATTENDANCE_LIVENESS_ENABLED", False),
        precheck_enabled=read_bool("ATTENDANCE_LIVENESS_PRECHECK_ENABLED", True),
        precheck_extra_cm=read_float(
            "ATTENDANCE_LIVENESS_PRECHECK_EXTRA_CM",
            50.0,
        ),
        pass_ttl_seconds=read_float(
            "ATTENDANCE_LIVENESS_PASS_TTL_SECONDS",
            3.0,
        ),
        min_frames=read_int("ATTENDANCE_LIVENESS_MIN_FRAMES", 12),
        threshold=read_float("ATTENDANCE_LIVENESS_THRESHOLD", 0.85),
        min_face_size=read_int(
            "ATTENDANCE_LIVENESS_MIN_FACE_SIZE",
            read_int("ATTENDANCE_MIN_FACE_SIZE", 90),
        ),
        crop_padding_ratio=read_float("ATTENDANCE_LIVENESS_CROP_PADDING_RATIO", 0.85),
        screen_check_enabled=read_bool("ATTENDANCE_LIVENESS_SCREEN_CHECK_ENABLED", True),
        model_path=liveness_model_path,
        v1se_model_path=liveness_v1se_model_path,
        model_proto_path=liveness_model_proto_path,
        input_width=read_int("ATTENDANCE_LIVENESS_INPUT_WIDTH", 80),
        input_height=read_int("ATTENDANCE_LIVENESS_INPUT_HEIGHT", 80),
        scale=read_float("ATTENDANCE_LIVENESS_SCALE", 1.0 / 255.0),
        mean_b=read_float("ATTENDANCE_LIVENESS_MEAN_B", 0.0),
        mean_g=read_float("ATTENDANCE_LIVENESS_MEAN_G", 0.0),
        mean_r=read_float("ATTENDANCE_LIVENESS_MEAN_R", 0.0),
        swap_rb=read_bool("ATTENDANCE_LIVENESS_SWAP_RB", True),
        live_class_index=read_int("ATTENDANCE_LIVENESS_LIVE_CLASS_INDEX", 1),
        torch_device=os.environ.get("ATTENDANCE_LIVENESS_TORCH_DEVICE", "auto").strip(),
        track_max_gap_seconds=read_float(
            "ATTENDANCE_LIVENESS_TRACK_MAX_GAP_SECONDS",
            0.75,
        ),
        screen_risk_threshold=read_float(
            "ATTENDANCE_LIVENESS_SCREEN_RISK_THRESHOLD",
            0.62,
        ),
        motion_check_enabled=read_bool(
            "ATTENDANCE_LIVENESS_MOTION_CHECK_ENABLED",
            True,
        ),
        motion_min_observations=read_int(
            "ATTENDANCE_LIVENESS_MOTION_MIN_OBSERVATIONS",
            4,
        ),
    )
    validate_liveness_config(liveness)

    camera_width = read_int("ATTENDANCE_CAMERA_WIDTH", 1920)
    camera_height = read_int("ATTENDANCE_CAMERA_HEIGHT", 1080)
    camera_fps = read_int("ATTENDANCE_CAMERA_FPS", 30)
    if camera_width < 1 or camera_height < 1 or camera_fps < 1:
        raise RuntimeError(
            "ATTENDANCE_CAMERA_WIDTH/HEIGHT/FPS must all be positive",
        )

    return Config(
        backend_url=read_required("ATTENDANCE_BACKEND_URL"),
        api_key=read_required("ATTENDANCE_API_KEY"),
        camera_id=read_required("ATTENDANCE_CAMERA_ID"),
        direction=direction,
        camera_index=read_int("ATTENDANCE_CAMERA_INDEX", 0),
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
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
        liveness=liveness,
    )
