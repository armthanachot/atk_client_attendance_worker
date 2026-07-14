from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from config import DistanceConfig, FaceDetectorConfig


FaceBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class DistanceCheck:
    distance_cm: float | None
    accepted: bool
    status: str


@dataclass(frozen=True)
class FaceDetection:
    box: FaceBox
    score: float | None = None


@dataclass
class FaceDetector:
    detector_type: str
    model: Any


def create_face_detector(config: FaceDetectorConfig) -> FaceDetector:
    if config.detector_type == "yunet":
        if not config.yunet_model_path.exists():
            raise RuntimeError(f"Cannot load YuNet model: {config.yunet_model_path}")
        detector = cv2.FaceDetectorYN.create(
            str(config.yunet_model_path),
            "",
            (320, 320),
            config.yunet_score_threshold,
            config.yunet_nms_threshold,
            config.yunet_top_k,
        )
        return FaceDetector("yunet", detector)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Cannot load OpenCV face cascade: {cascade_path}")
    return FaceDetector("haar", detector)


def detect_faces(
    detector: FaceDetector,
    frame: Any,
    min_face_size: int,
) -> list[FaceDetection]:
    if detector.detector_type == "yunet":
        return detect_faces_yunet(detector.model, frame, min_face_size)
    return detect_faces_haar(detector.model, frame, min_face_size)


def detect_faces_haar(
    detector: Any,
    frame: Any,
    min_face_size: int,
) -> list[FaceDetection]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )
    return [
        FaceDetection((int(x), int(y), int(w), int(h)))
        for x, y, w, h in faces
    ]


def detect_faces_yunet(
    detector: Any,
    frame: Any,
    min_face_size: int,
) -> list[FaceDetection]:
    frame_height, frame_width = frame.shape[:2]
    detector.setInputSize((frame_width, frame_height))
    _, faces = detector.detect(frame)
    if faces is None:
        return []

    detections: list[FaceDetection] = []
    for face in faces:
        x, y, w, h = face[:4]
        if w < min_face_size or h < min_face_size:
            continue
        detections.append(
            FaceDetection(
                (
                    max(0, int(round(x))),
                    max(0, int(round(y))),
                    int(round(w)),
                    int(round(h)),
                ),
                float(face[-1]),
            ),
        )
    return detections


def largest_face(faces: list[FaceDetection]) -> FaceBox | None:
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: face.box[2] * face.box[3]).box


def estimate_face_distance_cm(
    face_width_px: int,
    known_face_width_cm: float,
    focal_length_px: float,
) -> float | None:
    if face_width_px <= 0 or known_face_width_cm <= 0 or focal_length_px <= 0:
        return None
    return (known_face_width_cm * focal_length_px) / face_width_px


def check_face_distance(face: FaceBox, config: DistanceConfig) -> DistanceCheck:
    _, _, width, _ = face
    distance_cm = estimate_face_distance_cm(
        width,
        config.known_face_width_cm,
        config.camera_focal_length_px,
    )
    if distance_cm is None:
        return DistanceCheck(None, not config.gate_enabled, "distance unavailable")

    distance_text = f"distance={distance_cm:.0f}cm"
    if not config.gate_enabled:
        return DistanceCheck(distance_cm, True, distance_text)
    if distance_cm < config.min_distance_cm:
        return DistanceCheck(distance_cm, False, f"{distance_text} step back")
    if distance_cm > config.max_distance_cm:
        return DistanceCheck(distance_cm, False, f"{distance_text} move closer")
    return DistanceCheck(distance_cm, True, distance_text)


def crop_face(frame: Any, face: FaceBox, padding_ratio: float) -> Any:
    height, width = frame.shape[:2]
    x, y, w, h = face
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(width, x + w + pad_x)
    bottom = min(height, y + h + pad_y)
    return frame[top:bottom, left:right]


def encode_jpeg(image: Any, quality: int) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buffer.tobytes()


def face_metadata(face: FaceBox, distance: DistanceCheck) -> dict[str, Any]:
    x, y, w, h = face
    return {
        "faceBox": {"x": x, "y": y, "w": w, "h": h},
        "distanceCm": (
            round(distance.distance_cm, 1)
            if distance.distance_cm is not None
            else None
        ),
        "distanceGateAccepted": distance.accepted,
    }


def draw_face_overlay(frame: Any, face: FaceBox, distance: DistanceCheck) -> None:
    x, y, w, h = face
    color = (30, 180, 90) if distance.accepted else (30, 90, 220)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if distance.distance_cm is None:
        return
    cv2.putText(
        frame,
        f"{distance.distance_cm:.0f} cm",
        (x, max(24, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status(frame: Any, status: str) -> None:
    cv2.putText(
        frame,
        status[:120],
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def should_stop_from_preview_key(key: int) -> bool:
    key = key & 0xFF
    return key in {27, ord("q")}
