from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from config import DistanceConfig


FaceBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class DistanceCheck:
    distance_cm: float | None
    accepted: bool
    status: str


def create_face_detector() -> Any:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Cannot load OpenCV face cascade: {cascade_path}")
    return detector


def detect_faces(detector: Any, frame: Any, min_face_size: int) -> Any:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )


def largest_face(faces: Any) -> FaceBox | None:
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda face: int(face[2]) * int(face[3]))
    return int(x), int(y), int(w), int(h)


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
