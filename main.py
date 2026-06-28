from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


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

    @property
    def recognize_url(self) -> str:
        return f"{self.backend_url.rstrip('/')}/api/client-attendance/recognize"


def load_config() -> Config:
    load_env_file(ROOT / ".env")
    direction = os.environ.get("ATTENDANCE_DIRECTION", "entry").strip()
    if direction not in {"entry", "exit", "sighting"}:
        raise RuntimeError("ATTENDANCE_DIRECTION must be entry, exit, or sighting")

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
    )


def build_multipart_form(fields: dict[str, str], image: bytes) -> tuple[bytes, str]:
    boundary = f"----atkstore-{uuid.uuid4().hex}"
    parts: list[bytes] = []

    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ],
        )

    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="image"; '
                'filename="frame.jpg"\r\n'
            ).encode(),
            b"Content-Type: image/jpeg\r\n\r\n",
            image,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ],
    )
    return b"".join(parts), boundary


def post_recognition(config: Config, jpeg_bytes: bytes) -> dict[str, Any]:
    fields = {
        "cameraId": config.camera_id,
        "direction": config.direction,
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "metadata": json.dumps(
            {
                "worker": "client_attendance_worker",
                "cameraIndex": config.camera_index,
            },
        ),
    }
    body, boundary = build_multipart_form(fields, jpeg_bytes)
    request = urllib.request.Request(
        config.recognize_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "x-client-attendance-key": config.api_key,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend returned {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Cannot reach backend: {error}") from error


def largest_face(faces: Any) -> tuple[int, int, int, int] | None:
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: int(face[2]) * int(face[3]))


def crop_face(
    frame: Any,
    face: tuple[int, int, int, int],
    padding_ratio: float,
) -> Any:
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


def format_result(result: dict[str, Any]) -> str:
    event = result.get("event") or {}
    user = result.get("user") or {}
    visit = result.get("visit") or {}
    decision = event.get("decision", "unknown")
    similarity = event.get("similarity")
    name = user.get("name") or user.get("email") or "-"
    visit_status = visit.get("status") or "-"

    if similarity is None:
        return f"{decision} user={name} visit={visit_status}"
    return f"{decision} user={name} similarity={similarity:.2f} visit={visit_status}"


def should_stop_from_preview_key(key: int) -> bool:
    key = key & 0xFF
    return key in {27, ord("q")}


def main() -> None:
    config = load_config()
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Cannot load OpenCV face cascade: {cascade_path}")

    capture = cv2.VideoCapture(config.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera index {config.camera_index}")

    print(
        f"Camera worker started: cameraId={config.camera_id} "
        f"direction={config.direction} backend={config.backend_url}",
    )
    print("Press Esc or q in the preview window to stop.")

    last_request_at = 0.0
    last_status = "waiting"

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame read failed; retrying...")
                time.sleep(0.5)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(config.min_face_size, config.min_face_size),
            )
            face = largest_face(faces)

            if face is not None:
                x, y, w, h = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 180, 90), 2)
                now = time.monotonic()
                if now - last_request_at >= config.request_interval_seconds:
                    last_request_at = now
                    try:
                        crop = crop_face(frame, face, config.crop_padding_ratio)
                        jpeg_bytes = encode_jpeg(crop, config.jpeg_quality)
                        result = post_recognition(config, jpeg_bytes)
                        last_status = format_result(result)
                        print(last_status)
                    except Exception as error:
                        last_status = f"error: {error}"
                        print(last_status)
            else:
                last_status = "waiting for face"

            if config.display_preview:
                cv2.putText(
                    frame,
                    last_status[:120],
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("ATK Store Attendance Worker", frame)
                if should_stop_from_preview_key(cv2.waitKey(1)):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
