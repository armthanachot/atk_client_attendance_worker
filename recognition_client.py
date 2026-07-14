from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any

from config import Config


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


def build_metadata(config: Config, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "worker": "client_attendance_worker",
        "cameraIndex": config.camera_index,
    }
    if extra:
        metadata.update(extra)
    return metadata


def post_recognition(
    config: Config,
    jpeg_bytes: bytes,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "cameraId": config.camera_id,
        "direction": config.direction,
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "metadata": json.dumps(build_metadata(config, metadata)),
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
        with urllib.request.urlopen(
            request,
            timeout=15,
            context=ssl._create_unverified_context(),
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend returned {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Cannot reach backend: {error}") from error


def format_recognition_result(result: dict[str, Any]) -> str:
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
