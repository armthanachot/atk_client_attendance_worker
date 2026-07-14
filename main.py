from __future__ import annotations

import time

import cv2

from config import load_config
from recognition_client import format_recognition_result, post_recognition
from vision import (
    check_face_distance,
    create_face_detector,
    crop_face,
    detect_faces,
    draw_face_overlay,
    draw_status,
    encode_jpeg,
    face_metadata,
    largest_face,
    should_stop_from_preview_key,
)


def main() -> None:
    config = load_config()
    detector = create_face_detector(config.face_detector)
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

            faces = detect_faces(detector, frame, config.min_face_size)
            face = largest_face(faces)

            if face is None:
                last_status = "waiting for face"
            else:
                distance = check_face_distance(face, config.distance)
                draw_face_overlay(frame, face, distance)

                if distance.accepted:
                    now = time.monotonic()
                    if now - last_request_at >= config.request_interval_seconds:
                        last_request_at = now
                        try:
                            crop = crop_face(frame, face, config.crop_padding_ratio)
                            jpeg_bytes = encode_jpeg(crop, config.jpeg_quality)
                            result = post_recognition(
                                config,
                                jpeg_bytes,
                                face_metadata(face, distance),
                            )
                            last_status = format_recognition_result(result)
                            print(last_status)
                        except Exception as error:
                            last_status = f"error: {error}"
                            print(last_status)
                else:
                    last_status = distance.status

            if config.display_preview:
                draw_status(frame, last_status)
                cv2.imshow("ATK Store Attendance Worker", frame)
                if should_stop_from_preview_key(cv2.waitKey(1)):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
