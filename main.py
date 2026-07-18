from __future__ import annotations

import time

import cv2

from config import load_config
from liveness import LivenessDecision, LivenessGate
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


def format_liveness_status(decision: LivenessDecision) -> str:
    if not decision.enabled:
        return ""
    score = "" if decision.score is None else f" score={decision.score:.2f}"
    classes = ""
    if decision.class_scores:
        classes = " classes=[" + ",".join(
            f"{class_score:.2f}" for class_score in decision.class_scores
        ) + "]"
    return f" liveness={decision.status}{score}{classes}"


def main() -> None:
    config = load_config()
    detector = create_face_detector(config.face_detector)
    liveness_gate = LivenessGate(config.liveness)
    liveness_gate.set_distance_window(
        config.distance.min_distance_cm,
        config.distance.max_distance_cm,
    )
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
    completed_visit_ids: set[str] = set()
    attempted_visit_ids: set[str] = set()
    checkout_cooldowns: dict[str, float] = {}

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
                liveness_gate.no_face()
                last_status = "waiting for face"
            elif len(faces) > 1:
                liveness_gate.no_face()
                last_status = f"multiple faces detected ({len(faces)})"
                for detection in faces:
                    draw_face_overlay(
                        frame,
                        detection.box,
                        check_face_distance(detection.box, config.distance),
                    )
            else:
                distance = check_face_distance(face, config.distance)
                draw_face_overlay(frame, face, distance)
                now = time.monotonic()
                liveness_decision = liveness_gate.update(
                    frame,
                    face,
                    distance,
                    now,
                )
                liveness_status = format_liveness_status(liveness_decision)

                if distance.accepted:
                    if config.liveness.enabled and not liveness_decision.accepted:
                        last_status = f"{distance.status}{liveness_status}"
                    elif now - last_request_at >= config.request_interval_seconds:
                        last_request_at = now
                        try:
                            crop = crop_face(frame, face, config.crop_padding_ratio)
                            jpeg_bytes = encode_jpeg(crop, config.jpeg_quality)
                            metadata = face_metadata(face, distance)
                            metadata.update(liveness_decision.metadata)
                            result = post_recognition(
                                config,
                                jpeg_bytes,
                                metadata,
                            )
                            last_status = (
                                f"{format_recognition_result(result)}"
                                f"{liveness_status}"
                            )
                            print(last_status)
                        except Exception as error:
                            last_status = f"error: {error}"
                            print(last_status)
                else:
                    last_status = f"{distance.status}{liveness_status}"

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
