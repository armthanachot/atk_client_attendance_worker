from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import LivenessConfig
from vision import DistanceCheck, FaceBox, crop_face


@dataclass(frozen=True)
class LivenessDecision:
    enabled: bool
    accepted: bool
    status: str
    score: float | None
    frames_checked: int
    threshold: float
    model: str
    class_scores: tuple[float, ...] = ()
    valid_until_monotonic: float | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "livenessEnabled": self.enabled,
            "livenessDecision": "live" if self.accepted else self.status,
            "livenessScore": round(self.score, 4) if self.score is not None else None,
            "livenessFramesChecked": self.frames_checked,
            "livenessThreshold": self.threshold,
            "livenessModel": self.model,
            "livenessClassScores": [
                round(score, 4) for score in self.class_scores
            ],
        }


@dataclass(frozen=True)
class LivenessPrediction:
    live_score: float
    class_scores: tuple[float, ...]


class DnnLivenessModel:
    def __init__(self, config: LivenessConfig) -> None:
        if not config.model_path.exists():
            raise RuntimeError(
                "Liveness is enabled but model file is missing: "
                f"{config.model_path}. Set ATTENDANCE_LIVENESS_MODEL_PATH "
                "or disable ATTENDANCE_LIVENESS_ENABLED.",
            )
        if config.model_path.suffix == ".caffemodel" and not config.model_proto_path:
            raise RuntimeError(
                "Caffe liveness models need ATTENDANCE_LIVENESS_MODEL_PROTO_PATH",
            )
        if config.model_proto_path and not config.model_proto_path.exists():
            raise RuntimeError(
                f"Liveness model proto file is missing: {config.model_proto_path}",
            )

        if config.model_proto_path:
            self.net = cv2.dnn.readNet(
                str(config.model_path),
                str(config.model_proto_path),
            )
        else:
            self.net = cv2.dnn.readNet(str(config.model_path))

        self.config = config
        self.name = config.model_path.stem

    def predict(self, image: Any) -> LivenessPrediction:
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=self.config.scale,
            size=self.config.input_size,
            mean=(self.config.mean_b, self.config.mean_g, self.config.mean_r),
            swapRB=self.config.swap_rb,
            crop=False,
        )
        self.net.setInput(blob)
        output = np.asarray(self.net.forward(), dtype=np.float32).reshape(-1)

        if output.size == 0:
            raise RuntimeError("Liveness model returned no output")
        if output.size == 1:
            live_score = sigmoid(float(output[0]))
            return LivenessPrediction(live_score, (live_score,))
        if self.config.live_class_index >= output.size:
            raise RuntimeError(
                "ATTENDANCE_LIVENESS_LIVE_CLASS_INDEX is outside model output",
            )

        scores = softmax(output)
        class_scores = tuple(float(score) for score in scores)
        return LivenessPrediction(
            class_scores[self.config.live_class_index],
            class_scores,
        )


class TorchMiniFASNetLivenessModel:
    def __init__(self, config: LivenessConfig) -> None:
        if not config.model_path.exists():
            raise_missing_model(config.model_path)

        try:
            import torch
            import torch.nn.functional as torch_functional
            from mini_fasnet import (
                build_minifasnet,
                parse_model_name,
                strip_module_prefix,
            )
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required for .pth liveness models. "
                "Install dependencies with `pip install -r requirements.txt`.",
            ) from error

        self.torch = torch
        self.torch_functional = torch_functional
        self.device = choose_torch_device(torch, config.torch_device)
        self.height, self.width, model_type = read_torch_model_shape(config)
        self.model = build_minifasnet(model_type, self.height, self.width).to(self.device)
        state_dict = load_torch_state_dict(torch, config.model_path, self.device)
        self.model.load_state_dict(strip_module_prefix(state_dict))
        self.model.eval()
        self.config = config
        self.name = config.model_path.stem

    def predict(self, image: Any) -> LivenessPrediction:
        resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
        tensor = self.torch.from_numpy(tensor).div(255.0).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
            scores = self.torch_functional.softmax(output, dim=1).detach().cpu().numpy()[0]

        if self.config.live_class_index >= scores.size:
            raise RuntimeError(
                "ATTENDANCE_LIVENESS_LIVE_CLASS_INDEX is outside model output",
            )
        class_scores = tuple(float(score) for score in scores)
        return LivenessPrediction(
            class_scores[self.config.live_class_index],
            class_scores,
        )


class LivenessGate:
    def __init__(self, config: LivenessConfig) -> None:
        self.config = config
        self.model = create_liveness_model(config) if config.enabled else None
        self.scores: deque[float] = deque(maxlen=max(1, config.min_frames))
        self.class_scores: deque[tuple[float, ...]] = deque(maxlen=max(1, config.min_frames))
        self.track_face: FaceBox | None = None
        self.last_pass: LivenessDecision | None = None

    def reset(self) -> None:
        self.scores.clear()
        self.class_scores.clear()
        self.track_face = None
        self.last_pass = None

    def no_face(self) -> LivenessDecision:
        self.reset()
        return self._decision(False, "waiting for face", None)

    def update(
        self,
        frame: Any,
        face: FaceBox,
        distance: DistanceCheck,
        now: float | None = None,
    ) -> LivenessDecision:
        now = time.monotonic() if now is None else now
        if not self.config.enabled:
            return self._decision(True, "disabled", None)

        if self.track_face is not None and not same_face_track(self.track_face, face):
            self.scores.clear()
            self.class_scores.clear()
            self.last_pass = None
        self.track_face = face

        if self.last_pass and self.last_pass.valid_until_monotonic:
            if now > self.last_pass.valid_until_monotonic:
                self.last_pass = None

        if not self._can_sample(face, distance):
            self.scores.clear()
            self.class_scores.clear()
            return self._decision(False, "outside liveness zone", None)

        if self.model is None:
            raise RuntimeError("Liveness model is not initialized")

        if self.config.screen_check_enabled and has_screen_presentation(frame, face):
            self.scores.clear()
            self.class_scores.clear()
            self.last_pass = None
            return self._decision(False, "screen presentation suspected", None)

        image = crop_liveness_patch(
            frame,
            face,
            1.0 + 2.0 * self.config.crop_padding_ratio,
        )
        prediction = self.model.predict(image)
        latest_score = prediction.live_score
        self.scores.append(latest_score)
        self.class_scores.append(prediction.class_scores)

        aggregate_score = median(self.scores)
        aggregate_class_scores = median_scores(self.class_scores)
        if len(self.scores) < self.config.min_frames:
            return self._decision(
                False,
                f"liveness collecting {len(self.scores)}/{self.config.min_frames}",
                aggregate_score,
                aggregate_class_scores,
            )

        accepted = (
            aggregate_score >= self.config.threshold
            and latest_score >= self.config.threshold
        )
        status = "live" if accepted else "spoof suspected"
        decision = self._decision(accepted, status, aggregate_score, aggregate_class_scores)
        if accepted:
            decision = LivenessDecision(
                enabled=True,
                accepted=True,
                status="live",
                score=aggregate_score,
                frames_checked=len(self.scores),
                threshold=self.config.threshold,
                model=self.model.name,
                class_scores=aggregate_class_scores,
                valid_until_monotonic=now + self.config.pass_ttl_seconds,
            )
            self.last_pass = decision
        return decision

    def _can_sample(self, face: FaceBox, distance: DistanceCheck) -> bool:
        _, _, width, height = face
        if width < self.config.min_face_size or height < self.config.min_face_size:
            return False
        if distance.distance_cm is None:
            return distance.accepted

        distance_cm = distance.distance_cm
        # Use the configured pre-zone as an approach buffer before the normal
        # verify zone. Too-close faces are not sampled because crop quality is
        # often poor and the user should step back anyway.
        return (
            distance_cm >= self.config_min_distance_cm
            and distance_cm <= self.config_max_distance_cm
        )

    @property
    def config_min_distance_cm(self) -> float:
        # These are injected after construction by main() to avoid duplicating
        # distance config inside model config.
        return getattr(self, "_min_distance_cm", 0.0)

    @property
    def config_max_distance_cm(self) -> float:
        return getattr(self, "_max_distance_cm", math.inf)

    def set_distance_window(self, min_distance_cm: float, max_distance_cm: float) -> None:
        self._min_distance_cm = min_distance_cm
        self._max_distance_cm = (
            max_distance_cm + self.config.precheck_extra_cm
            if self.config.precheck_enabled
            else max_distance_cm
        )

    def _decision(
        self,
        accepted: bool,
        status: str,
        score: float | None,
        class_scores: tuple[float, ...] = (),
    ) -> LivenessDecision:
        model_name = self.model.name if self.model else "none"
        return LivenessDecision(
            enabled=self.config.enabled,
            accepted=accepted,
            status=status,
            score=score,
            frames_checked=len(self.scores),
            threshold=self.config.threshold,
            model=model_name,
            class_scores=class_scores,
        )


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def median(values: deque[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def median_scores(values: deque[tuple[float, ...]]) -> tuple[float, ...]:
    if len(values) == 0:
        return ()
    score_count = len(values[0])
    return tuple(median(deque(scores[index] for scores in values)) for index in range(score_count))


def crop_liveness_patch(frame: Any, face: FaceBox, scale: float) -> Any:
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = face
    scale = min((frame_height - 1) / height, min((frame_width - 1) / width, scale))

    new_width = width * scale
    new_height = height * scale
    center_x = width / 2 + x
    center_y = height / 2 + y

    left = center_x - new_width / 2
    top = center_y - new_height / 2
    right = center_x + new_width / 2
    bottom = center_y + new_height / 2

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > frame_width - 1:
        left -= right - frame_width + 1
        right = frame_width - 1
    if bottom > frame_height - 1:
        top -= bottom - frame_height + 1
        bottom = frame_height - 1

    left = max(0, int(left))
    top = max(0, int(top))
    right = min(frame_width - 1, int(right))
    bottom = min(frame_height - 1, int(bottom))
    return frame[top : bottom + 1, left : right + 1]


def has_screen_presentation(frame: Any, face: FaceBox) -> bool:
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = face
    face_area = width * height
    if face_area <= 0:
        return False

    search_scale = 5.0
    crop = crop_liveness_patch(frame, face, search_scale)
    crop_height, crop_width = crop.shape[:2]
    if crop_width < width * 1.6 or crop_height < height * 1.6:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 130)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    face_center_global = (x + width / 2, y + height / 2)
    search_box = locate_liveness_crop(frame_width, frame_height, face, search_scale)
    search_left, search_top, _, _ = search_box
    face_center = (
        face_center_global[0] - search_left,
        face_center_global[1] - search_top,
    )

    for contour in contours:
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
        rect_area = rect_w * rect_h
        if rect_area < face_area * 3.0:
            continue
        if rect_area > crop_width * crop_height * 0.9:
            continue

        aspect = rect_w / max(1, rect_h)
        portrait_screen = 0.35 <= aspect <= 0.85
        landscape_screen = 1.2 <= aspect <= 2.8
        if not (portrait_screen or landscape_screen):
            continue

        if not (
            rect_x <= face_center[0] <= rect_x + rect_w
            and rect_y <= face_center[1] <= rect_y + rect_h
        ):
            continue

        face_left = x - search_left
        face_top = y - search_top
        face_right = face_left + width
        face_bottom = face_top + height
        if (
            rect_x <= face_left
            and rect_y <= face_top
            and rect_x + rect_w >= face_right
            and rect_y + rect_h >= face_bottom
        ):
            return True

    return False


def locate_liveness_crop(
    frame_width: int,
    frame_height: int,
    face: FaceBox,
    scale: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = face
    scale = min((frame_height - 1) / height, min((frame_width - 1) / width, scale))

    new_width = width * scale
    new_height = height * scale
    center_x = width / 2 + x
    center_y = height / 2 + y

    left = center_x - new_width / 2
    top = center_y - new_height / 2
    right = center_x + new_width / 2
    bottom = center_y + new_height / 2

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > frame_width - 1:
        left -= right - frame_width + 1
        right = frame_width - 1
    if bottom > frame_height - 1:
        top -= bottom - frame_height + 1
        bottom = frame_height - 1

    left = max(0, int(left))
    top = max(0, int(top))
    right = min(frame_width - 1, int(right))
    bottom = min(frame_height - 1, int(bottom))
    return left, top, right, bottom


def same_face_track(previous: FaceBox, current: FaceBox) -> bool:
    px, py, pw, ph = previous
    cx, cy, cw, ch = current
    previous_center = (px + pw / 2.0, py + ph / 2.0)
    current_center = (cx + cw / 2.0, cy + ch / 2.0)
    average_width = max(1.0, (pw + cw) / 2.0)
    center_distance = math.dist(previous_center, current_center)
    width_ratio = max(pw, cw) / max(1.0, min(pw, cw))
    height_ratio = max(ph, ch) / max(1.0, min(ph, ch))
    return (
        center_distance <= average_width * 0.35
        and width_ratio <= 1.5
        and height_ratio <= 1.5
    )


def create_liveness_model(config: LivenessConfig) -> Any:
    suffix = config.model_path.suffix.lower()
    if suffix == ".pth":
        return TorchMiniFASNetLivenessModel(config)
    return DnnLivenessModel(config)


def raise_missing_model(model_path: Path) -> None:
    raise RuntimeError(
        "Liveness is enabled but model file is missing: "
        f"{model_path}. Set ATTENDANCE_LIVENESS_MODEL_PATH "
        "or disable ATTENDANCE_LIVENESS_ENABLED.",
    )


def choose_torch_device(torch_module: Any, requested: str) -> Any:
    if requested != "auto":
        return torch_module.device(requested)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def load_torch_state_dict(
    torch_module: Any,
    model_path: Path,
    device: Any,
) -> dict[str, Any]:
    try:
        loaded = torch_module.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        loaded = torch_module.load(model_path, map_location=device)

    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise RuntimeError("PyTorch liveness model must contain a state dict")
    return loaded


def read_torch_model_shape(config: LivenessConfig) -> tuple[int, int, str]:
    from mini_fasnet import parse_model_name

    try:
        return parse_model_name(config.model_path.name)
    except (IndexError, ValueError):
        model_type = config.model_path.stem
        return config.input_height, config.input_width, model_type
