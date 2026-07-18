from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import LivenessConfig
from vision import DistanceCheck, FaceBox, FaceLandmarks


@dataclass(frozen=True)
class ScreenSignals:
    risk: float
    rectangle: float
    moire: float
    banding: float
    flicker: float
    glare: float
    display_color: float

    @property
    def metadata(self) -> dict[str, float]:
        return {
            "screenRisk": round(self.risk, 4),
            "screenRectangle": round(self.rectangle, 4),
            "screenMoire": round(self.moire, 4),
            "screenBanding": round(self.banding, 4),
            "screenFlicker": round(self.flicker, 4),
            "screenGlare": round(self.glare, 4),
            "screenDisplayColor": round(self.display_color, 4),
        }


@dataclass(frozen=True)
class MotionSignals:
    observed: bool
    parallax: float
    planar_risk: float
    motion_ratio: float
    reprojection_error_ratio: float
    flicker_risk: float


@dataclass(frozen=True)
class SignalFusion:
    accepted: bool
    score: float
    reasons: tuple[str, ...]


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
    model_scores: tuple[float, ...] = ()
    screen_risk: float = 0.0
    screen_cues: tuple[tuple[str, float], ...] = ()
    motion_score: float = 0.0
    planar_risk: float = 0.0
    motion_observations: int = 0
    reasons: tuple[str, ...] = ()
    track_id: int | None = None
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
            "livenessClassScores": [round(score, 4) for score in self.class_scores],
            "livenessModelScores": [round(score, 4) for score in self.model_scores],
            "livenessScreenRisk": round(self.screen_risk, 4),
            "livenessScreenCues": dict(self.screen_cues),
            "livenessMotionScore": round(self.motion_score, 4),
            "livenessPlanarRisk": round(self.planar_risk, 4),
            "livenessMotionObservations": self.motion_observations,
            "livenessReasons": list(self.reasons),
            "livenessTrackId": self.track_id,
            "livenessExceptionAction": (
                None if self.accepted else "do_not_record_ask_walk_through_again"
            ),
        }


@dataclass(frozen=True)
class LivenessPrediction:
    live_score: float
    class_scores: tuple[float, ...]
    model_scores: tuple[float, ...] = ()


class DnnLivenessModel:
    def __init__(self, config: LivenessConfig) -> None:
        if not config.model_path.exists():
            raise_missing_model(config.model_path)
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
            return LivenessPrediction(live_score, (live_score,), (live_score,))
        if self.config.live_class_index >= output.size:
            raise RuntimeError(
                "ATTENDANCE_LIVENESS_LIVE_CLASS_INDEX is outside model output",
            )
        scores = softmax(output)
        class_scores = tuple(float(score) for score in scores)
        live_score = class_scores[self.config.live_class_index]
        return LivenessPrediction(live_score, class_scores, (live_score,))


class TorchMiniFASNetLivenessModel:
    def __init__(self, config: LivenessConfig) -> None:
        if not config.model_path.exists():
            raise_missing_model(config.model_path)
        try:
            import torch
            import torch.nn.functional as torch_functional
            from mini_fasnet import build_minifasnet, strip_module_prefix
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
        resized = cv2.resize(
            image,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
        tensor = self.torch.from_numpy(tensor).div(255.0).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
            scores = (
                self.torch_functional.softmax(output, dim=1)
                .detach()
                .cpu()
                .numpy()[0]
            )
        if self.config.live_class_index >= scores.size:
            raise RuntimeError(
                "ATTENDANCE_LIVENESS_LIVE_CLASS_INDEX is outside model output",
            )
        class_scores = tuple(float(score) for score in scores)
        live_score = class_scores[self.config.live_class_index]
        return LivenessPrediction(live_score, class_scores, (live_score,))


@dataclass(frozen=True)
class EnsembleMember:
    model: Any
    crop_scale: float


class MiniFASNetEnsemble:
    def __init__(self, config: LivenessConfig) -> None:
        if not config.v1se_model_path.exists():
            raise_missing_model(config.v1se_model_path)
        v1se_config = replace(config, model_path=config.v1se_model_path)
        self.members = (
            EnsembleMember(
                TorchMiniFASNetLivenessModel(config),
                1.0 + 2.0 * config.crop_padding_ratio,
            ),
            EnsembleMember(
                TorchMiniFASNetLivenessModel(v1se_config),
                model_crop_scale(config.v1se_model_path, 4.0),
            ),
        )
        self.name = "+".join(member.model.name for member in self.members)

    def predict(self, frame: Any, face: FaceBox) -> LivenessPrediction:
        predictions = [
            member.model.predict(crop_liveness_patch(frame, face, member.crop_scale))
            for member in self.members
        ]
        model_scores = tuple(prediction.live_score for prediction in predictions)
        class_count = min(len(prediction.class_scores) for prediction in predictions)
        class_scores = tuple(
            float(np.mean([prediction.class_scores[index] for prediction in predictions]))
            for index in range(class_count)
        )
        return LivenessPrediction(
            float(np.mean(model_scores)),
            class_scores,
            model_scores,
        )


class SingleFrameModel:
    def __init__(self, config: LivenessConfig, model: Any) -> None:
        self.model = model
        self.crop_scale = 1.0 + 2.0 * config.crop_padding_ratio
        self.name = model.name

    def predict(self, frame: Any, face: FaceBox) -> LivenessPrediction:
        return self.model.predict(crop_liveness_patch(frame, face, self.crop_scale))


class TemporalSignalAnalyzer:
    def __init__(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.previous_points: np.ndarray | None = None
        self.previous_landmarks: FaceLandmarks = ()
        self.previous_patch: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_gray = None
        self.previous_points = None
        self.previous_landmarks = ()
        self.previous_patch = None

    def update(
        self,
        frame: Any,
        face: FaceBox,
        landmarks: FaceLandmarks,
    ) -> MotionSignals:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patch = normalized_gray_patch(frame, face)
        flicker = temporal_flicker_score(self.previous_patch, patch)
        self.previous_patch = patch

        if (
            self.previous_gray is None
            or self.previous_gray.shape != gray.shape
            or self.previous_points is None
            or len(self.previous_points) < 8
        ):
            self._prime(gray, face, landmarks)
            return MotionSignals(False, 0.0, 0.0, 0.0, 0.0, flicker)

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or status is None:
            self._prime(gray, face, landmarks)
            return MotionSignals(False, 0.0, 0.0, 0.0, 0.0, flicker)

        good_mask = status.reshape(-1) == 1
        previous = self.previous_points.reshape(-1, 2)[good_mask]
        current = current_points.reshape(-1, 2)[good_mask]
        if len(previous) < 8:
            self._prime(gray, face, landmarks)
            return MotionSignals(False, 0.0, 0.0, 0.0, 0.0, flicker)

        face_scale = max(1.0, math.hypot(face[2], face[3]))
        displacement = np.linalg.norm(current - previous, axis=1)
        motion_ratio = float(np.median(displacement) / face_scale)
        homography, _ = cv2.findHomography(
            previous,
            current,
            cv2.RANSAC,
            max(1.0, face_scale * 0.006),
        )
        error_ratio = 0.0
        if homography is not None:
            projected = cv2.perspectiveTransform(
                previous.reshape(-1, 1, 2),
                homography,
            ).reshape(-1, 2)
            feature_error = float(
                np.percentile(np.linalg.norm(projected - current, axis=1), 75),
            )
            landmark_error = landmark_reprojection_error(
                homography,
                self.previous_landmarks,
                landmarks,
            )
            error_ratio = max(feature_error, landmark_error) / face_scale

        observed = motion_ratio >= 0.002 and homography is not None
        parallax = (
            clamp01((error_ratio - 0.0015) / 0.010)
            if observed and homography is not None
            else 0.0
        )
        planar_risk = (
            clamp01((0.0045 - error_ratio) / 0.0045)
            * clamp01((motion_ratio - 0.002) / 0.010)
            if observed and homography is not None
            else 0.0
        )

        self.previous_gray = gray
        self.previous_landmarks = landmarks
        self.previous_points = current.reshape(-1, 1, 2).astype(np.float32)
        if len(self.previous_points) < 24:
            self.previous_points = feature_points(gray, face)
        return MotionSignals(
            observed,
            parallax,
            planar_risk,
            motion_ratio,
            error_ratio,
            flicker,
        )

    def _prime(
        self,
        gray: np.ndarray,
        face: FaceBox,
        landmarks: FaceLandmarks,
    ) -> None:
        self.previous_gray = gray
        self.previous_points = feature_points(gray, face)
        self.previous_landmarks = landmarks


class LivenessGate:
    def __init__(self, config: LivenessConfig) -> None:
        self.config = config
        self.model = create_liveness_model(config) if config.enabled else None
        history_size = max(config.min_frames, 24)
        self.scores: deque[float] = deque(maxlen=history_size)
        self.weakest_model_scores: deque[float] = deque(maxlen=history_size)
        self.per_model_scores: deque[tuple[float, ...]] = deque(maxlen=history_size)
        self.class_scores: deque[tuple[float, ...]] = deque(maxlen=history_size)
        self.screen_risks: deque[float] = deque(maxlen=history_size)
        self.motion_scores: deque[float] = deque(maxlen=history_size)
        self.planar_risks: deque[float] = deque(maxlen=history_size)
        self.track_face: FaceBox | None = None
        self.track_id: int | None = None
        self.next_track_id = 1
        self.last_seen_at: float | None = None
        self.last_pass: LivenessDecision | None = None
        self.last_screen_signals: ScreenSignals | None = None
        self.temporal = TemporalSignalAnalyzer()

    def reset(self) -> None:
        self.scores.clear()
        self.weakest_model_scores.clear()
        self.per_model_scores.clear()
        self.class_scores.clear()
        self.screen_risks.clear()
        self.motion_scores.clear()
        self.planar_risks.clear()
        self.track_face = None
        self.track_id = None
        self.last_seen_at = None
        self.last_pass = None
        self.last_screen_signals = None
        self.temporal.reset()

    def no_face(self, now: float | None = None) -> LivenessDecision:
        now = time.monotonic() if now is None else now
        if (
            self.last_seen_at is not None
            and now - self.last_seen_at > self.config.track_max_gap_seconds
        ):
            self.reset()
        return self._decision(False, "waiting for face", None)

    def update(
        self,
        frame: Any,
        face: FaceBox,
        distance: DistanceCheck,
        now: float | None = None,
        landmarks: FaceLandmarks = (),
    ) -> LivenessDecision:
        now = time.monotonic() if now is None else now
        if not self.config.enabled:
            return self._decision(True, "disabled", None)

        track_expired = (
            self.last_seen_at is not None
            and now - self.last_seen_at > self.config.track_max_gap_seconds
        )
        if self.track_face is None or track_expired or not same_face_track(
            self.track_face,
            face,
        ):
            self.reset()
            self.track_id = self.next_track_id
            self.next_track_id += 1
        self.track_face = face
        self.last_seen_at = now

        if not self._can_sample(face, distance):
            return self._decision(False, "outside liveness zone", None)
        if self.model is None:
            raise RuntimeError("Liveness model is not initialized")

        motion = self.temporal.update(frame, face, landmarks)
        if motion.observed:
            self.motion_scores.append(motion.parallax)
            self.planar_risks.append(motion.planar_risk)
        screen = analyze_screen_presentation(frame, face, motion.flicker_risk)
        self.last_screen_signals = screen
        self.screen_risks.append(screen.risk)

        prediction = self.model.predict(frame, face)
        self.scores.append(prediction.live_score)
        self.class_scores.append(prediction.class_scores)
        self.per_model_scores.append(prediction.model_scores)
        self.weakest_model_scores.append(min(prediction.model_scores))

        aggregate_score = aggregate_track_scores(self.scores)
        aggregate_classes = aggregate_class_scores(self.class_scores)
        aggregate_models = aggregate_model_scores(self.per_model_scores)
        weakest_score = aggregate_track_scores(self.weakest_model_scores)
        screen_risk = percentile(self.screen_risks, 75)
        motion_score = mean(self.motion_scores)
        planar_risk = mean(self.planar_risks)

        if len(self.scores) < self.config.min_frames:
            return self._decision(
                False,
                f"collecting track {len(self.scores)}/{self.config.min_frames}",
                aggregate_score,
                aggregate_classes,
                aggregate_models,
                screen_risk,
                motion_score,
                planar_risk,
            )

        fusion = fuse_signal_scores(
            model_score=aggregate_score,
            weakest_model_score=weakest_score,
            screen_risk=screen_risk if self.config.screen_check_enabled else 0.0,
            motion_score=motion_score,
            planar_risk=planar_risk,
            motion_observations=len(self.motion_scores),
            threshold=self.config.threshold,
            screen_risk_threshold=self.config.screen_risk_threshold,
            motion_min_observations=(
                self.config.motion_min_observations
                if self.config.motion_check_enabled
                else 0
            ),
        )
        if fusion.accepted:
            decision = self._decision(
                True,
                "live",
                fusion.score,
                aggregate_classes,
                aggregate_models,
                screen_risk,
                motion_score,
                planar_risk,
            )
            decision = replace(
                decision,
                valid_until_monotonic=now + self.config.pass_ttl_seconds,
            )
            self.last_pass = decision
            return decision

        spoof_reasons = {"screen cues", "planar motion", "model disagreement"}
        status = (
            "spoof suspected"
            if spoof_reasons.intersection(fusion.reasons)
            else "uncertain - walk through again"
        )
        return self._decision(
            False,
            status,
            fusion.score,
            aggregate_classes,
            aggregate_models,
            screen_risk,
            motion_score,
            planar_risk,
            fusion.reasons,
        )

    def _can_sample(self, face: FaceBox, distance: DistanceCheck) -> bool:
        _, _, width, height = face
        if width < self.config.min_face_size or height < self.config.min_face_size:
            return False
        if distance.distance_cm is None:
            return distance.accepted
        return (
            distance.distance_cm >= self.config_min_distance_cm
            and distance.distance_cm <= self.config_max_distance_cm
        )

    @property
    def config_min_distance_cm(self) -> float:
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
        model_scores: tuple[float, ...] = (),
        screen_risk: float = 0.0,
        motion_score: float = 0.0,
        planar_risk: float = 0.0,
        reasons: tuple[str, ...] = (),
    ) -> LivenessDecision:
        return LivenessDecision(
            enabled=self.config.enabled,
            accepted=accepted,
            status=status,
            score=score,
            frames_checked=len(self.scores),
            threshold=self.config.threshold,
            model=self.model.name if self.model else "none",
            class_scores=class_scores,
            model_scores=model_scores,
            screen_risk=screen_risk,
            screen_cues=(
                tuple(self.last_screen_signals.metadata.items())
                if self.last_screen_signals
                else ()
            ),
            motion_score=motion_score,
            planar_risk=planar_risk,
            motion_observations=len(self.motion_scores),
            reasons=reasons,
            track_id=self.track_id,
        )


def fuse_signal_scores(
    *,
    model_score: float,
    weakest_model_score: float,
    screen_risk: float,
    motion_score: float,
    planar_risk: float,
    motion_observations: int,
    threshold: float,
    screen_risk_threshold: float,
    motion_min_observations: int,
) -> SignalFusion:
    score = clamp01(
        model_score * 0.72
        + motion_score * 0.28
        - screen_risk * 0.45
        - planar_risk * 0.18,
    )
    reasons: list[str] = []
    if weakest_model_score < max(0.25, threshold - 0.15):
        reasons.append("model disagreement")
    if screen_risk >= screen_risk_threshold:
        reasons.append("screen cues")
    if motion_observations < motion_min_observations:
        reasons.append("motion not observed")
    if planar_risk >= 0.55 and motion_score < 0.25:
        reasons.append("planar motion")
    if score < threshold:
        reasons.append("combined score")
    return SignalFusion(not reasons, score, tuple(reasons))


def aggregate_track_scores(values: deque[float]) -> float:
    if not values:
        return 0.0
    raw = np.asarray(values, dtype=np.float32)
    return float(0.60 * np.mean(raw) + 0.40 * np.percentile(raw, 10))


def aggregate_class_scores(
    values: deque[tuple[float, ...]],
) -> tuple[float, ...]:
    if not values:
        return ()
    count = min(len(value) for value in values)
    return tuple(
        aggregate_track_scores(deque(value[index] for value in values))
        for index in range(count)
    )


def aggregate_model_scores(
    values: deque[tuple[float, ...]],
) -> tuple[float, ...]:
    return aggregate_class_scores(values)


def analyze_screen_presentation(
    frame: Any,
    face: FaceBox,
    flicker_risk: float = 0.0,
) -> ScreenSignals:
    patch = crop_liveness_patch(frame, face, 4.5)
    if patch.size == 0:
        return ScreenSignals(0.0, 0.0, 0.0, 0.0, flicker_risk, 0.0, 0.0)
    rectangle = full_frame_rectangle_score(frame, face)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    moire = frequency_moire_score(gray)
    banding = refresh_banding_score(gray)
    glare = glare_score(patch)
    display_color = display_color_score(patch)
    risk = clamp01(
        rectangle * 0.68
        + moire * 0.18
        + banding * 0.18
        + flicker_risk * 0.12
        + glare * 0.10
        + display_color * 0.12,
    )
    return ScreenSignals(
        risk,
        rectangle,
        moire,
        banding,
        flicker_risk,
        glare,
        display_color,
    )


def has_screen_presentation(frame: Any, face: FaceBox) -> bool:
    return analyze_screen_presentation(frame, face).risk >= 0.62


def full_frame_rectangle_score(frame: Any, face: FaceBox) -> float:
    frame_height, frame_width = frame.shape[:2]
    scale = min(1.0, 960.0 / max(1, frame_width))
    if scale < 1.0:
        image = cv2.resize(frame, None, fx=scale, fy=scale)
    else:
        image = frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    x, y, width, height = (float(value) * scale for value in face)
    face_area = width * height
    face_bounds = (x, y, x + width, y + height)
    image_area = image.shape[0] * image.shape[1]
    best = 0.0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(polygon)
        area = float(cv2.contourArea(polygon))
        if area < face_area * 2.2 or area > image_area * 0.96:
            continue
        aspect = rect_w / max(1.0, rect_h)
        if not 0.35 <= aspect <= 2.9:
            continue
        left, top, right, bottom = face_bounds
        margin = max(width, height) * 0.08
        if not (
            rect_x <= left + margin
            and rect_y <= top + margin
            and rect_x + rect_w >= right - margin
            and rect_y + rect_h >= bottom - margin
        ):
            continue
        rectangularity = area / max(1.0, rect_w * rect_h)
        best = max(best, clamp01((rectangularity - 0.70) / 0.20))
    return best


def frequency_moire_score(gray: np.ndarray) -> float:
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    residual = small.astype(np.float32) - cv2.GaussianBlur(
        small.astype(np.float32),
        (0, 0),
        1.8,
    )
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual)))
    spectrum[52:76, 52:76] = 0
    positive = spectrum[spectrum > 0]
    if positive.size < 10:
        return 0.0
    peak_ratio = float(np.percentile(positive, 99.8) / (np.median(positive) + 1e-6))
    return clamp01((peak_ratio - 18.0) / 35.0)


def refresh_banding_score(gray: np.ndarray) -> float:
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(
        np.float32,
    )
    scores: list[float] = []
    for profile in (small.mean(axis=0), small.mean(axis=1)):
        profile = profile - cv2.GaussianBlur(
            profile.reshape(1, -1),
            (0, 0),
            4,
        ).ravel()
        spectrum = np.abs(np.fft.rfft(profile))[2:]
        if spectrum.size:
            scores.append(float(np.max(spectrum) / (np.median(spectrum) + 1e-6)))
    return clamp01((max(scores, default=0.0) - 8.0) / 20.0)


def glare_score(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= 245) & (hsv[:, :, 1] <= 70)).astype(np.uint8)
    fraction = float(mask.mean())
    if fraction <= 0.003:
        return 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    largest = 0 if count <= 1 else int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    connected_fraction = largest / max(1, mask.size)
    return clamp01((fraction - 0.003) / 0.035) * clamp01(
        connected_fraction / 0.015,
    )


def display_color_score(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    clipped = float(((value <= 6) | (value >= 249)).mean())
    saturation = float(np.percentile(hsv[:, :, 1], 90)) / 255.0
    contrast = float(np.std(value)) / 64.0
    return clamp01((clipped - 0.08) / 0.22) * 0.55 + clamp01(
        (saturation * contrast - 0.55) / 0.45,
    ) * 0.45


def temporal_flicker_score(
    previous: np.ndarray | None,
    current: np.ndarray,
) -> float:
    if previous is None or previous.shape != current.shape:
        return 0.0
    mean_delta = abs(float(np.mean(current)) - float(np.mean(previous)))
    row_delta = float(
        np.mean(np.abs(current.mean(axis=1) - previous.mean(axis=1))),
    )
    return clamp01((mean_delta - 2.0) / 12.0) * 0.5 + clamp01(
        (row_delta - 3.0) / 15.0,
    ) * 0.5


def feature_points(gray: np.ndarray, face: FaceBox) -> np.ndarray | None:
    mask = np.zeros_like(gray)
    x, y, width, height = face
    inset_x = int(width * 0.08)
    inset_y = int(height * 0.08)
    left = max(0, x + inset_x)
    top = max(0, y + inset_y)
    right = min(gray.shape[1], x + width - inset_x)
    bottom = min(gray.shape[0], y + height - inset_y)
    if right <= left or bottom <= top:
        return None
    cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=80,
        qualityLevel=0.008,
        minDistance=4,
        mask=mask,
        blockSize=5,
    )


def landmark_reprojection_error(
    homography: np.ndarray,
    previous: FaceLandmarks,
    current: FaceLandmarks,
) -> float:
    if len(previous) != len(current) or len(previous) < 5:
        return 0.0
    previous_points = np.asarray(previous, dtype=np.float32).reshape(-1, 1, 2)
    current_points = np.asarray(current, dtype=np.float32)
    projected = cv2.perspectiveTransform(previous_points, homography).reshape(-1, 2)
    return float(np.median(np.linalg.norm(projected - current_points, axis=1)))


def normalized_gray_patch(frame: Any, face: FaceBox) -> np.ndarray:
    patch = crop_liveness_patch(frame, face, 2.7)
    if patch.size == 0:
        return np.zeros((64, 64), dtype=np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)


def crop_liveness_patch(frame: Any, face: FaceBox, scale: float) -> Any:
    left, top, right, bottom = locate_liveness_crop(
        frame.shape[1],
        frame.shape[0],
        face,
        scale,
    )
    return frame[top : bottom + 1, left : right + 1]


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
    return (
        max(0, int(left)),
        max(0, int(top)),
        min(frame_width - 1, int(right)),
        min(frame_height - 1, int(bottom)),
    )


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
        center_distance <= average_width * 0.45
        and width_ratio <= 1.65
        and height_ratio <= 1.65
    )


def create_liveness_model(config: LivenessConfig) -> Any:
    if config.model_path.suffix.lower() == ".pth":
        return MiniFASNetEnsemble(config)
    return SingleFrameModel(config, DnnLivenessModel(config))


def model_crop_scale(model_path: Path, default: float) -> float:
    first = model_path.stem.split("_", 1)[0]
    try:
        scale = float(first)
    except ValueError:
        return default
    return scale if scale > 0 else default


def raise_missing_model(model_path: Path) -> None:
    raise RuntimeError(
        "Liveness is enabled but model file is missing: "
        f"{model_path}. Configure the checkpoint or disable liveness.",
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
        return config.input_height, config.input_width, config.model_path.stem


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def mean(values: deque[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def percentile(values: deque[float], value: float) -> float:
    return float(np.percentile(values, value)) if values else 0.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
