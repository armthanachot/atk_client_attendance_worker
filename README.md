# ATK Store Client Attendance Worker

Python camera worker for the in-store face recognition PoC.

The worker:

- reads one local camera
- detects the largest face locally with OpenCV YuNet
- estimates face distance from the detected face width
- runs MiniFASNetV2 + MiniFASNetV1SE as an ensemble
- keeps liveness evidence on a continuous person track instead of a frame median
- checks the full frame for display boundaries and the face area for moire,
  refresh banding/flicker, glare, and display color/contrast
- checks optical-flow parallax and rejects motion that fits one flat homography
- fuses the model, screen, and motion signals and fails closed when uncertain
- skips recognition when the person is outside the configured distance range
- crops the face area
- sends only the selected JPEG crop to the ATK Store backend
- never indexes new faces
- never stores raw images locally

## Backend env

Add this to the Next.js app `.env`:

```txt
CLIENT_ATTENDANCE_API_KEY=use-a-long-random-secret
CLIENT_ATTENDANCE_MAX_IMAGE_BYTES=800000
```

Restart `npm run dev` after changing `.env`.

For two notebooks on the same Wi-Fi, start Next.js on the main machine with a
LAN-accessible host, for example:

```sh
npm run dev -- --hostname 0.0.0.0
```

The second Mac should use `http://<main-mac-lan-ip>:3000` as
`ATTENDANCE_BACKEND_URL`.

## Worker setup

```sh
cd client_attendance_worker
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

If OpenCV does not install on Python 3.14, recreate the venv with Python 3.12:

```sh
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Face detector

The worker uses OpenCV YuNet by default:

```txt
ATTENDANCE_FACE_DETECTOR=yunet
ATTENDANCE_YUNET_MODEL_PATH=models/face_detection_yunet_2023mar.onnx
ATTENDANCE_YUNET_SCORE_THRESHOLD=0.8
ATTENDANCE_YUNET_NMS_THRESHOLD=0.3
ATTENDANCE_YUNET_TOP_K=5000
```

The default model file is stored at
`models/face_detection_yunet_2023mar.onnx` and comes from the official
[OpenCV Zoo YuNet model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

Set `ATTENDANCE_FACE_DETECTOR=haar` to use the previous Haar cascade detector.

## Example configs

Entry camera:

```txt
ATTENDANCE_BACKEND_URL=http://127.0.0.1:3000
ATTENDANCE_API_KEY=same-value-as-client-attendance-api-key
ATTENDANCE_CAMERA_ID=entry-camera-01
ATTENDANCE_DIRECTION=entry
ATTENDANCE_CAMERA_INDEX=0
```

Exit camera:

```txt
ATTENDANCE_BACKEND_URL=http://<main-mac-lan-ip>:3000
ATTENDANCE_API_KEY=same-value-as-client-attendance-api-key
ATTENDANCE_CAMERA_ID=exit-camera-01
ATTENDANCE_DIRECTION=exit
ATTENDANCE_CAMERA_INDEX=0
```

## Distance gate

The worker estimates distance with the detected face width:

```txt
distance_cm = (ATTENDANCE_FACE_KNOWN_WIDTH_CM * ATTENDANCE_CAMERA_FOCAL_LENGTH_PX) / face_width_px
```

To calibrate `ATTENDANCE_CAMERA_FOCAL_LENGTH_PX`, stand at a known distance
from the camera and use:

```txt
focal_length_px = (face_width_px * known_distance_cm) / ATTENDANCE_FACE_KNOWN_WIDTH_CM
```

For example, if a face is 150 px wide at 60 cm and the assumed real face width
is 15 cm, the focal length is `600`.

Press `Esc` or `q` in the preview window to stop.

## Passive liveness precheck

The worker runs two official checkpoints before sending a recognition request.
Place MiniFASNetV2 at `models/MiniFASNetV2.pth` and MiniFASNetV1SE at
`models/4_0_0_80x80_MiniFASNetV1SE.pth`.

Official model:

```txt
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/refs/heads/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth
```

`.pth` models use PyTorch. OpenCV DNN-compatible `.onnx` and `.caffemodel`
files can still be used by pointing `ATTENDANCE_LIVENESS_MODEL_PATH` at them.

Recommended starting point for the MacBook camera:

```txt
ATTENDANCE_MAX_DISTANCE_CM=90.0
ATTENDANCE_LIVENESS_ENABLED=true
ATTENDANCE_LIVENESS_MODEL_PATH=models/MiniFASNetV2.pth
ATTENDANCE_LIVENESS_V1SE_MODEL_PATH=models/4_0_0_80x80_MiniFASNetV1SE.pth
ATTENDANCE_LIVENESS_PRECHECK_ENABLED=true
ATTENDANCE_LIVENESS_PRECHECK_EXTRA_CM=50
ATTENDANCE_LIVENESS_PASS_TTL_SECONDS=3
ATTENDANCE_LIVENESS_MIN_FRAMES=12
ATTENDANCE_LIVENESS_THRESHOLD=0.50
ATTENDANCE_LIVENESS_CROP_PADDING_RATIO=0.85
ATTENDANCE_LIVENESS_LIVE_CLASS_INDEX=1
ATTENDANCE_LIVENESS_TRACK_MAX_GAP_SECONDS=0.75
ATTENDANCE_LIVENESS_SCREEN_RISK_THRESHOLD=0.62
ATTENDANCE_LIVENESS_MOTION_CHECK_ENABLED=true
ATTENDANCE_LIVENESS_MOTION_MIN_OBSERVATIONS=4
```

With these values, `40-90 cm` is the verify zone and `90-140 cm` is the
pre-liveness zone. Liveness must pass within the TTL before the worker sends the
JPEG crop to the backend.

Start with `ATTENDANCE_LIVENESS_THRESHOLD=0.50` to inspect real-face and spoof
scores from the actual camera setup. Raise it only after real faces pass
consistently under the store lighting.

An uncertain track never calls the attendance API. The preview/log asks the
person to walk through again and records signal scores plus reason codes. This
is still passive RGB PAD: it materially raises the bar for phone replay but
cannot provide the same assurance as depth/NIR hardware. Calibrate thresholds
with real users and the actual store lighting before production use.
