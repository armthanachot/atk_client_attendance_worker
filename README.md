# ATK Store Client Attendance Worker

Python camera worker for the in-store face recognition PoC.

The worker:

- reads one local camera
- detects the largest face locally with OpenCV YuNet
- estimates face distance from the detected face width
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
