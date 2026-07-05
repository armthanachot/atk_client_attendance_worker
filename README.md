# ATK Store Client Attendance Worker

Python camera worker for the in-store face recognition PoC.

The worker:

- reads one local camera
- detects the largest face locally with OpenCV
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
`ATK_STORE_API_BASE_URL`.

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

## Example configs

Entry camera:

```txt
ATK_STORE_API_BASE_URL=http://127.0.0.1:3000
CLIENT_ATTENDANCE_API_KEY=same-value-as-client-attendance-api-key
ATTENDANCE_CAMERA_ID=entry-camera-01
ATTENDANCE_DIRECTION=entry
ATTENDANCE_CAMERA_INDEX=0
```

Exit camera:

```txt
ATK_STORE_API_BASE_URL=http://<main-mac-lan-ip>:3000
CLIENT_ATTENDANCE_API_KEY=same-value-as-client-attendance-api-key
ATTENDANCE_CAMERA_ID=exit-camera-01
ATTENDANCE_DIRECTION=exit
ATTENDANCE_CAMERA_INDEX=0
```

In exit mode, the worker still sends the recognition frame first. When the
backend returns a recognized visit with status `exited`, the worker calls the
server checkout endpoint for that visit. The worker does not send payment data,
clear carts, or calculate totals.

Older env names, `ATTENDANCE_BACKEND_URL` and `ATTENDANCE_API_KEY`, are still
accepted as fallbacks.

Press `Esc` or `q` in the preview window to stop.
