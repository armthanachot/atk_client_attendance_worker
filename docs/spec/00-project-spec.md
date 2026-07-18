# Camera Worker Liveness Contract

## Goal

Recognize a person while they walk through the camera zone without an active
challenge. Before any attendance request, the worker must establish one
continuous person track and combine two MiniFASNet checkpoints, full-frame
screen cues, and temporal motion/parallax cues.

## Security behavior

- MiniFASNet class index `1` is the only real-face class.
- MiniFASNetV2 and MiniFASNetV1SE must both load when `.pth` liveness is enabled.
- Evidence never transfers between discontinuous face tracks.
- A screen-risk threshold breach, flat homography-like motion, model
  disagreement, insufficient temporal evidence, or a fused score below the
  threshold must not call the attendance API.
- The exception action is to ask the person to walk through again. No
  attendance record is created for an uncertain track.
- Signal scores, reason codes, and track ID are included in recognition
  metadata for accepted tracks and in local logs for rejected tracks.

## Operational boundary

This is passive RGB presentation-attack detection. It raises the cost of phone
and print replay but is not equivalent to depth/NIR capture. Production
thresholds require calibration against real users and attacks under final
camera placement and lighting.
