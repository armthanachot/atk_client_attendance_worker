# Face Distance Calculation

## Env
- DISTANCE_GATE_ENABLED เปิด/ปิดการกรองด้วยระยะ
- FACE_KNOWN_WIDTH_CM ความกว้างใบหน้าโดยประมาณ เช่น 15 cm
- CAMERA_FOCAL_LENGTH_PX ค่ากล้องหลัง calibrate
- MIN_DISTANCE_CM ใกล้สุดที่ยอมรับ
- MAX_DISTANCE_CM ไกลสุดที่ยอมรับ

## CAMERA_FOCAL_LENGTH_PX ทำไมถึงต้อง 600?

มันเป็นค่าคงที่ของกล้อง ในหน่วย pixel ที่ใช้แปลงจากขนาดของใบหน้า เป็นระยะจริง

> distance_cm = (face_width_cm * focal_length_px) / face_width_px

โดย:
- face_width_px = ความกว้างหน้าที่ OpenCV เห็นในภาพ หน่วย pixel
- focal_length_px = ค่าคงที่ของกล้อง หน่วย pixel
- face_width_cm = ความกว้างหน้าจริงโดยประมาณ หน่วย cm
- ผลลัพธ์ distance_cm = ระยะห่างจากกล้อง หน่วย cm

example
```txt
face_width_cm = 15
focal_length_px = 600
face_width_px = 150

distance_cm = (15 * 600) / 150
distance_cm = 60 cm
```

600 ไม่ใช่สูตรตายตัว มันเป็นค่าตัวอย่างของ `focal_length_px` ที่ต้องมีการ calibrate จากกล้องจริง

มันมาจากสูตร `pinhole camera model`

> distance_cm = (known_face_width_cm × focal_length_px) ÷ face_width_px

ถ้าเรารู้ระยะจริงตอน calibrate เช่น:

```txt
known_distance_cm = 60
known_face_width_cm = 15
face_width_px = 150
```

เราย้ายสูตรเพื่อหา focal length:
> focal_length_px = (face_width_px × known_distance_cm) ÷ known_face_width_cm

แทนค่า
```txt
focal_length_px = (150 × 60) ÷ 15
focal_length_px = 600
```

ดังนั้น 600 มาจาก
- คนยืนห่างกล้องจริง 60 cm
- OpenCV ตรวจเจอหน้ากว้าง 150 px
- สมมติความกว้างหน้าจริง 15 cm

ถ้ากล้อง/ความละเอียด/มุมกล้องเปลี่ยน ค่าอาจไม่ใช่ 600 เช่นถ้า calibrate แล้ว face box (ความหว้างในหน่วย px) ได้ 180 px ที่ระยะ 60 cm
> focal_length_px = (180 × 60) ÷ 15 = 720