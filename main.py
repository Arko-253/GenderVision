import cv2
import torch
from ultralytics import YOLO
from deep_sort_tracker import create_tracker
from gender_classifier import classify_gender_resnet  # updated file name

# ✅ Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Device: {device}")

# ✅ Load YOLOv8n-face
detector = YOLO("yolov8n-face.pt").to(device)

# ✅ Load Deep SORT
tracker = create_tracker()

# Gender memory
gender_map = {}

# 🎥 Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found.")
    exit()

print("🔄 Real-time Face Tracking + Gender Detection Started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    results = detector(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf < 0.6:
            continue
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, "face"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        l, t = max(0, l), max(0, t)
        r, b = min(frame.shape[1], r), min(frame.shape[0], b)

        face_crop = frame[t:b, l:r]
        h, w = face_crop.shape[:2]

        # Skip poor crops
        if h < 60 or w < 60:
            continue
        aspect_ratio = w / h
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        if aspect_ratio < 0.6 or aspect_ratio > 1.5 or brightness < 40:
            continue

        # Classify gender
        if track_id not in gender_map:
            try:
                gender, conf = classify_gender_resnet(face_crop, track_id)
            except Exception as e:
                print(f"[!] Gender classification error: {e}")
                gender = "Unknown"
            gender_map[track_id] = gender
        else:
            gender = gender_map[track_id]

        # Draw bounding box and gender
        color = (255, 0, 0) if gender == "Male" else (0, 255, 0) if gender == "Female" else (0, 0, 255)
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, gender, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Gender count
    visible_ids = [t.track_id for t in tracks if t.is_confirmed()]
    visible_genders = [gender_map.get(i, "Unknown") for i in visible_ids]
    males = visible_genders.count("Male")
    females = visible_genders.count("Female")

    cv2.putText(frame, f"Males: {males}  Females: {females}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Real-Time Face Tracking + Gender Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
