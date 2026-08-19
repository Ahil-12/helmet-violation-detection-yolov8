
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

cap = cv2.VideoCapture("videos/test.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        conf=0.30
    )

    for r in results:

        if r.boxes is None:
            continue

        print("IDs:", r.boxes.id)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()