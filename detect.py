from ultralytics import YOLO
import cv2

model = YOLO("best.pt")   # <-- use the new model

results = model.predict(
    source="images/i.jpg",
    conf=0.20
)

for r in results:
    img = r.plot()

cv2.imshow("Helmet Test", img)
cv2.imwrite("output/result.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()