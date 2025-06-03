from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # lightweight model

img = cv2.imread('test.jpeg')  # any sample image with people
results = model(img)

# Draw results
for result in results:
    for box in result.boxes:
        if int(box.cls[0]) == 0:  # Class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
