from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import cv2
import numpy as np

mtcnn = MTCNN()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(pil_image)
    if boxes is not None:
        draw = ImageDraw.Draw(pil_image)
        for box in boxes:
            draw.rectangle(box.tolist(), outline='red', width=3)
        print("Face Detected: True")
    else:
        print("Face Detected: False")

    frame_with_boxes = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imshow('Webcam Feed', frame_with_boxes)
    if boxes is not None:
        pil_image.save('webcam_image.jpeg')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()