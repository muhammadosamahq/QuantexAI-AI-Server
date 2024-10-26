from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import cv2
import numpy as np
from fastapi import FastAPI, BackgroundTasks
import torch

app = FastAPI()

# Initialize face detection model and embedding model
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def save_face_embedding(name, embedding):
    """Save the embedding tensor with the user's name."""
    torch.save(embedding, f"{name}_embedding.pt")
    print(f"Embedding saved for {name}")

def start_face_detection():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

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
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if boxes is not None:
                name = input("Enter the user's name: ")
                face_crops = [pil_image.crop(box) for box in boxes]
                for face in face_crops:
                    face_tensor = mtcnn(face)
                    if face_tensor is not None:
                        face_embedding = resnet(face_tensor)
                        save_face_embedding(name, face_embedding)
                    else:
                        print("Warning: Could not extract face embedding.")

    cap.release()
    cv2.destroyAllWindows()

@app.get("/start-detection")
async def start_detection(background_tasks: BackgroundTasks):
    background_tasks.add_task(start_face_detection)
    return {"message": "Face detection process started in the background."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)