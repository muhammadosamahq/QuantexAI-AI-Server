from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
import torch
import os
from PIL import Image

# Initialize face detection model and embedding model
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def load_face_embeddings():
    """Load saved embeddings from the current directory."""
    embeddings = {}
    for file in os.listdir():
        if file.endswith('_embedding.pt'):
            name = file.split('_')[0]
            embeddings[name] = torch.load(file)
    return embeddings

def verify_face(embeddings):
    """Capture a frame and verify against stored embeddings."""
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
            face_crops = [pil_image.crop(box) for box in boxes]
            for face in face_crops:
                face_tensor = mtcnn(face)
                if face_tensor is not None:
                    face_embedding = resnet(face_tensor)
                    verify_match(face_embedding, embeddings)
                else:
                    print("Warning: Could not extract face embedding.")
        else:
            print("No face detected.")

        cv2.imshow('Webcam Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def verify_match(face_embedding, embeddings):
    """Compare the captured embedding with saved embeddings."""
    for name, saved_embedding in embeddings.items():
        similarity = torch.nn.functional.cosine_similarity(face_embedding, saved_embedding).item()
        print(f"Similarity with {name}: {similarity:.4f}")
        if similarity > 0.8:
            print(f"Match found: {name}")
            return
    
    print("No match found.")

if __name__ == "__main__":
    embeddings = load_face_embeddings()
    verify_face(embeddings)
