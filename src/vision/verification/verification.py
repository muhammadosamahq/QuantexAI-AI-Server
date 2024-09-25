from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import cv2
import pickle
import numpy as np
from deepface import DeepFace
import os
import shutil

app = FastAPI()


embedding_dir = "src/vision/verification/embeddings"
os.makedirs(embedding_dir, exist_ok=True)


@app.post("/check_faces/")
async def check_faces(video_file: UploadFile = File(...)):
    video_path = f"temp_video/{video_file.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video_file.file, f)

    results = check_registered_faces(video_path)
    return results


def process_frame(video_path, frame_width=640, frame_height=480):
    cap = cv2.VideoCapture(video_path)
    face_detected_count = 0
    registered_faces = {}
    name = None
    embeddings = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))

        try:
            faces = DeepFace.extract_faces(img_path=frame, detector_backend='mtcnn', enforce_detection=False)
        except ValueError as e:
            print(f"{str(e)}")
            faces = []

        if len(faces) == 0:
            cv2.putText(frame, "Face not detected", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:    
            best_face = None
            highest_confidence = 0.0

            for face in faces:
                confidence = face['confidence']
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_face = face

            if best_face and highest_confidence > 0.80:
                facial_area = best_face['facial_area']
                cv2.rectangle(frame, 
                              (facial_area['x'], facial_area['y']), 
                              (facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h']), 
                              (0, 255, 0), 2)
                face_detected_count += 1
                if face_detected_count == 5:
                    name = input("Please enter your name for registration: ")
                    face_embedding = best_face['face']
                    registered_faces[name] = face_embedding
                    embeddings = face_embedding
                    face_detected_count = 0
                    break
            else:
                face_detected_count = 0

        cv2.imshow('Video Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if name and embeddings:
        return name, embeddings
    else:
        return None, None


def check_registered_faces(video_path, frame_width=640, frame_height=480):
    results = []
    try:
        registered_faces = {}
        for file in os.listdir(embedding_dir):
            with open(os.path.join(embedding_dir, file), "rb") as f:
                registered_faces.update(pickle.load(f))
    except FileNotFoundError:
        print("No registered faces found.")
        return
    
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (frame_width, frame_height))

        try:
            faces = DeepFace.extract_faces(img_path=frame, detector_backend='mtcnn', enforce_detection=False)

        except ValueError as e:
            print(f"{str(e)}")
            faces = []

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for face in faces:
                face_image = face['face']
                facial_area = face['facial_area']
                cv2.rectangle(frame, 
                              (facial_area['x'], facial_area['y']), 
                              (facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h']), 
                              (0, 255, 0), 2)

                match_found = False
                for name, registered_face_path in registered_faces.items():
                    result = DeepFace.verify(face_image, registered_face_path, model_name='Facenet')
                    if result['verified']:
                        match_found = True
                        results.append({"name": name, "status": "matched"})
                        break

                if not match_found:
                    results.append({"status": "no match found"})

        cv2.imshow('Video Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
