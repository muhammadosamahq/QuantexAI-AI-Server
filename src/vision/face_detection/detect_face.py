import requests
import threading
import time
import cv2
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in a given frame
def detect_faces_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0, faces

# Function to show video and detect faces in real-time
def show_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Could not open camera.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces in the current frame
        face_detected, faces = detect_faces_in_frame(frame)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Show the video feed with face detection
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# FastAPI setup for face detection API
app = FastAPI()

class FaceDetectionResponse(BaseModel):
    face_detected: bool

@app.get("/detect_face", response_model=FaceDetectionResponse)
def detect_face():
    # Open the camera and read a frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"face_detected": False}
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return {"face_detected": False}
    
    # Detect faces in the frame
    face_detected, _ = detect_faces_in_frame(frame)
    return {"face_detected": face_detected}

# Start threads for video and API
video_thread = threading.Thread(target=show_video)
video_thread.daemon = True
video_thread.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
