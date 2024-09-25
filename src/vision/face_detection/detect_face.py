from fastapi import FastAPI
from pydantic import BaseModel
import threading
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Error: Could not open camera.")
def detect_faces_in_frame():
    ret, frame = cap.read()
    if not ret:
        raise Exception("Error: Could not read frame from the camera.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0
def show_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
app = FastAPI()
class FaceDetectionResponse(BaseModel):
    face_detected: bool
# video_thread = threading.Thread(target=show_video)
# video_thread.daemon = True
# video_thread.start()

@app.get("/detect_face", response_model=FaceDetectionResponse)
def detect_face():
    face_detected = detect_faces_in_frame()
    return {"face_detected": face_detected}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)