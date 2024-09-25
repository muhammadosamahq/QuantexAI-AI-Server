import cv2
from fastapi import FastAPI
from pydantic import BaseModel
import threading

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Error: Could not open camera.")

# Define a response model
class FaceDetectionResponse(BaseModel):
    face_detected: bool

# A function to display camera feed with bounding boxes
def show_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show the frame with detected faces
        cv2.imshow("Face Detection", frame)

        # Exit if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Start the video thread to show the camera window with bounding boxes
video_thread = threading.Thread(target=show_video)
video_thread.daemon = True
video_thread.start()

@app.get("/detect_face", response_model=FaceDetectionResponse)
def detect_face():
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        raise Exception("Error: Could not read frame from the camera.")

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Return True if any face is detected, otherwise False
    return {"face_detected": len(faces) > 0}

# Make sure to run this code using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

