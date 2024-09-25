import requests
import threading
import time
import cv2

url = "http://127.0.0.1:8000/detect_face"

def test_face_detection():
    while True:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"Face detected: {data['face_detected']}")
        else:
            print(f"Failed to connect to API: Status Code {response.status_code}")
        time.sleep(2)

def show_video():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

video_thread = threading.Thread(target=show_video)
video_thread.daemon = True
video_thread.start()

api_thread = threading.Thread(target=test_face_detection)
api_thread.daemon = True
api_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting program.")