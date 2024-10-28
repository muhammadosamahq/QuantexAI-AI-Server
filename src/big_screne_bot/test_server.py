import cv2
import requests
import numpy as np

# FastAPI endpoint URL
url = "http://127.0.0.1:8000/recognize_face"

# Open a connection to the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to capture image")
        break

    # Encode the frame as a JPEG to send to the server
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Send the frame to the FastAPI server
    response = requests.post(url, files={"file": ("frame.jpg", img_bytes, "image/jpeg")})

    # Process the server response
    if response.status_code == 200:
        result = response.json()
        if result["recognized"]:
            name = result["name"]
            cv2.putText(frame, f"Recognized: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face Not Recognized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Error in recognition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to break the loop and close the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
