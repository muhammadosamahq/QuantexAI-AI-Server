import cv2
import requests
import numpy as np
from io import BytesIO

API_URL = "http://localhost:8000/recognize/"

def capture_and_send_frame(api_url):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    train_captured = False
    train_image = None
    last_status = "No face processed"  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.putText(frame, last_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "recognized" in last_status else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1)

        if key == ord('t'):  
            if not train_captured:
                print("Capturing training face...")
                _, buffer = cv2.imencode('.jpg', frame)
                train_image = BytesIO(buffer)
                train_captured = True
                last_status = "Training image captured"
            else:
                last_status = "Training image already captured. Press 't' to recapture."

        elif key == ord('c') and train_captured: 
            _, buffer_test = cv2.imencode('.jpg', frame)
            test_image = BytesIO(buffer_test)

            files = {
                'train_image': ('train_image.jpg', train_image.getvalue(), 'image/jpeg'),
                'test_image': ('test_image.jpg', test_image.getvalue(), 'image/jpeg')
            }

            last_status = "Recognizing . . ." 
            try:
                response = requests.post(api_url, files=files)
                response.raise_for_status()  
                result = response.json()

                print(f"API response: {result}")

                if result.get("status") == "recognized":
                    last_status = "Face recognized"
                else:
                    last_status = "Face not recognized"

           
                train_image = None
                train_captured = False

            except requests.RequestException as e:
                last_status = f"Error during recognition: {str(e)}"
            except ValueError:
                last_status = "Error processing response from API"

        elif key == ord('q'):  
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_send_frame(API_URL)