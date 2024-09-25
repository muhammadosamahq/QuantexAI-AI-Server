import cv2
import requests
import numpy as np
import concurrent.futures

# FastAPI endpoint
PREDICT_API_URL = "http://127.0.0.1:8000/predict"
frame_width = 640
frame_height = 480

def make_request(url, files):
    response = requests.post(url, files=files)
    return response

def process_video():

    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

  
        frame = cv2.resize(frame, (frame_width, frame_height))
  
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        files = {'file': ('frame.jpg', img_bytes, 'image/jpeg')}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            
            future_predict = executor.submit(make_request, PREDICT_API_URL, files)
            response_predict = future_predict.result()

        if response_predict.status_code == 200:
            result_predict = response_predict.json()
            emotion = result_predict.get("emotion", "unknown")
            category = result_predict.get("category", "unknown")
        else:
            emotion = "unknown"
            category = "unknown"
            print("Failed to get response from predict API")


        x, y = 50, 50

        cv2.putText(frame, f"Emotion: {emotion}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        print("Person: ", emotion)
        #cv2.putText(frame, f"Category: {category}", (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()