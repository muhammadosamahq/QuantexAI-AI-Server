import requests
import time

# URL of the FastAPI endpoint
url = "http://localhost:8000/detect_face"

while True:
    try:
        # Call the FastAPI endpoint to detect faces
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Check if a face was detected
            if data.get("face_detected"):
                print("Face detected!")
            else:
                print("No face detected.")
        else:
            print(f"Error: Received status code {response.status_code}")

        # Wait for 1 second before calling again
        time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(1)
