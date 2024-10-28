import cv2
from mtcnn import MTCNN
import face_recognition
import pickle
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from io import BytesIO

app = FastAPI()
logging.basicConfig(level=logging.INFO)

ENCODINGS_FILE = "/home/xloop/Downloads/high_value_customer/encodings/combined_images.pkl"
TOLERANCE = 0.6

# Load face encodings
data = None
recognized_individuals = set()

def load_encodings(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.error(f"Encodings file not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading encodings: {e}")
        return None

data = load_encodings(ENCODINGS_FILE)
if not data:
    raise ValueError("Encoding data could not be loaded.")

# Initialize MTCNN face detector
face_detector = MTCNN()

def recognize_faces(image, face_detector, data):
    # Convert to RGB for face recognition
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(rgb)
    response = {"recognized": False, "name": None}

    for face in faces:
        x, y, w, h = face['box']
        encodings = face_recognition.face_encodings(image, [(y, x + w, y + h, x)])
        
        if encodings:
            encodings = encodings[0]
            matches = face_recognition.compare_faces(data["encodings"], encodings, tolerance=TOLERANCE)
            
            if any(matches):
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                name = data["names"][matched_idxs[0]]
                
                # Draw a box around the face and label it with the name
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                
                # Update the response with the recognized face name
                response["recognized"] = True
                response["name"] = name
                break  # Stop after first recognized face
                
    return response

@app.post("/recognize_face")
async def recognize_face_endpoint(file: UploadFile = File(...)):
    try:
        # Read image bytes and convert to numpy array
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Perform face recognition
        result = recognize_faces(image, face_detector, data)

        # Return JSON response
        return JSONResponse(content=result)
    
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
