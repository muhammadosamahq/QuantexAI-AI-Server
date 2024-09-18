# VGG-Face, Google, FaceNet, Open-Face, Facebook DeepFace, DeepID, ArcFace, Dlib

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import os
from deepface import DeepFace
from typing import Optional

app = FastAPI()

REGISTERED_FACES_DIR = 'registered_faces'
os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)

class Face(BaseModel):
    name: str

@app.post("/register_face/")
async def register_face(file: UploadFile = File(...), name: str = Form(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    temp_file_path = 'temp.jpg'
    cv2.imwrite(temp_file_path, img)
    try:
        embeddings = DeepFace.represent(img_path=temp_file_path, model_name='VGG-Face', enforce_detection=False)
        face_encoding = embeddings[0]['embedding']
        face_filename = os.path.join(REGISTERED_FACES_DIR, f"{name}.npy")
        np.save(face_filename, face_encoding)
        os.remove(temp_file_path)
        return {"message": "Face registered successfully"}
    except Exception as e:
        os.remove(temp_file_path)
        return {"message": f"Error: {str(e)}"}

@app.get("/list_registered_faces/")
def list_registered_faces():
    registered_faces = [f.replace('.npy', '') for f in os.listdir(REGISTERED_FACES_DIR)]
    return {"registered_faces": registered_faces}

@app.get("/get_face_encoding/{name}")
def get_face_encoding(name: str):
    face_filename = os.path.join(REGISTERED_FACES_DIR, f"{name}.npy")
    if os.path.exists(face_filename):
        face_encoding = np.load(face_filename)
        return {"name": name, "encoding": face_encoding.tolist()}
    else:
        raise HTTPException(status_code=404, detail="Face not found")

@app.post("/detect_emotion/")
async def detect_emotion(file: UploadFile = File(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    temp_file_path = 'temp.jpg'
    cv2.imwrite(temp_file_path, img)
    try:
        analysis = DeepFace.analyze(img_path=temp_file_path, actions=['emotion'], enforce_detection=False)
        emotion = analysis["dominant_emotion"]
        os.remove(temp_file_path)
        return {"dominant_emotion": emotion}
    except Exception as e:
        os.remove(temp_file_path)
        return {"message": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)