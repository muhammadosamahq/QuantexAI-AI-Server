from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from deepface import DeepFace
import cv2
import numpy as np
import os
import json
import time
from scipy.spatial.distance import cosine

app = FastAPI()
 # Flag to check if face has been registered

# Emotion mapping dictionary
emotion_mapping = {
    "happy": "positive",
    "surprise": "positive",
    "sad": "negative",
    "angry": "negative",
    "fear": "negative",
    "disgust": "negative",
    "neutral": "neutral"
}

class PredictionResponse(BaseModel):
    emotion: str
    category: str


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Analyze the image for emotions
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]  # Take the first face's result

        emotion = analysis['dominant_emotion']
        emotion_category = emotion_mapping.get(emotion, "neutral")

        return {"emotion": emotion, "category": emotion_category}
    except Exception as e:
        return {"emotion": "error", "category": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)