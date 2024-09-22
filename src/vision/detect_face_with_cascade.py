import cv2
import time
import os
import psutil
from deepface import DeepFace
import numpy as np

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def pos_bottom_left(frame, grid=29):
    height = int(frame.shape[0])
    dim = (10, int(height * (grid / 30)))
    return dim

def save_embedding(img, face):
    try:
        # Extract face embeddings using DeepFace
        embedding = DeepFace.represent(img, model_name='Facenet')[0]["embedding"]
        # Save the embeddings to a file
        np.save(f"embeddings/face_embedding_{int(time.time())}.npy", embedding)
        print("Face embeddings saved successfully!")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def face_detect(write_to_audit=False):
    face_cascade = cv2.CascadeClassifier('my_face_emotion/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_time = time.time()
        ret, frame = cap.read()
        img = rescale_frame(frame, percent=50)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        bColor = (255, 0, 0)
        gColor = (0, 255, 0)
        rColor = (0, 0, 255)

        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            face_confidence = (w * h) / (img.shape[0] * img.shape[1]) * 100  # Confidence based on face area
            print(f"Face detected with confidence: {face_confidence}%")

            if face_confidence > 40:
                save_embedding(face_img, (x, y, w, h))
            
            # Draw rectangles around detected faces
            if len(faces) % 3 == 0:
                cv2.rectangle(img, (x, y), (x+w, y+h), rColor, 2)
            elif len(faces) % 3 == 1:
                cv2.rectangle(img, (x, y), (x+w, y+h), gColor, 2)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), bColor, 2)

        # CPU and Memory load
        cpu_load = psutil.cpu_percent()
        mem_load = psutil.virtual_memory().percent
        compute_fps = 1.0 / (time.time() - start_time)

        if write_to_audit:
            with open("audit.txt", "a+") as file:
                file.write(f"{fps},{compute_fps},{cpu_load},{mem_load}\n")

        # Display FPS and system usage
        cv2.putText(img, f"V-FPS: {fps}, C-FPS: {compute_fps:.2f}", pos_bottom_left(img), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 8)
        cv2.putText(img, f"CPU: {cpu_load}%", pos_bottom_left(img, 27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 8)
        cv2.putText(img, f"Memory: {mem_load}%", pos_bottom_left(img, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 8)

        cv2.imshow('Detected face', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27 or k == 13:  # ESC or ENTER to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    args = sys.argv[1] if len(sys.argv) > 1 else False
    face_detect(bool(args))
