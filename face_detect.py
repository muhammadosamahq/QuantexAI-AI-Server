# VGG-Face, Google, FaceNet, Open-Face, Facebook DeepFace, DeepID, ArcFace, Dlib
from deepface import DeepFace
import cv2
import numpy as np
import pickle

def process_video(video_path, frame_width=640, frame_height=480):
    cap = cv2.VideoCapture(video_path)
    face_detected_count = 0
    registered_faces = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))

        try:
            faces = DeepFace.extract_faces(img_path=frame, detector_backend='mtcnn', enforce_detection=False)
        except ValueError as e:
            print(f"{str(e)}")
            faces = []

        if len(faces) == 0:
            face_detected_count = 0
            cv2.putText(frame, "Face not detected", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:    
            best_face = None
            highest_confidence = 0.0

            for face in faces:
                confidence = face['confidence']
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_face = face

            if best_face and highest_confidence > 0.80:
                print("Best_face:", best_face)
                facial_area = best_face['facial_area']
                cv2.rectangle(frame, 
                              (facial_area['x'], facial_area['y']), 
                              (facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h']), 
                              (0, 255, 0), 2)
                cv2.putText(frame, f'Face: {highest_confidence:.2f}', 
                            (facial_area['x'], facial_area['y'] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, "Face Detected", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                face_detected_count += 1
                if face_detected_count == 5:
                    name = input("Please enter your name for registration: ")
                    face_embedding = best_face['face']
                    registered_faces[name] = face_embedding
                    embeddings = face_embedding
                    print(f"Registered {name} with embeddings.")
                    face_detected_count = 0
                    file_name = f"{name}_face_embedding.pkl"
                    with open(file_name, "wb") as f:
                        pickle.dump(registered_faces, f)
                    cap.release()
                    cv2.destroyAllWindows()
            else:
                face_detected_count = 0
                cv2.putText(frame, "Face not detected", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Video Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cap.release()
    cv2.destroyAllWindows()
    return name, embeddings


name, embeddings = process_video("video_2.mp4")
print("name:", name)
print("embedding:", embeddings)