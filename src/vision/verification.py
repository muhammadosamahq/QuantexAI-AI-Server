import cv2
import pickle
import numpy as np
from deepface import DeepFace

def compare_embeddings(embedding1, embedding2, threshold=0.6):
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity > threshold

def check_registered_faces(video_path, frame_width=640, frame_height=480, embedding_file="registered_faces.pkl"):
    try:
        with open(embedding_file, "rb") as f:
            registered_faces = pickle.load(f)
    except FileNotFoundError:
        print("No registered faces found.")
        return
    
    cap = cv2.VideoCapture(video_path)

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
            cv2.putText(frame, "No face detected", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for face in faces:
                face_embedding = face['face']
                facial_area = face['facial_area']
                cv2.rectangle(frame, 
                              (facial_area['x'], facial_area['y']), 
                              (facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h']), 
                              (0, 255, 0), 2)

                match_found = False
                for name, registered_embedding in registered_faces.items():
                    if compare_embeddings(face_embedding, registered_embedding):
                        match_found = True
                        cv2.putText(frame, f"Matched: {name}", 
                                    (facial_area['x'], facial_area['y'] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        break

                if not match_found:
                    cv2.putText(frame, "No match found", 
                                (facial_area['x'], facial_area['y'] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Video Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

check_registered_faces("video_test.mp4")
