import cv2
import pickle
import numpy as np
from deepface import DeepFace

def check_registered_faces(video_path, frame_width=640, frame_height=480, embedding_file="src/vision/embeddings/Harry_face_embedding.pkl"):
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
            # Use DeepFace to extract faces
            faces = DeepFace.extract_faces(img_path=frame, detector_backend='mtcnn', enforce_detection=False)

        except ValueError as e:
            print(f"{str(e)}")
            faces = []

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for face in faces:
                face_image = face['face']
                facial_area = face['facial_area']
                cv2.rectangle(frame, 
                              (facial_area['x'], facial_area['y']), 
                              (facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h']), 
                              (0, 255, 0), 2)

                match_found = False
                for name, registered_face_path in registered_faces.items():
                    # Use DeepFace.verify to check if the faces match
                    print("face_image:", face_image)
                    print("register_face_image:", registered_face_path)
                    result = DeepFace.verify(face_image, registered_face_path, model_name='Facenet')
                    if result['verified']:
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

check_registered_faces("src/vision/video_2.mp4")
