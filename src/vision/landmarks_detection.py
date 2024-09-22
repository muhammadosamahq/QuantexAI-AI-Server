import cv2
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize Mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    _, image = cap.read()
    
    # Convert the frame to RGB, as required by Mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect facial landmarks
    result = face_mesh.process(rgb_image)
    
    # If landmarks are detected
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw facial landmarks on the image
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
    
    # Show the output image with the face landmarks
    cv2.imshow("Output", image)
    
    # Break the loop if 'Esc' is pressed
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Release the video capture and close windows
cv2.destroyAllWindows()
cap.release()
