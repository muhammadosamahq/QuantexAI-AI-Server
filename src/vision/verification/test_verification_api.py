import cv2
import httpx

def capture_and_send_frame():
    url = "http://localhost:8000/check_faces/"
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            _, buffer = cv2.imencode('.jpg', frame)
            response = httpx.post(url, files={"frame": ("frame.jpg", buffer.tobytes(), "image/jpeg")})
            print("Response:", response.json())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_send_frame()
