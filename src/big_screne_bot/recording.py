# import cv2
# import requests
# import numpy as np
# import pyaudio
# import wave
# import time

# # FastAPI endpoint URL
# url = "http://127.0.0.1:8000/recognize_face"

# # Open a connection to the webcam
# video_capture = cv2.VideoCapture(0)

# # Audio settings
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100

# p = pyaudio.PyAudio()
# stream = None
# audio_frames = []
# current_person = None

# def start_audio_recording():
#     """Initialize the audio stream and start recording."""
#     global stream, audio_frames
#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#     audio_frames = []
#     print("Audio recording started")

# def stop_audio_recording(name):
#     """Stop recording and save the audio file with the person's name."""
#     global stream, audio_frames
#     stream.stop_stream()
#     stream.close()
#     file_name = f"{name}_{int(time.time())}.wav"
#     with wave.open(file_name, 'wb') as wf:
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(p.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(audio_frames))
#     print(f"Audio recording saved as {file_name}")

# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
    
#     if not ret:
#         print("Failed to capture image")
#         break

#     # Encode the frame as a JPEG to send to the server
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     img_bytes = img_encoded.tobytes()

#     # Send the frame to the FastAPI server
#     response = requests.post(url, files={"file": ("frame.jpg", img_bytes, "image/jpeg")})

#     # Process the server response
#     if response.status_code == 200:
#         result = response.json()
#         if result["recognized"]:
#             name = result["name"]
#             cv2.putText(frame, f"Recognized: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
#             if current_person != name:
#                 # New person recognized, start recording
#                 if current_person is not None:
#                     stop_audio_recording(current_person)
#                 current_person = name
#                 start_audio_recording()
#             # Continue recording audio while the person is recognized
#             audio_data = stream.read(CHUNK)
#             audio_frames.append(audio_data)
#         else:
#             # If no person is recognized and we were recording someone, stop recording
#             if current_person is not None:
#                 stop_audio_recording(current_person)
#                 current_person = None
#             cv2.putText(frame, "Face Not Recognized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     else:
#         cv2.putText(frame, "Error in recognition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # Display the resulting frame
#     cv2.imshow('Video', frame)

#     # Press 'q' to break the loop and close the application
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Stop recording if someone was still being recorded when the script ends
# if current_person is not None:
#     stop_audio_recording(current_person)

# # When everything is done, release the capture and audio resources
# video_capture.release()
# cv2.destroyAllWindows()
# p.terminate()



import cv2
import requests
import numpy as np
import pyaudio
import wave
import base64
import os
import time
import threading

# FastAPI endpoint URL for facial recognition
url = "http://127.0.0.1:8000/recognize_face"

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
AUDIO_FOLDER = "recordings"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Global variables to manage recording state
recording = False
frames = []
stream = None

def start_recording():
    global recording, frames, stream
    recording = True
    frames = []
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")

    while recording:
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop the stream and save the recording when recording is set to False
    stream.stop_stream()
    stream.close()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    wave_filename = os.path.join(AUDIO_FOLDER, f"recording_{timestamp}.wav")
    wf = wave.open(wave_filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    print(f"Recording complete. Saved as {wave_filename}")

def process_facial_recognition():
    global recording

    # Open a connection to the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image")
            break

        # Encode the frame as a JPEG to send to the server
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        # Send the frame to the FastAPI server
        response = requests.post(url, files={"file": ("frame.jpg", img_bytes, "image/jpeg")})

        # Process the server response
        if response.status_code == 200:
            result = response.json()
            if result["recognized"]:
                name = result["name"]
                cv2.putText(frame, f"Recognized: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Start the recording thread if not already recording
                if not recording:
                    recording_thread = threading.Thread(target=start_recording)
                    recording_thread.start()
            else:
                cv2.putText(frame, "Face Not Recognized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Stop recording if person is no longer recognized
                if recording:
                    recording = False

        else:
            cv2.putText(frame, "Error in recognition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Press 'q' to break the loop and close the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

# Run facial recognition in the main thread
process_facial_recognition()

# When everything is done, terminate PyAudio
p.terminate()
