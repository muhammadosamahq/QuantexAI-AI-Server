import cv2
import pyaudio
import numpy as np
import subprocess


def check_webcam() -> str:
    cap = cv2.VideoCapture(0)  # Try to open a non-existent camera index
    print(f"Is camera opened: {cap.isOpened()}")  # Debug line

    if not cap.isOpened():
        return "Camera not accessible. It might be disabled or used by another application."

    ret, frame = cap.read()
    print(f"Frame captured: {ret}, Frame: {frame}")  # Debug line
    cap.release()

    if not ret or frame is None:
        return "Camera is available but not streaming video."

    return "Webcam is working properly."


result = check_webcam()
print(result)


def is_microphone_muted() -> bool:
    # Check microphone status using amixer
    result = subprocess.run(
        ["amixer", "get", "Capture"], capture_output=True, text=True
    )
    return "[off]" in result.stdout


def check_microphone(seconds: int = 3) -> str:
    if is_microphone_muted():
        return "Microphone is muted, please unmute it."

    # Existing microphone check code
    try:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
        )

        print("Recording audio...")
        audio_data = []

        for _ in range(0, int(44100 / 1024 * seconds)):
            data = stream.read(1024, exception_on_overflow=False)
            audio_data.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()
        p.terminate()

        if np.max(np.hstack(audio_data)) > 100:  # Adjust the threshold as necessary
            return "Microphone is working properly."
        else:
            return "Microphone available but no audio input detected."

    except OSError as e:
        return f"Microphone not accessible: {e}"
