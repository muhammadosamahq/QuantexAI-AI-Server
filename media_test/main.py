from fastapi import FastAPI, HTTPException
from src.utils import check_webcam
from src.utils import check_microphone

app = FastAPI()


@app.get("/check-devices")
def check_devices():
    video_status = check_webcam()
    audio_status = check_microphone()

    # Initialize a response dictionary to track the status of each device
    response = {
        "video_status": video_status,
        "audio_status": audio_status,
    }

    # Determine the overall status based on individual device checks
    if "working properly" in video_status and "working properly" in audio_status:
        response["status"] = "OK"
        response["message"] = "Both webcam and microphone are working properly."
    elif "working properly" in video_status:
        response["status"] = "Warning"
        response["message"] = "Camera is working, but microphone is not."
    elif "working properly" in audio_status:
        response["status"] = "Warning"
        response["message"] = "Microphone is working, but camera is not."
    else:
        raise HTTPException(
            status_code=500, detail="Both devices are not functioning properly."
        )

    return response


# Camera
# sudo modprobe -r uvcvideo # for down the camera
# sudo modprobe uvcvideo # for up the camera
# lsmod | grep uvcvideo  # This should return no output if disabled

# audio
# amixer set Capture nocap # for mute
# amixer set Capture cap # for unmute
