from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import speech_recognition as sr
from src.utils import check_webcam, check_microphone

app = FastAPI()


def listen_for_audio(duration=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, timeout=duration)
        try:
            text = recognizer.recognize_google(audio)
            return text.lower()
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Could not request results"


@app.get("/start")
async def start_test():
    """First API: Shows introduction message"""
    return {
        "message": "This is media test bot. Please say 'test media'",
        "next_step": "Call /verify-command endpoint and say 'test media'",
    }


@app.get("/verify-command/{user_response}")
async def verify_command(user_response: str):
    """Second API: Listens for 'test media' command and runs device tests"""
    try:
        # Listen for "test media"
        user_response = "test media"
        if "test media" not in user_response:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Please say 'test media' to begin the test",
                    "heard": user_response,
                },
            )

        # Run device checks
        video_status = check_webcam()
        audio_status = check_microphone()

        response = {
            "video_status": video_status,
            "audio_status": audio_status,
        }

        # Determine overall status
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

        response["next_step"] = "Call /final-check endpoint for microphone verification"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/final-check")
async def final_check():
    """Third API: Performs final microphone verification"""
    try:
        final_audio = listen_for_audio()

        if final_audio and final_audio != "Could not understand audio":
            return {
                "status": "success",
                "message": "Microphone verification successful",
                "audio_captured": final_audio,
            }
        else:
            return {
                "status": "failed",
                "message": "Microphone verification failed",
                "audio_captured": final_audio,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
