# Media Test Bot

This project is a FastAPI-based application that checks the status of audio and video devices (microphone and webcam) on your system.

## Features

- **Check Webcam Status**: Verifies if the webcam is operational.
- **Check Microphone Status**: Verifies if the microphone is operational.
- **Response Status**: Provides a detailed response indicating the status of both devices.

## Device Management

### Webcam

- **Disable the Webcam**:

  ```bash
  sudo modprobe -r uvcvideo  # This unloads the webcam driver

  ```

- **Enable the Webcam**:

  ```bash
  sudo modprobe uvcvideo  # This loads the webcam driver

  ```

### Microphone

- **Mute the Microphone**:

```bash
   amixer set Capture nocap  # This mutes the microphone


```

- **Unmute the Microphone**:

  ```bash
  amixer set Capture cap  # This unmutes the microphone


  ```
