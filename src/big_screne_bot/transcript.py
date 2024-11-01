# import requests

# # Define the URL for the Deepgram API endpoint
# url = "https://api.deepgram.com/v1/listen"

# # Define the headers for the HTTP request
# headers = {
#     "Authorization": "Token 19c0c660d45307bb748731c7e25f81c47bbfe6ed",
#     "Content-Type": "audio/*"
# }

# # Get the audio file
# with open("./recordings/recording_20241028-201713.wav", "rb") as audio_file:
#     # Make the HTTP request
#     response = requests.post(url, headers=headers, data=audio_file)

# data = response.json()
# print(data)

# transcript = data['results']['channels'][0]['alternatives'][0]['transcript']
# print(transcript)

# file_path = f"transcription/test.txt"
# with open(file_path, 'w') as file:
#     file.write(transcript)

# print(f"Transcript saved to {file_path}")

import os
import requests

# Define the URL for the Deepgram API endpoint
url = "https://api.deepgram.com/v1/listen"

# Define the headers for the HTTP request
headers = {
    "Authorization": "Token 19c0c660d45307bb748731c7e25f81c47bbfe6ed",
    "Content-Type": "audio/*"
}

# Folder paths
BASE_RECORDING_FOLDER = "recordings"
TRANSCRIPTION_FOLDER = "transcriptions"
os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)  # Create the transcription folder if it doesn't exist

def transcribe_audio_file(audio_file_path):
    """Transcribes a single audio file using Deepgram API and returns the transcript text."""
    with open(audio_file_path, "rb") as audio_file:
        response = requests.post(url, headers=headers, data=audio_file)
    
    if response.status_code == 200:
        data = response.json()
        transcript = data['results']['channels'][0]['alternatives'][0]['transcript']
        return transcript
    else:
        print(f"Failed to transcribe {audio_file_path}")
        return ""

def process_recordings():
    """Processes all recordings for each person and creates a combined transcription file."""
    for person_name in os.listdir(BASE_RECORDING_FOLDER):
        person_folder = os.path.join(BASE_RECORDING_FOLDER, person_name)
        
        if os.path.isdir(person_folder):
            combined_transcript = []
            
            # Sort audio files by name
            audio_files = sorted(os.listdir(person_folder))
            
            # Iterate through all audio files in the person's folder in sorted order
            for audio_file_name in audio_files:
                audio_file_path = os.path.join(person_folder, audio_file_name)
                
                if audio_file_path.endswith(".wav"):
                    print(f"Transcribing {audio_file_path} for {person_name}...")
                    transcript = transcribe_audio_file(audio_file_path)
                    
                    # Append each transcript to the combined transcript list
                    if transcript:
                        combined_transcript.append(transcript)

            # Save the combined transcript for the person
            transcript_text = "\n".join(combined_transcript)
            transcript_file_path = os.path.join(TRANSCRIPTION_FOLDER, f"{person_name}.txt")
            with open(transcript_file_path, 'w') as transcript_file:
                transcript_file.write(transcript_text)
            
            print(f"Combined transcript saved to {transcript_file_path}")

# Run the process
process_recordings()
