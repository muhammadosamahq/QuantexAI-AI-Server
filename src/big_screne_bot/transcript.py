import requests

# Define the URL for the Deepgram API endpoint
url = "https://api.deepgram.com/v1/listen"

# Define the headers for the HTTP request
headers = {
    "Authorization": "Token 19c0c660d45307bb748731c7e25f81c47bbfe6ed",
    "Content-Type": "audio/*"
}

# Get the audio file
with open("./recordings/recording_20241028-201713.wav", "rb") as audio_file:
    # Make the HTTP request
    response = requests.post(url, headers=headers, data=audio_file)

data = response.json()
print(data)

transcript = data['results']['channels'][0]['alternatives'][0]['transcript']
print(transcript)

file_path = f"transcription/test.txt"
with open(file_path, 'w') as file:
    file.write(transcript)

print(f"Transcript saved to {file_path}")