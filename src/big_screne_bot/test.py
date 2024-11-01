import os

# Define the folder containing transcriptions
TRANSCRIPTION_FOLDER = "transcriptions"

def load_all_transcriptions():
    """Loads all transcriptions from the folder and returns a dictionary with names and their transcripts."""
    transcriptions = {}
    
    # Iterate through each file in the transcriptions folder
    for file_name in os.listdir(TRANSCRIPTION_FOLDER):
        # Only process text files
        if file_name.endswith(".txt"):
            # Extract the person's name from the filename (e.g., 'osama' from 'osama.txt')
            person_name = os.path.splitext(file_name)[0]
            
            # Read the transcription content
            file_path = os.path.join(TRANSCRIPTION_FOLDER, file_name)
            with open(file_path, 'r') as file:
                transcript_content = file.read()
            
            # Store the transcript content in the dictionary with the person's name as the key
            transcriptions[person_name] = transcript_content
    
    return transcriptions

# Load all transcriptions and print them
all_transcriptions = load_all_transcriptions()
for name, content in all_transcriptions.items():
    print(f"Name: {name}")
    print(f"Transcription:\n{content}\n")
