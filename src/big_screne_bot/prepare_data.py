import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Directory where you want to save the JSON files
DATA_DIRECTORY = "data"
os.makedirs(DATA_DIRECTORY, exist_ok=True)  # Ensure the directory exists

# Read the transcript content from the file
def load_all_transcriptions():
    TRANSCRIPTION_FOLDER = "transcriptions"
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

all_transcriptions = load_all_transcriptions()
for name, transcript_content in all_transcriptions.items():
    json_object = {}
    json_object["name"] = name
    json_object["transcription"] = transcript_content
    print(f"Name: {name}")
    print(f"Transcription:\n{transcript_content}\n")


    # Define the system prompt to generate questions in JSON format
    system_prompt = """Role: Following is a speech of a person, generate 3 intellectual questions from this speech in JSON format.
                        The output should be structured as follows:
                        {{
                            "key1": "Question 1",
                            "key2": "Question 2",
                            "key3": "Question 3"
                        }}
                        Speech: {transcript_content}"""

    # Prepare the input for the model
    input_data = {
        "transcript_content": transcript_content
    }

    # Create the prompt using the template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Please generate questions based on the following speech.")
        ]
    )

    # Invoke the model with the prepared prompt
    response = llm.invoke(prompt.invoke(input_data))

    # Parse the response to JSON
    try:
        questions_json = json.loads(response.content)
        print(questions_json)
        questions = {}
        for counter, (key, value) in enumerate(questions_json.items()):
            questions[counter+1] = value
        json_object["questions"] = questions
        print("json_object", json_object)

        # Define the file path using the name field
        file_name = f"{json_object['name']}.json"
        file_path = os.path.join(DATA_DIRECTORY, file_name)

        # Save json_object to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(json_object, json_file, indent=4)

        print(f"JSON object saved to {file_path}")
        

    except json.JSONDecodeError:
        print("Failed to decode JSON response:", response.content)

