import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Read the transcript content from the file
file_path = "transcription/test.txt"
with open(file_path, 'r') as file:
    transcript_content = file.read()

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
except json.JSONDecodeError:
    print("Failed to decode JSON response:", response.content)
