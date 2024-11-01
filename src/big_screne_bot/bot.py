import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))


personData = {
    "name": "osama",
    "transcription": "okay let's talk about this another important topic which is the best moment of the life so i'm gonna just tell you what's the best what's the best moment of my\nthat was actually when i was traveling from pakistan to dubai and i was just and i was about to sit in the plane and there were some challenges i faced that because that was the first time i was going inside the plane so i didn't know i don't know what are the protocols actually that we that andy that every passenger has to follow so that's the first thing i was stuck there i was is struggling in that part finally when i sat on my seat and the plane just took off there were some amazing experiences i felt\nplugs okay thank you",
    "questions": {
        "1": "What specific challenges did the speaker face while preparing to board the plane for the first time?",
        "2": "How did the speaker's feelings change from the moment of boarding to the moment the plane took off?",
        "3": "What does the speaker consider to be the significance of this travel experience in the context of their life?"
    }
}

system_prompt = """You are provided with information about a person in the form of a JSON object called `personData`. Your task is to answer any queries based solely on the details within this JSON object. Follow these guidelines:

1. **Access and Identify Data**: Locate relevant keys and values in `personData` that match the query topic.
2. **Answer Scope**: Respond strictly based on the JSON data. If the answer is not in `personData`, respond with: “This information is not available in the provided data.”
3. **Natural, Personalized Responses**: Use full sentences, and include the person’s name or other relevant identifiers where possible to make responses feel more personalized.
4. **Conciseness and Specificity**: Keep answers brief and focused on the query, avoiding unnecessary information.

Here is the JSON object variable format to reference in your responses:

**JSON Object Variable:**
```json
{{personData}}
"""

person_data_str = json.dumps(personData, indent=2)
# # Prepare the input for the model
# input_data = {
#     "transcript_content": personData
# }

# Create the prompt using the template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# chain = llm | prompt

# print(chain.invoke({"transcript_content": person_data_str, "input": "wdo you have any questions for osama?"}))


response = llm.invoke(prompt.invoke({"transcript_content": person_data_str, "input": "wdo you have any questions for osama?"}))

print(response)