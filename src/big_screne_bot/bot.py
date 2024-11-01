# import os
# import json
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage

# from dotenv import load_dotenv

# load_dotenv()

# # # Initialize the language model
# llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))


# # personData = {
# #     "name": "osama",
# #     "transcription": "okay let's talk about this another important topic which is the best moment of the life so i'm gonna just tell you what's the best what's the best moment of my\nthat was actually when i was traveling from pakistan to dubai and i was just and i was about to sit in the plane and there were some challenges i faced that because that was the first time i was going inside the plane so i didn't know i don't know what are the protocols actually that we that andy that every passenger has to follow so that's the first thing i was stuck there i was is struggling in that part finally when i sat on my seat and the plane just took off there were some amazing experiences i felt\nplugs okay thank you",
# #     "questions": {
# #         "1": "What specific challenges did the speaker face while preparing to board the plane for the first time?",
# #         "2": "How did the speaker's feelings change from the moment of boarding to the moment the plane took off?",
# #         "3": "What does the speaker consider to be the significance of this travel experience in the context of their life?"
# #     }
# # }

# # system_prompt = """You are provided with information about a person in the form of a JSON object called `personData`. Your task is to answer any queries based solely on the details within this JSON object. Follow these guidelines:

# # 1. **Access and Identify Data**: Locate relevant keys and values in `personData` that match the query topic.
# # 2. **Answer Scope**: Respond strictly based on the JSON data. If the answer is not in `personData`, respond with: “This information is not available in the provided data.”
# # 3. **Natural, Personalized Responses**: Use full sentences, and include the person’s name or other relevant identifiers where possible to make responses feel more personalized.
# # 4. **Conciseness and Specificity**: Keep answers brief and focused on the query, avoiding unnecessary information.

# # Here is the JSON object variable format to reference in your responses:

# # **JSON Object Variable:**
# # ```json
# # {{personData}}
# # """

# # person_data_str = json.dumps(personData, indent=2)
# # # Prepare the input for the model
# # input_data = {
# #     "transcript_content": personData
# # }

# # Create the prompt using the template
# # prompt = ChatPromptTemplate.from_messages(
# #     [
# #         ("system", system_prompt),
# #         ("human", "{input}")
# #     ]
# # )

# # chain = llm | prompt

# # # print(chain.invoke({"transcript_content": person_data_str, "input": "wdo you have any questions for osama?"}))


# # response = llm.invoke(prompt.invoke({"transcript_content": person_data_str, "input": "wdo you have any questions for osama?"}))

# # print(response)



# from typing import List, Optional

# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from pydantic import BaseModel, Field, validator


# def load_person_data(name):
#     # Ensure the name is lowercase to match the file naming convention
#     name = name.lower()
    
#     # Construct the path to the JSON file
#     file_path = os.path.join("data", f"{name}.json")
    
#     # Check if the file exists and load the JSON content
#     if os.path.exists(file_path):
#         with open(file_path, "r") as file:
#             person_data = json.load(file)
#         return person_data
#     else:
#         print(f"No data found for {name}. Make sure the file '{name}.json' exists in the 'data' folder.")
#         return None
    

# class Person(BaseModel):
#     """Information about a person."""

#     name: str = Field(..., description="The name of the person")
#     ask_for_questions: bool = Field(
#         ..., description="ask for questions to bot"
#     )


# # class People(BaseModel):
# #     """Identifying information about all people in a text."""

# #     people: List[Person]


# # Set up a parser
# parser = PydanticOutputParser(pydantic_object=Person)

# # Prompt
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
#         ),
#         ("human", "{query}"),
#     ]
# ).partial(format_instructions=parser.get_format_instructions())


# prompt_template = PromptTemplate.from_template(
#     """
#     You are an assistant. A user has asked the following question: "{user_query}"

#     You have the following questions to ask {name}:
#     {questions}

#     Please provide a natural response that includes the questions you would like to ask {name} based on the user's query.
#     """
# )



# query = "do you have any questions for osama"
# chain = prompt | llm | parser
# ask_for = chain.invoke({"query": query})

# data = load_person_data(ask_for.name.lower())
# print(data)

# if data and ask_for.ask_for_questions:
#     print(data["questions"])

#     formatted_prompt = prompt_template.format(
#     user_query=query,
#     name=data["name"],
#     questions=data["questions"]
#     )
    
#     print(formatted_prompt)

#     response = llm.invoke([HumanMessage(content=formatted_prompt)])

#     # Print the response
#     print(response.content)


import os
import json
from typing import Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
import uvicorn

load_dotenv()

app = FastAPI()

# Initialize the language model
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define Pydantic Models for the input and output
class QueryRequest(BaseModel):
    station_id: int
    query: str

class Person(BaseModel):
    """Information about a person."""
    name: str = Field(..., description="The name of the person")
    ask_for_questions: bool = Field(..., description="Ask if there are questions for the bot")

# Load JSON data from the data folder based on a name
def load_person_data(name: str) -> Optional[dict]:
    name = name.lower()
    file_path = os.path.join("data", f"{name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            person_data = json.load(file)
        return person_data
    else:
        print(f"No data found for {name}. Make sure the file '{name}.json' exists in the 'data' folder.")
        return None

# Set up output parser
parser = PydanticOutputParser(pydantic_object=Person)

# Define prompt template
system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user query. Wrap the output in `json` tags\n{format_instructions}"),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Define follow-up question prompt template
question_prompt_template = PromptTemplate.from_template(
    """
    You are an assistant. A user has asked the following question: "{user_query}"

    You have the following questions to ask {name}:
    {questions}

    Please provide a natural response that includes the questions you would like to ask {name} based on the user's query.
    """
)

@app.get("/query/")
async def get_response(station_id: int = Query(...), query: str = Query(...)):
    # Process the query with initial prompt
    chain = system_prompt | llm | parser
    ask_for = chain.invoke({"query": query})
    
    # Load person data based on the name from the parsed output
    # data = load_person_data(ask_for.name.lower())
    data = load_person_data("usama")
    
    if data and ask_for.ask_for_questions:
        # Format questions if available
        formatted_prompt = question_prompt_template.format(
            user_query=query,
            name=data["name"],
            questions=data["questions"]
        )
        
        # Get the final response from the language model
        response = llm.invoke([HumanMessage(content=formatted_prompt)])

        return {
            "station_id": station_id,
            "response": response.content
        }
    else:
        return {"station_id": station_id, "response": "No questions available or data not found for the person."}

# Entry point
if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
