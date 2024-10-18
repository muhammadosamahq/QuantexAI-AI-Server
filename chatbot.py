from langchain_openai import ChatOpenAI
from src.utils import (
    filter_response,
    ask_for_info,
)
from src.model import PersonalDetails

from dotenv import load_dotenv

load_dotenv()

# Ensure you import the LLM you're using

# Initialize the model (e.g., OpenAI, GPT-4)
llm = ChatOpenAI(
    temperature=0, model="gpt-4o-mini"
)  # Ensure you have the model initialized

user_details = PersonalDetails()
# print(user_details)

ask_for = [
    "name",
    "confirm_name",
    "gender",
    "profile_picture",
    "voice_sample",
]

# while True:
#     if ask_for:
#         if (
#             user_details.confirm_name == ""
#             or user_details.confirm_name == user_details.name
#         ):
#             ai_response = ask_for_info(ask_for, llm)
#             print(ai_response)
#             text_input = input()
#             overall_input = f"{ai_response} + \n{text_input.lower()}"
#             # print(overall_input)
#             user_details, ask_for = filter_response(overall_input, user_details, llm)
#             # print(ask_for)
#             # print(user_details)
#         else:
#             print("name and confirm_name should be equal")
#             user_details = PersonalDetails()
#             ask_for = [
#                 "name",
#                 "confirm_name",
#                 "gender",
#                 "profile_picture",
#                 "voice_sample",
#             ]
#     else:
#         print("Everything gathered move to next phase")
#         break

while True:
    if ask_for:
        if (
            user_details.confirm_name == ""
            or user_details.confirm_name == user_details.name
        ):
            ai_response, last_field = ask_for_info(ask_for, llm)
            print(ai_response)
            # print(">>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<")
            # print(last_field)

            # Store the last asked field for cross-question handling
            user_details.last_asked_field = last_field
            print("-----------------------------------------------------")
            print(user_details)

            text_input = input()

            # overall_input = f"{ai_response} + \n{text_input.lower()}"
            overall_input = f"{last_field} : {text_input.lower()}"
            # print("|||||||||||||||||||||||||||||||||")
            # print(overall_input)

            user_details, ask_for = filter_response(overall_input, user_details, llm)
            # print(f"ask_for=>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{ask_for}")
        else:
            print("Name and confirm_name should be equal.")
            user_details = PersonalDetails()
            ask_for = [
                "name",
                "confirm_name",
                "gender",
                "profile_picture",
                "voice_sample",
            ]
    else:
        print("Everything gathered. Moving to the next phase.")
        break
# print(user_details)
# print(ask_for)
