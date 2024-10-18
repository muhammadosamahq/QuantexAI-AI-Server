from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from src.utils import filter_response, ask_for_info
from src.model import PersonalDetails

from dotenv import load_dotenv

load_dotenv()
# Initialize FastAPI app
app = FastAPI()

# Initialize the LLM model
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")


# Define a Pydantic model for the input data
class UserInput(BaseModel):
    text_input: str


# Initialize the PersonalDetails object and required fields
user_details = PersonalDetails()
ask_for = ["name", "confirm_name", "gender", "profile_picture", "voice_sample"]


@app.post("/register")
async def register_user(user_input: UserInput):
    global user_details, ask_for

    if ask_for:
        # Check if name and confirm_name are consistent
        if (
            user_details.confirm_name == ""
            or user_details.confirm_name == user_details.name
        ):
            # Get AI response and track the last asked field
            ai_response, last_field = ask_for_info(ask_for, llm)
            user_details.last_asked_field = last_field  # Store last asked field

            # Prepare input for filtering
            overall_input = f"{last_field}: {user_input.text_input.lower()}"
            user_details, ask_for = filter_response(overall_input, user_details, llm)

            # If more info is required, return the next prompt
            if ask_for:
                return {
                    "status": "in_progress",
                    "next_question": ai_response,
                    "details": user_details.dict(),
                }
            else:
                # All information is gathered, proceed to the next phase
                return {
                    "status": "completed",
                    "message": "All details gathered!",
                    "user_details": user_details.dict(),
                }
        else:
            # Reset if name confirmation fails
            user_details = PersonalDetails()
            ask_for = [
                "name",
                "confirm_name",
                "gender",
                "profile_picture",
                "voice_sample",
            ]
            raise HTTPException(
                status_code=400, detail="Name and confirm_name should be equal."
            )
    else:
        # All fields are already gathered
        return {
            "status": "completed",
            "message": "All details already gathered.",
            "user_details": user_details.dict(),
        }


# To run the server, use: uvicorn main:app --reload
