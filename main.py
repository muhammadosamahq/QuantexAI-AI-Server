from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import uuid
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import get_db, User
from utils import (
    capture_photo,
    detect_cross_question,
    handle_cross_question,
    record_voice,
)
import asyncio
from model import PersonalDetails
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the model (e.g., OpenAI, GPT-4)
llm = ChatOpenAI(
    temperature=0, model="gpt-4o-mini"
)  # Ensure you have the model initialized

chain = llm.with_structured_output(PersonalDetails)
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegistrationState:
    def __init__(self):
        self.username = None
        self.confirmed_name = None
        self.gender = None
        self.profile_pic = None
        self.voice_sample = None
        self.current_step = "name"


# Store user states with UUID as key
registration_states = {}


class ChatMessage(BaseModel):
    message: str
    image_data: Optional[str] = None
    audio_data: Optional[str] = None


@app.post("/start_registration")
async def start_registration():
    # Generate a unique user ID
    user_id = str(uuid.uuid4())
    registration_states[user_id] = RegistrationState()
    return {"user_id": user_id, "message": "Welcome! Please tell me your name."}


@app.post("/chat/{user_id}")
async def chat(user_id: str, chat_message: ChatMessage, db: Session = Depends(get_db)):
    if user_id not in registration_states:
        raise HTTPException(status_code=404, detail="Session not found")

    state = registration_states[user_id]
    print(f"state:{state}")
    text_input = chat_message.message
    is_cross_question = detect_cross_question(text_input)
    if is_cross_question:
        current_step = state.current_step
        print(current_step)
        state.username = None
        return handle_cross_question(current_step)
    message = chain.invoke(text_input)
    # print(message.name)
    if state.current_step == "name":
        if not state.username:
            state.username = message.name
            print(state.username)
            return {
                "response": f"Thank you! To confirm, is your name {message.name}? (yes/no)"
            }
        else:
            if text_input.lower() == "yes":
                state.confirmed_name = state.username
                print(state.confirmed_name)
                print(state.username)
                state.current_step = "gender"
                return {"response": "Great! What is your gender? (male/female)"}
            else:
                state.username = None
                return {"response": "Let's start over. What is your name?"}

    elif state.current_step == "gender":
        is_cross_question = detect_cross_question(text_input)
        if is_cross_question:
            current_step = state.current_step
            state.gender = None
            return handle_cross_question(current_step)
        if text_input.lower() in ["male", "female"]:
            state.gender = text_input.lower()
            state.current_step = "photo"
            return {
                "response": "Perfect! Now, please stand straight in front of the camera so I can take your picture. Just say 'ready' whenever you're ready to capture it."
            }
        else:
            return {"response": "Please specify either 'male' or 'female'."}

    elif state.current_step == "photo":
        is_cross_question = detect_cross_question(text_input)
        if is_cross_question:
            current_step = state.current_step
            chat_message.image_data = None
            return handle_cross_question(current_step)
        if text_input.lower() == "ready":
            await asyncio.sleep(5)
            chat_message.image_data = capture_photo()
            state.profile_pic = (
                chat_message.image_data
            )  # Set the profile picture from the incoming image data
            state.current_step = "voice"
            return {
                "response": "Photo saved! Now, please say the following phrase: 'I am the luckiest guy in the world,' or anything else you'd like. Just say 'ready' when you're good to start recording your voice sample."
            }
        return {
            "response": "A profile picture is necessary for identification at the event. Please provide a profile picture and say 'ready' when yu are ready."
        }

    elif state.current_step == "voice":
        is_cross_question = detect_cross_question(text_input)
        if is_cross_question:
            current_step = state.current_step
            chat_message.image_data = None
            return handle_cross_question(current_step)
        if text_input.lower() == "ready":
            await asyncio.sleep(5)
            chat_message.audio_data = await record_voice()
            state.voice_sample = chat_message.audio_data
            # Save user data to database
            new_user = User(
                id=user_id,
                username=state.username,
                gender=state.gender,
                profile_pic=state.profile_pic,
                voice_sample=state.voice_sample,
            )
            db.add(new_user)
            db.commit()

            # Clean up registration state
            registration_data = {
                "username": state.username,
                "gender": state.gender,
                "profile_pic": state.profile_pic,
                "voice_sample": state.voice_sample,
            }
            del registration_states[user_id]
            return {
                "response": "Registration complete! Thank you for registering.",
                "registration_data": registration_data,
            }
        return {"response": "Please record your voice using the microphone button."}

    return {"response": "Invalid state"}


# Add endpoint to retrieve all users
@app.get("/users")
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users
