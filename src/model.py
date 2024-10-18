from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate

from enum import Enum
from pydantic import BaseModel, Field


class PersonalDetails(BaseModel):
    name: str = Field(
        default="",
        description="This field asks for the person's name.",
    )
    confirm_name: str = Field(
        default="",
        description="This field asks for the person's name again.",
    )
    gender: str = Field(
        default="",
        description="This field asks about gender (Male or Female).",
    )
    profile_picture: str = Field(
        default="",
        description="This field is for capturering a profile picture in base64 format.",
    )
    voice_sample: str = Field(
        default="",
        description="This field is for capturing a voice sample in base64 format.",
    )
    last_asked_field: str = Field(
        default="",
        description="Tracks the last field that was asked to the user.",
    )
