# from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
from enum import Enum
from pydantic import BaseModel, Field

from model import PersonalDetails


# model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# structured_llm = model.with_structured_output(PersonalDetails)

# structured_llm.invote(
#     "My name is Osama and i am 27 year old , i am male , i live in cairo , i am developer"
# )
from typing import Optional

from pydantic import BaseModel, Field


# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


structured_llm = llm.with_structured_output(Joke)

structured_llm.invoke("Tell me a joke about cats")
print(type(structured_llm))
