from pydantic import BaseModel, Field


class PersonalDetails(BaseModel):
    name: str = Field(
        default="",
        description="This field asks for the person's name.",
    )
