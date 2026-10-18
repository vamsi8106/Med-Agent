"""Request and response bodies for the HTTP API, with input bounds."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from medagent.core.config import MIN_PASSWORD_LENGTH
from medagent.core.models import PATIENT_ID_PATTERN

MAX_MESSAGE_CHARS = 4000

Message = Annotated[str, StringConstraints(min_length=1, max_length=MAX_MESSAGE_CHARS)]


class AssessRequest(BaseModel):
    message: Message


class DrugCheckRequest(BaseModel):
    patient_id: str = Field(pattern=PATIENT_ID_PATTERN)
    new_drug: Annotated[str, StringConstraints(min_length=1, max_length=200)]


class CreateUserRequest(BaseModel):
    username: str = Field(pattern=r"^[A-Za-z0-9._-]{3,64}$")
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=256)
    role: Literal["doctor", "admin"] = "doctor"


Password = Annotated[str, StringConstraints(min_length=MIN_PASSWORD_LENGTH, max_length=256)]


class ResetPasswordRequest(BaseModel):
    new_password: Password


class ChangePasswordRequest(BaseModel):
    current_password: Annotated[str, StringConstraints(max_length=256)]
    new_password: Password


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
