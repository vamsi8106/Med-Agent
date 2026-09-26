"""Request and response bodies for the HTTP API."""

from pydantic import BaseModel


class AssessRequest(BaseModel):
    message: str


class DrugCheckRequest(BaseModel):
    patient_id: str
    new_drug: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "doctor"


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
