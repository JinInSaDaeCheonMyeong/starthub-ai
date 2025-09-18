from pydantic import BaseModel


class UserInterests(BaseModel):
    interests: list[str]
