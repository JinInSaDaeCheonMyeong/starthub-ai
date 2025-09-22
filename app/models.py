from pydantic import BaseModel


class UserInterests(BaseModel):
    interests: list[str]


class SearchQuery(BaseModel):
    query: str