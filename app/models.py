from pydantic import BaseModel
from typing import List, Optional


class LikedAnnouncement(BaseModel):
    url: str


class LikedAnnouncements(BaseModel):
    content: List[LikedAnnouncement]


class RecommendationRequest(BaseModel):
    interests: list[str]
    liked_announcements: Optional[LikedAnnouncements] = None


class SearchQuery(BaseModel):
    query: str
