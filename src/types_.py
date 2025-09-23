# src/types_.py
from dataclasses import dataclass
from typing import NewType, Optional, FrozenSet

UserId = NewType("UserId", str)
MovieId = NewType("MovieId", str)
ReviewId = NewType("ReviewId", str)
TagId = NewType("TagId", str)

@dataclass(frozen=True)
class Rating:
    value: float
    
    def __post_init__(self) -> None:
        if not (0.5 <= self.value <= 5.0):
            raise ValueError(f"invalid rating: {self.value}")

@dataclass(frozen=True)
class Review:
    review_id : ReviewId
    rating : Rating
    movie_id : MovieId
    user_id : UserId

    
@dataclass(frozen=True)
class Movie:
    movie_id : MovieId
    title : str
    genres : FrozenSet[str]
    year : Optional[int]
    imdb_id: Optional[str] = None      
    tmdb_id: Optional[str] = None      
    tags: FrozenSet[str] = frozenset() 
    
@dataclass(frozen=True)
class User:
    user_id: UserId
    
@dataclass(frozen=True)
class Tag:
    tag_id: TagId
    tag: str


    