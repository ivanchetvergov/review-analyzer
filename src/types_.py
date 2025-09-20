from dataclasses import dataclass
from typing import NewType, Optional, FrozenSet

UserId = NewType("UserId", str)
MovieId = NewType("MovieId", str)
ReviewId = NewType("ReviewId", str)

@dataclass(frozen=True)
class Rating:
    value: float
    
    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 10.0):
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
    
@dataclass(frozen=True)
class User:
    user_id: UserId
    age: int
    gender: str
    occupation: str
    zip_code: str
    

    