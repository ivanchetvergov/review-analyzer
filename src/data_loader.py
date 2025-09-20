# data_loader.py
import csv
import logging
from datetime import datetime
from typing import List, FrozenSet, Optional, TypeVar, Callable

from types_ import Movie, User, Review, Rating, UserId, MovieId, ReviewId

# жанры: последние 19 полей, 1/0
_GENRE_LABELS = [
    "unknown", "action", "adventure", "animation", "children", "comedy",
    "crime", "documentary", "drama", "fantasy", "film-noir", "horror",
    "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"
]

# минимальные колонки в датасетах
_MIN_COLS_MOVIES = 5 + len(_GENRE_LABELS)  # id, title, release, imdb-url + жанры
_MIN_COLS_REVIEWS = 4                      # user_id, movie_id, rating, timestamp
_MIN_COLS_USERS = 5                        # user_id, age, gender, occupation, zip

T = TypeVar("T")

def load_movies(path: str) -> List[Movie]:
    return _load_data(
        path,
        parser_func=_parse_movie_row,
        delimiter="|",
        min_columns=_MIN_COLS_MOVIES,
        log_name="movie"
    )

def load_reviews(path: str) -> List[Review]:
    return _load_data(
        path,
        parser_func=_parse_review_row,
        delimiter="\t",
        min_columns=_MIN_COLS_REVIEWS,
        log_name="review"
    )

def load_users(path: str) -> List[User]:
    return _load_data(
        path,
        parser_func=_parse_user_row,
        delimiter="|",
        min_columns=_MIN_COLS_USERS,
        log_name="user"
    )

def _load_data(
    path: str,
    parser_func: Callable[[list[str], int], T],
    delimiter: str,
    min_columns: int,
    log_name: str
) -> List[T]:
    """
    универсальная функция для загрузки данных из csv-файла
    """
    records: List[T] = []
    with open(path, encoding="latin-1") as file:
        reader = csv.reader(file, delimiter=delimiter)
        for idx, row in enumerate(reader, start=1):
            if not row or len(row) < min_columns:
                logging.debug(f"skipped ({log_name}) row {idx}: {row!r}")
                continue
            try:
                record = parser_func(row, idx)
                records.append(record)
            except Exception as e:
                logging.error(f"error parsing ({log_name}) row {idx}: {row!r} ({e})")
                continue
    return records

def _parse_year(release_date: str) -> Optional[int]:
    if not release_date:
        return None
    s = release_date.strip()
    # try exact format like "01-Jan-1995"
    try:
        return datetime.strptime(s, "%d-%b-%Y").year
    except Exception:
        # fallback: if last 4 chars are digits -> try parse
        if len(s) >= 4 and s[-4:].isdigit():
            return int(s[-4:])
    return None

def _parse_movie_row(row: list[str], _: int) -> Movie:
    mid = MovieId(row[0].strip())
    title = row[1].strip()
    release_date = row[2].strip()
    year = _parse_year(release_date)

    genre_flags = row[-len(_GENRE_LABELS):]
    genres: FrozenSet[str] = frozenset(
        g for g, flag in zip(_GENRE_LABELS, genre_flags) if flag == "1"
    )

    return Movie(movie_id=mid, title=title, genres=genres, year=year)

def _parse_review_row(row: list[str], idx: int) -> Review:
    uid = UserId(row[0])
    mid = MovieId(row[1])
    rid = ReviewId(f"r{uid}_{mid}_{idx}")  # уникальный id для отзыва
    rating = Rating(float(row[2]))
    return Review(review_id=rid, rating=rating, movie_id=mid, user_id=uid)

def _parse_user_row(row: list[str], _: int) -> User:
    uid = UserId(row[0])
    age = int(row[1])
    gender = row[2].strip()
    occupation = row[3].strip()
    zip_code = row[4].strip()
    return User(user_id=uid, age=age, gender=gender, occupation=occupation, zip_code=zip_code)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    movies = load_movies("data/dataset/u.item")
    reviews = load_reviews("data/dataset/u.data")
    users = load_users("data/dataset/u.user")

    print(f"загружено {len(movies)} фильмов, {len(reviews)} отзывов, {len(users)} пользователей.")
    print("пример фильма:", movies[0])
    print("пример отзыва:", reviews[0])
    print("пример пользователя:", users[0])
