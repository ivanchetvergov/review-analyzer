# src/data_loader.py
import csv
import re
import logging
from datetime import datetime
from typing import List, FrozenSet, Optional, TypeVar, Callable

from src.types_ import Movie, User, Review, Rating, UserId, MovieId, ReviewId

_GENRE_LABELS = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    "(no genres listed)"
]

# минимальные колонки в датасетах
_MIN_COLS_MOVIES = 3  # id, title, release, imdb-url + жанры
_MIN_COLS_REVIEWS = 4                      # user_id, movie_id, rating, timestamp

def load_movies(path: str, nrows: Optional[int] = None) -> List[Movie]:
    """
    Загружает данные о фильмах из movies.csv.
    """
    return _load_data(
        path,
        parser_func=_parse_movie_row,
        delimiter=",",
        min_columns=_MIN_COLS_MOVIES,
        log_name="movie",
        skip_header=True,
        nrows=nrows
    )

def load_reviews(path: str, nrows: Optional[int] = None) -> List[Review]:
    """
    Загружает данные об отзывах из ratings.csv.
    """
    return _load_data(
        path,
        parser_func=_parse_review_row,
        delimiter=",",
        min_columns=_MIN_COLS_REVIEWS,
        log_name="review",
        skip_header=True,
        nrows=nrows
    )

T = TypeVar("T")

def _load_data(
    path: str,
    parser_func: Callable[[list[str], int], T],
    delimiter: str,
    min_columns: int,
    log_name: str,
    skip_header: bool = False, 
    nrows: Optional[int] = None
) -> List[T]:
    """
    Универсальная функция для загрузки данных из csv-файла
    """
    records: List[T] = []
    with open(path, encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=delimiter)
        if skip_header:
            next(reader, None)
        for idx, row in enumerate(reader, start=1):
            # Проверяем, достигнуто ли нужное количество строк
            if nrows is not None and len(records) >= nrows:
                logging.info(f"reached limit im {nrows} rows. stop reading.")
                break
            
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

def _parse_movie_row(row: list[str], _: int) -> Movie:
    mid = MovieId(row[0].strip())
    title = row[1].strip()
    
    genres_str = row[2].strip()
    genres: FrozenSet[str] = frozenset(genres_str.split('|'))
    
    year = None
    if "(" in title and ")" in title:
        try:
            year_str = title[title.rfind('(')+1:title.rfind(')')]
            if year_str.isdigit():
                year = int(year_str)
        except:
            pass
            
    return Movie(movie_id=mid, title=title, genres=genres, year=year)

def _parse_movie_row(row: list[str], _: int) -> Movie:
    """
    Парсит строку о фильме из movies.csv.
    """
    mid = row[0].strip()
    title = row[1]

    year_match = re.search(r'\((\d{4})\)', title)
    year = int(year_match.group(1)) if year_match else None
    
    # чистим название фильма от года
    if year_match:
        title = title.replace(year_match.group(0), "").strip()

    genres_str = row[2]
    genres: FrozenSet[str] = frozenset(genres_str.split('|')) if genres_str else frozenset()
    
    return Movie(movie_id=mid, title=title, genres=genres, year=year)

def _parse_review_row(row: list[str], _: int) -> Review:
    """
    Парсит строку отзыва из ratings.csv.
    """
    uid = row[0]
    mid = row[1]
    rating = Rating(float(row[2]))
    rid = f"r{uid}_{mid}"
    return Review(review_id=rid, rating=rating, movie_id=mid, user_id=uid)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    movies = load_movies("data/dataset/movies.csv")
    reviews = load_reviews("data/dataset/ratings.csv")

    print(f"загружено {len(movies)} фильмов, {len(reviews)} отзывов.")
    print("пример фильма:", movies[0])
    print("пример отзыва:", reviews[0])