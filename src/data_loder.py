# data_loader.py
import csv
from datetime import datetime
from typing import List
from types_ import Movie, User, Review, Rating, UserId, MovieId, ReviewId
from typing import FrozenSet, Optional

# жанры: последние 19 полей, 1/0
GENRE_LABELS = [
    "unknown", "action", "adventure", "animation", "children", "comedy",
    "crime", "documentary", "drama", "fantasy", "film-noir", "horror",
    "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"
]

_EXPECTED_MIN_COLS = 5 + len(GENRE_LABELS)

def _parse_year(release_date : str) -> Optional[int]:
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


def load_movies(path : str) -> List[Movie]:
    movies : List[Movie] = []
    with open(path, encoding="latin-1") as file:
        reader = csv.reader(file, delimiter="|")
        for idx, row in enumerate(reader, start= 1):
            if not row or len(row) < 5:
                print(f"skipped row {idx}: {row!r}")
                continue
            try:
                mid = MovieId(row[0].strip())
                title = row[1].strip()
                release_date = row[2].strip()
                year = _parse_year(release_date)
                genre_flags = row[6:6 + len(GENRE_LABELS)]
                genres: FrozenSet[str] = frozenset(
                    g for g, flag in zip(GENRE_LABELS, genre_flags) if flag == "1"
                )
                movies.append(Movie(movie_id=mid, title=title, genres=genres, year=year))
            except Exception as e:
                print(f"error parsing row {idx}: {row!r} ({e})")
                continue
    return movies

            
movies = load_movies("data/dataset/u.item")

print(movies[:2])
