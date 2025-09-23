# data_processor.py
import pandas as pd
from typing import List
from src.types_ import Movie, Review, User
import logging

def create_dataframes(
    movies_list: List[Movie],
    reviews_list: List[Review],
    users_list: List[User]
):
    logging.info("converting to DataFrame...")

    movies_df = pd.DataFrame([
        {
            "movie_id": m.movie_id,
            "title": m.title,
            "year": m.year,
            "genres": ",".join(sorted(m.genres))
        }
        for m in movies_list
    ])

    reviews_df = pd.DataFrame([
        {
            "review_id": r.review_id,
            "movie_id": r.movie_id,
            "user_id": r.user_id,
            "rating_value": r.rating.value
        }
        for r in reviews_list
    ])

    users_df = pd.DataFrame([
        {
            "user_id": u.user_id,
            "age": getattr(u, "age", None),
            "gender": getattr(u, "gender", None),
            "occupation": getattr(u, "occupation", None),
            "zip_code": getattr(u, "zip_code", None)
        }
        for u in users_list
    ])

    return movies_df, reviews_df, users_df


def merge_datasets(
    movies_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    users_df: pd.DataFrame
):
    """
    Объединяет DataFrame'ы в один полный датасет.
    """
    logging.info("Merging DataFrames...")
    reviews_with_movies = pd.merge(reviews_df, movies_df, on="movie_id")
    full_dataset_df = pd.merge(reviews_with_movies, users_df, on="user_id")

    return full_dataset_df