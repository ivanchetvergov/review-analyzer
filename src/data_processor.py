# data_processor.py
import pandas as pd
from typing import List
from types_ import Movie, Review, User
import logging

def create_dataframes(
    movies_list: List[Movie],
    reviews_list: List[Review],
    users_list: List[User]
):
    """
    Создает и объединяет Pandas DataFrame из списков объектов.
    """
    logging.info("Converting to DataFrame...")
    movies_df = pd.DataFrame(movies_list)
    reviews_df = pd.DataFrame(reviews_list)
    users_df = pd.DataFrame(users_list)

    # присваиваем колонкам 'movie_id' и 'user_id' правильные типы для оптимизации
    reviews_df['movie_id'] = reviews_df['movie_id'].astype('string')
    reviews_df['user_id'] = reviews_df['user_id'].astype('string')
    movies_df['movie_id'] = movies_df['movie_id'].astype('string')
    users_df['user_id'] = users_df['user_id'].astype('string')

    # извлекаем числовое значение рейтинга
    reviews_df['rating_value'] = reviews_df['rating'].apply(lambda x: x['value'])

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