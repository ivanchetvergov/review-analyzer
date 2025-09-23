# model/lightfm/prepare_data.py
import logging
import pandas as pd
import numpy as np
from typing import Tuple
from src.data_loader import load_movies, load_reviews
from src.data_processor import create_dataframes, merge_datasets
from sklearn.preprocessing import MinMaxScaler



def get_dataframes(
    movie_rows : int,
    review_rows: int,
    path_movies="data/ml-20m/movies.csv",
    path_reviews="data/ml-20m/ratings.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame] :
    
    logging.info("loading movies and reviews...")
    movies_list = load_movies(path_movies, nrows=movie_rows) 
    reviews_list = load_reviews(path_reviews, nrows=review_rows) 

    logging.info("creating dataframes...")
    movies_df, reviews_df, _ = create_dataframes(
        movies_list, reviews_list, users_list=None
    )
    
    logging.info("merging dataframes...")
    full_dataset_df = merge_datasets(
        movies_df, reviews_df, users_df=None
    )
    
    logging.info(f"dataframes ready: {len(reviews_df)} reviews, {len(full_dataset_df)} full records")
    
    return reviews_df, full_dataset_df

def preprocess_features(user_features_df: pd.DataFrame, item_features_df: pd.DataFrame):
    """
    Нормализация числовых фичей и обработка бинарных для пользователей и фильмов
    """

    # числовые фичи у пользователей
    numeric_user_cols = ["num_ratings", "mean_rating", "rating_var"]
    # числовые фичи у фильмов
    numeric_item_cols = ["num_ratings", "avg_rating"]

    scaler = MinMaxScaler()

    # нормализация числовых user features
    for col in numeric_user_cols:
        if col in user_features_df.columns:
            vals = user_features_df[[col]].fillna(0)
            user_features_df[col] = scaler.fit_transform(vals)

    # нормализация числовых item features
    for col in numeric_item_cols:
        if col in item_features_df.columns:
            vals = item_features_df[[col]].fillna(0)
            item_features_df[col] = scaler.fit_transform(vals)

    # обработка бинарных: заменим NaN на 0
    user_features_df = user_features_df.fillna(0)
    item_features_df = item_features_df.fillna(0)

    # если встречается nan-фича (например top_genre_nan), можем оставить её как отдельный индикатор
    user_features_df = user_features_df.rename(
        columns={c: c.replace("nan", "unknown") for c in user_features_df.columns}
    )
    item_features_df = item_features_df.rename(
        columns={c: c.replace("nan", "unknown") for c in item_features_df.columns}
    )

    return user_features_df, item_features_df
