# model/lightfm/build_matrices.py
import pandas as pd
import numpy as np
import logging
from scipy.sparse import coo_matrix
from typing import Dict, Tuple

def build_interactions_matrix(ratings_df : pd.DataFrame) -> Tuple[coo_matrix, Dict[str, int], Dict[str, int]]:
    """
    Создает разреженную матрицу взаимодействий (пользователи-фильмы).
    """
    unique_users = ratings_df['user_id'].dropna().unique()
    unique_movies = ratings_df['movie_id'].dropna().unique()
    # создаем словарик {id : numeral_id (noy u2137ue1)}
    user_to_idx = {user_id : num_id for num_id, user_id in enumerate(unique_users)}
    movie_to_idx = {movie_id : num_id for num_id, movie_id in enumerate(unique_movies)}
    
    # меняем все user_id на user_to_idx
    rows = ratings_df['user_id'].map(user_to_idx).to_numpy()
    cols = ratings_df['movie_id'].map(movie_to_idx).to_numpy()
    data = ratings_df['rating_value'].to_numpy(dtype=np.float32)
    
    interactions = coo_matrix(
        (data, (rows, cols)), 
        shape=(len(user_to_idx), len(movie_to_idx))
    )
    
    return interactions, user_to_idx, movie_to_idx

def build_features_matrices(
    full_dataset_df: pd.DataFrame,
    user_to_idx: Dict[str, int],
    movie_to_idx: Dict[str, int]
) -> Tuple[coo_matrix, coo_matrix]:
    """
    Создает разреженные матрицы фичей для пользователей и фильмов для LightFM.
    """

    logging.info("Building user features matrix...")

    # --- USER FEATURES ---
    user_features_df = full_dataset_df.drop_duplicates(subset=['user_id']).set_index('user_id')
    user_features_df = user_features_df.reindex(user_to_idx.keys())  # упорядочиваем по словарю
    user_features_df = user_features_df.fillna({
        'age': user_features_df['age'].median(),
        'gender': 'unknown',
        'occupation': 'unknown',
        'zip_code': '00000'
    })

    # нормализация возраста
    user_features_df['age'] = (user_features_df['age'] - user_features_df['age'].mean()) / user_features_df['age'].std()

    # one-hot кодирование категориальных признаков
    user_features_df = pd.get_dummies(user_features_df, columns=['gender', 'occupation'], drop_first=False)

    # добавляем user_bias
    user_features_df['user_bias'] = 1.0

    # удаляем лишние колонки
    drop_cols = ['review_id', 'rating_value', 'movie_id', 'title', 'genres', 'year', 'zip_code']
    user_features_df = user_features_df.drop(columns=[c for c in drop_cols if c in user_features_df.columns])

    user_features_matrix = coo_matrix(user_features_df.values.astype(np.float32))

    logging.info(f"user features matrix shape: {user_features_matrix.shape}")

    # --- ITEM FEATURES ---
    logging.info("Building item features matrix...")

    item_features_df = full_dataset_df.drop_duplicates(subset=['movie_id']).set_index('movie_id')
    item_features_df = item_features_df.reindex(movie_to_idx.keys())
    item_features_df = item_features_df.fillna({
        'year': item_features_df['year'].median(),
        'genres': ''
    })

    # one-hot кодирование жанров
    genres_df = item_features_df['genres'].str.get_dummies(sep=',')
    item_features_df = pd.concat([item_features_df.drop(columns=['genres']), genres_df], axis=1)

    # добавляем item_bias
    item_features_df['item_bias'] = 1.0

    # удаляем лишние колонки, которые не числовые
    drop_cols_item = ['age', 'gender', 'occupation', 'title', 'rating_value', 'review_id', 'user_id', 'zip_code']
    item_features_df = item_features_df.drop(columns=[c for c in drop_cols_item if c in item_features_df.columns])

    item_features_matrix = coo_matrix(item_features_df.values.astype(np.float32))

    return user_features_matrix, item_features_matrix