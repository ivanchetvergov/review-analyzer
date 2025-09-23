# model/lightfm/build_matrices.py
import pandas as pd
import numpy as np
import logging
from scipy.sparse import coo_matrix
from typing import Dict, Tuple

from .features_builder import build_user_features, build_item_features

def build_interactions_matrix(ratings_df : pd.DataFrame) -> Tuple[coo_matrix, Dict[str, int], Dict[str, int]]:
    """
    Создает разреженную матрицу взаимодействий (пользователи-фильмы).
    
    Аргументы:
        ratings_df (pd.DataFrame): DataFrame с данными об отзывах.
        
    Возвращает:
        Tuple[coo_matrix, Dict[str, int], Dict[str, int]]:
            interactions (разреженная матрица), user_to_idx, movie_to_idx.
    """
    unique_users = ratings_df['user_id'].dropna().unique()
    unique_movies = ratings_df['movie_id'].dropna().unique()
    # создаем словарик {id : numeral_id (noy u2137ue1)}
    user_to_idx = {user_id : num_id for num_id, user_id in enumerate(unique_users)}
    movie_to_idx = {movie_id : num_id for num_id, movie_id in enumerate(unique_movies)}
    
    # меняем все user_id на user_to_idx
    rows = ratings_df['user_id'].map(user_to_idx).to_numpy()
    cols = ratings_df['movie_id'].map(movie_to_idx).to_numpy()
    data = ratings_df['rating'].to_numpy(dtype=np.float32)
    
    interactions = coo_matrix(
        (data, (rows, cols)), 
        shape=(len(user_to_idx), len(movie_to_idx))
    )
    
    return interactions, user_to_idx, movie_to_idx

def build_features_matrices(
    reviews_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_to_idx: Dict[str, int],
    movie_to_idx: Dict[str, int]
) -> Tuple[coo_matrix, coo_matrix]:
    """
    Создает разреженные матрицы фичей для пользователей и фильмов для LightFM.
    """
    logging.info("Building user and item features matrices...")

    # --- USER FEATURES ---
    user_features_df = build_user_features(reviews_df, movies_df)
    
    if user_features_df is None:
        user_features_df = pd.DataFrame({'user_id': user_to_idx.keys()})
        user_features_df['user_bias'] = 1.0
    else:
        user_features_df['user_bias'] = 1.0
    
    # реиндексируем, чтобы гарантировать правильный порядок
    user_features_df = user_features_df.drop_duplicates(subset=['user_id'])
    user_features_df = user_features_df.set_index('user_id').reindex(user_to_idx.keys()).fillna(0)
    
    user_features_matrix = coo_matrix(user_features_df.values.astype(np.float32))
    logging.info(f"User features shape: {user_features_matrix.shape}")

    # --- ITEM FEATURES ---
    item_features_df = build_item_features(reviews_df, movies_df)

    if item_features_df is None:
        item_features_df = pd.DataFrame({'movie_id': movie_to_idx.keys()})
        item_features_df['item_bias'] = 1.0
    else:
        # добавляем item_bias
        item_features_df['item_bias'] = 1.0
        
    # реиндексируем, чтобы гарантировать правильный порядок
    item_features_df = item_features_df.set_index('movie_id').reindex(movie_to_idx.keys()).fillna(0)

    item_features_matrix = coo_matrix(item_features_df.values.astype(np.float32))
    logging.info(f"Item features shape: {item_features_matrix.shape}")

    return user_features_matrix, item_features_matrix