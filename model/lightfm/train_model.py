# model/lightfm/train_model.py
import logging
from sklearn.model_selection import train_test_split
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score # type: ignore
from analyze.analyze_model_weights import analyze_models_weights
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np

# Импортируем наши функции для построения матриц
from .build_matrices import build_interactions_matrix, build_features_matrices
from src.data_loader import load_movies, load_reviews
from src.data_processor import create_dataframes

def run_training():
    logging.basicConfig(level=logging.INFO)
    
    logging.info("Loading and processing data...")

    movies_list = load_movies("data/ml-20m/movies.csv", nrows= 1_000_000)
    reviews_list = load_reviews("data/ml-20m/ratings.csv", nrows=3_000_000)
    
    movies_df, reviews_df, _ = create_dataframes(movies_list, reviews_list, None)
    
    train_df, test_df = train_test_split(reviews_df, test_size=0.2, random_state=42)
    
    # cтроим матрицу взаимодействий и маппинги
    interactions, user_to_idx, movie_to_idx = build_interactions_matrix(train_df)
    # cтроим матрицы фичей, передавая reviews_df и movies_df
    user_features, item_features, user_feature_names, item_feature_names = build_features_matrices(
        reviews_df, movies_df, user_to_idx, movie_to_idx
    )
    
    logging.info("Fitting LightFM model...")
    # инициализируем модель с учетом фичей
    model = LightFM(
        loss='warp',
        no_components=50,
        learning_schedule='adagrad',
        user_alpha=0.6,    # L2 регуляризация для пользователей
        item_alpha=0.6,    # L2 регуляризация для предметов
    )
    
    # обучаем модель
    model.fit(
        interactions=interactions,
        user_features=user_features,
        item_features=item_features,
        epochs=35,
        num_threads=4
    )
     
    # оцениваем модель
    test_interactions, _, _ = build_interactions_matrix(test_df)
    
    # вычисляем precision
    k = 5
    train_precision = precision_at_k(
        model,
        interactions,
        user_features=user_features,
        item_features=item_features,
        k=k
    ).mean()
    
    test_precision = precision_at_k(
        model,
        test_interactions,
        user_features=user_features,
        item_features=item_features,
        k=k
    ).mean()
    
    logging.info(f"Train precision@{k}: {train_precision:.4f}")
    logging.info(f"Test precision@{k}: {test_precision:.4f}")

    # Вычисляем AUC
    train_auc = auc_score(
        model,
        interactions,
        user_features=user_features,
        item_features=item_features
    ).mean()
    
    test_auc = auc_score(
        model,
        test_interactions,
        user_features=user_features,
        item_features=item_features
    ).mean()
    
    logging.info(f"Train AUC: {train_auc:.4f}")
    logging.info(f"Test AUC: {test_auc:.4f}")
    
    analyze_models_weights(model, user_features, item_features, user_feature_names, item_feature_names, top_n=15)

if __name__ == "__main__":
    run_training()