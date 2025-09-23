import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix 
from model.lightfm.build_matrices import build_features_matrices, build_interactions_matrix
from src.data_loader import load_movies, load_reviews
from src.data_processor import create_dataframes

def analyze_features_weights():
    movies_list = load_movies("data/ml-20m/movies.csv", nrows=1_000_000)
    reviews_list = load_reviews("data/ml-20m/ratings.csv", nrows=1_000_000)
    movies_df, reviews_df, _ = create_dataframes(movies_list, reviews_list, None)
    
    # получаем маппинги из train
    _, user_to_idx, movie_to_idx = build_interactions_matrix(reviews_df)
    
    user_features, item_features, user_feature_names, item_feature_names  = build_features_matrices(
        reviews_df, movies_df, user_to_idx, movie_to_idx
    )
    print("User feature names:", user_feature_names)
    print("Item feature names:", item_feature_names)
    
    print("User features shape:", user_features.shape)
    print("Item features shape:", item_features.shape)
    
    # анализ плотности
    print("User features density:", user_features.nnz / (user_features.shape[0] * user_features.shape[1]))
    print("Item features density:", item_features.nnz / (item_features.shape[0] * item_features.shape[1]))
    
    # примеры первых строк
    user_features_csr = user_features.tocsr()
    item_features_csr = item_features.tocsr()
    print("User features (first 5 rows):")
    print(user_features_csr[:5].toarray())
    print("Item features (first 5 rows):")
    print(item_features_csr[:5].toarray())
    
    # анализ уникальности
    print("User features - уникальных строк:", np.unique(user_features.toarray(), axis=0).shape[0])
    print("Item features - уникальных строк:", np.unique(item_features.toarray(), axis=0).shape[0])
    
    # анализ вещественных признаков пользователей
    user_dense = user_features_csr.toarray()
    print("\nUser features (numeric columns) stats:")
    for i in range(3):  # первые 3 признака
        col = user_dense[:, i]
        print(f"Col {i}: mean={col.mean():.3f}, std={col.std():.3f}, min={col.min():.3f}, max={col.max():.3f}")

    # анализ вещественных признаков фильмов
    item_dense = item_features_csr.toarray()
    print("\nItem features (numeric columns) stats:")
    for i in range(2):  # первые 2 признака
        col = item_dense[:, i]
        print(f"Col {i}: mean={col.mean():.3f}, std={col.std():.3f}, min={col.min():.3f}, max={col.max():.3f}")

    print("\nUser binary features (sum per column):")
    print(user_dense[:, 3:].sum(axis=0))
    print("Item binary features (sum per column):")
    print(item_dense[:, 2:].sum(axis=0))
    
    print("\nUser features correlation (first 10 columns):")
    print(np.corrcoef(user_dense[:, :10], rowvar=False))
    print("Item features correlation (first 10 columns):")
    print(np.corrcoef(item_dense[:, :10], rowvar=False))

if __name__ == "__main__":
    analyze_features_weights()