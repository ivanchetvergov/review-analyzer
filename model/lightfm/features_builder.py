# model/lightfm/features.py
import pandas as pd
import logging
from typing import Optional

def build_user_features(
    reviews_df: pd.DataFrame,
    movies_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Синтезирует признаки пользователей на основе их активности и жанровых предпочтений.
    
    Аргументы:
        reviews_df (pd.DataFrame): DataFrame с данными об отзывах.
        movies_df (pd.DataFrame): DataFrame с данными о фильмах.
        
    Возвращает:
        Optional[pd.DataFrame]: DataFrame с признаками пользователей или None, если reviews_df пуст.
    """
    if reviews_df.empty:
        logging.warning("Reviews DataFrame is empty. Cannot build user features.")
        return None

    logging.info("Building user features based on activity and genre preferences...")
    
    user_activity_features = reviews_df.groupby('user_id')['rating'].agg(
        num_ratings='count',
        mean_rating='mean',
        rating_var='var'
    ).reset_index()
    
    # заполняем NaN-значения для дисперсии (у пользователей с одной оценкой)
    user_activity_features['rating_var'] = user_activity_features['rating_var'].fillna(0)
    
    merged_df = pd.merge(reviews_df, movies_df, on='movie_id')
    
    # разделяем и разворачиваем жанры
    merged_df['genre'] = merged_df['genres'].str.split('|')
    exploded_df = merged_df.explode('genre')
    
    # агрегируем по user_id и жанру
    genre_features = exploded_df.groupby(['user_id', 'genre'])['rating'].agg(
        avg_rating='mean',
        count='count'
    ).reset_index()
    
    # cоздаем pivot-таблицу для средних рейтингов по жанрам
    avg_rating_pivot = genre_features.pivot(index='user_id', columns='genre', values='avg_rating').fillna(0)
    avg_rating_pivot.columns = [f'avg_rating_{col.replace(" ", "_")}' for col in avg_rating_pivot.columns]
    
    # Создаем pivot-таблицу для количества оценок по жанрам
    count_pivot = genre_features.pivot(index='user_id', columns='genre', values='count').fillna(0)
    count_pivot.columns = [f'count_{col.replace(" ", "_")}' for col in count_pivot.columns]

    # Объединяем все признаки
    user_features = pd.merge(user_activity_features, avg_rating_pivot, on='user_id')
    user_features = pd.merge(user_features, count_pivot, on='user_id')

    return user_features

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    from src.data_loader import load_movies, load_reviews
    
    movies_list = load_movies("data/ml-20m/movies.csv", nrows=10_000)
    reviews_list = load_reviews("data/ml-20m/ratings.csv", nrows=1_000_000)

    from src.data_processor import create_dataframes
    
    movies_df, reviews_df, _ = create_dataframes(movies_list, reviews_list, None)
    
    # Строим признаки пользователей
    user_features_df = build_user_features(reviews_df, movies_df)
    
    if user_features_df is not None:
        print("\nСинтезированные признаки пользователей:")
        print(user_features_df.head())
        print(f"\nРазмер DataFrame с признаками: {user_features_df.shape}")
        print("\nТипы данных колонок:")
        user_features_df.info()

    
    
    
