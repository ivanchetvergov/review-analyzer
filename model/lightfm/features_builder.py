# model/lightfm/features.py
import pandas as pd
import numpy as np
import logging
from typing import Optional
from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans

def build_user_features(
    reviews_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_genre_users=50
) -> Optional[pd.DataFrame]:
    """
    Формирует признаки для пользователей на основе их активности и жанровых предпочтений.

    Основные шаги:
    - Считает статистики по отзывам пользователя: количество, средний рейтинг, дисперсия.
    - Стандартизирует числовые признаки.
    - Выделяет жанры, которые пользователь смотрел, и бинаризует их.
    - Оставляет только популярные жанры (просмотрены минимум min_genre_users пользователями).
    - Для каждого пользователя определяет топ-3 жанра и кодирует их one-hot.
    - Объединяет статистики и жанровые признаки в итоговый DataFrame.
    - Добавляет признак user_bias для учета смещения пользователя.

    Возвращает:
        DataFrame с признаками пользователей, где строки — пользователи, столбцы — признаки.
    """
    if reviews_df.empty:
        logging.warning("Reviews DataFrame is empty. Cannot build user features.")
        return None

    # --- базовые статистики ---
    user_stats = reviews_df.groupby('user_id')['rating'].agg(
        num_ratings='count',
        mean_rating='mean',
        rating_var='var'
    ).fillna(0).reset_index()
    user_stats['num_ratings'] = np.log1p(user_stats['num_ratings'])
    scaler = StandardScaler()
    user_stats[['mean_rating', 'rating_var']] = scaler.fit_transform(user_stats[['mean_rating', 'rating_var']])

    # --- жанровые предпочтения ---
    merged = pd.merge(reviews_df, movies_df[['movie_id', 'genres']], on='movie_id')
    merged['genre'] = merged['genres'].str.split('|')
    exploded = merged.explode('genre')
    user_genres = exploded.groupby(['user_id', 'genre'])['rating'].count().unstack(fill_value=0)

    # оставляем только жанры, которые посмотрели > min_genre_users
    genre_counts = (user_genres > 0).sum(axis=0)
    kept_genres = genre_counts[genre_counts >= min_genre_users].index
    user_genres = user_genres[kept_genres]
    user_genres[user_genres > 0] = 1  # бинаризация

    # топ-3 жанра
    top_genres = user_genres.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
    top_genres_df = pd.DataFrame(index=user_genres.index)
    for i in range(3):
        top_genres_df[f'top_genre_{i+1}'] = top_genres.apply(lambda x: x[i] if len(x) > i else None)
    top_genres_df = pd.get_dummies(top_genres_df, dummy_na=True)

    # объединяем статистики и жанры
    user_features = pd.concat([user_stats.set_index('user_id'), top_genres_df], axis=1)
    user_features.reset_index(inplace=True)

    # добавляем bias
    user_features['user_bias'] = 1.0

    return user_features

def build_item_features(
    reviews_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_genre_movies=20
) -> Optional[pd.DataFrame]:
    """
    Формирует признаки для фильмов на основе жанра, года выпуска, рейтинга

    Основные шаги:
    - Кодирует жанры фильмов в one-hot, оставляя только популярные (просмотрены минимум min_genre_movies раз).
    - Бинует год выпуска фильма и кодирует его one-hot.
    - Считает средний рейтинг и количество отзывов для каждого фильма, стандартизирует рейтинг.
    - Выделяет признак популярности (топ-10% по количеству отзывов).
    - Объединяет все признаки в итоговый DataFrame.
    - Добавляет признак item_bias для учета смещения фильма.

    Возвращает:
        DataFrame с признаками фильмов, где строки — фильмы, столбцы — признаки.
    """
    if movies_df.empty:
        logging.warning("Movies DataFrame is empty. Cannot build item features.")
        return None

    # --- жанры ---
    genres_df = movies_df['genres'].str.get_dummies(sep='|')
    # оставляем только популярные жанры
    genre_counts = (genres_df > 0).sum(axis=0)
    kept_genres = genre_counts[genre_counts >= min_genre_movies].index
    genres_df = genres_df[kept_genres]
    genres_df.columns = [f'genre_{c}' for c in genres_df.columns]

    # --- возраст фильма ---
    current_year = 2025
    movies_df['movie_age'] = current_year - movies_df['year']
    bins = [1900, 1980, 1990, 2000, 2010, 2020, 2025]
    labels = ['<1980','1980-1989','1990-1999','2000-2009','2010-2019','>=2020']
    year_bin = pd.cut(movies_df['year'], bins=bins, labels=labels, right=False)
    year_bin_df = pd.get_dummies(year_bin, prefix='year_bin')

    # --- рейтинг и популярность ---
    movie_stats = reviews_df.groupby('movie_id')['rating'].agg(
        avg_rating='mean',
        num_ratings='count'
    ).reset_index()
    movie_stats['num_ratings'] = np.log1p(movie_stats['num_ratings'])
    movie_stats[['avg_rating']] = StandardScaler().fit_transform(movie_stats[['avg_rating']])

    # топ-10% популярных
    threshold = movie_stats['num_ratings'].quantile(0.9)
    movie_stats['popular'] = (movie_stats['num_ratings'] >= threshold).astype(int)

    # объединяем
    item_features = movies_df[['movie_id']].merge(movie_stats, on='movie_id', how='left')
    item_features = pd.concat([item_features, genres_df, year_bin_df], axis=1)

    # добавляем bias
    item_features['item_bias'] = 1.0

    return item_features

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    from src.data_loader import load_movies, load_reviews
    
    movies_list = load_movies("data/ml-20m/movies.csv", nrows=100_000)
    reviews_list = load_reviews("data/ml-20m/ratings.csv", nrows=1_000_000)

    from src.data_processor import create_dataframes
    
    movies_df, reviews_df, _ = create_dataframes(movies_list, reviews_list, None)

    # строим признаки фильмов
    item_features_df = build_item_features(movies_df)
    user_features_df = build_user_features(reviews_df, movies_df)
    
    if item_features_df is not None and user_features_df is not None:
        print("\nСинтезированные признаки пользователей:")
        print(user_features_df.head())
        print(f"\nРазмер DataFrame с признаками: {user_features_df.shape}")
        print("\nТипы данных колонок:")
        user_features_df.info()

        print("\nСинтезированные признаки фильмов:")
        print(item_features_df.head())
        print(f"\nРазмер DataFrame с признаками: {item_features_df.shape}")
        print("\nТипы данных колонок:")
        item_features_df.info()




    
    
    
