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
    формирует user features для lightfm:
    - все числовые статистики (num_ratings, mean_rating, rating_var) бинируются
    - count_<genre> бинируется
    - avg_rating_<genre> стандартизированы
    - топ-3 жанра one-hot
    - добавлен user_bias
    """

    if reviews_df.empty:
        logging.warning("Reviews DataFrame is empty. Cannot build user features.")
        return None

    logging.info("Building user features...")

    # --- 1. базовые статистики ---
    user_stats = reviews_df.groupby('user_id')['rating'].agg(
        num_ratings='count',
        mean_rating='mean',
        rating_var='var'
    ).fillna(0).reset_index()
    
    user_stats['num_ratings'] = np.log1p(user_stats['num_ratings'])
    # не стандартизируем, будем бинировать
    user_numeric_cols = ['num_ratings', 'mean_rating', 'rating_var']

    # --- 2. жанровые предпочтения ---
    merged = pd.merge(reviews_df, movies_df[['movie_id', 'genres']], on='movie_id')
    merged['genre'] = merged['genres'].str.split('|')
    exploded = merged.explode('genre')

    genre_features = exploded.groupby(['user_id', 'genre'])['rating'].agg(
        count='count', avg_rating='mean'
    ).reset_index()

    count_pivot = genre_features.pivot(index='user_id', columns='genre', values='count').fillna(0)
    count_pivot.columns = [f'count_{col.replace(" ", "_")}' for col in count_pivot.columns]

    avg_rating_pivot = genre_features.pivot(index='user_id', columns='genre', values='avg_rating').fillna(0)
    avg_rating_pivot.columns = [f'avg_rating_{col.replace(" ", "_")}' for col in avg_rating_pivot.columns]

    # --- 3. объединяем и бинируем ---
    user_features_numeric = pd.merge(user_stats, count_pivot, on='user_id', how='left').fillna(0)
    final_features_df = user_features_numeric[['user_id']].copy()

    # все числовые колонки для бинирования
    cols_to_bin = user_numeric_cols + [col for col in user_features_numeric.columns if col.startswith('count_')]

    for col in cols_to_bin:
        try:
            binned = pd.qcut(user_features_numeric[col], q=5, labels=False, duplicates='drop')
        except ValueError:
            binned = pd.qcut(user_features_numeric[col], q=3, labels=False, duplicates='drop')
        binned_dummies = pd.get_dummies(binned, prefix=f'bin_{col}')
        final_features_df = pd.concat([final_features_df, binned_dummies], axis=1)

    # --- 4. стандартизированные avg_rating_<genre> ---
    scaler_avg = StandardScaler()
    avg_scaled = avg_rating_pivot.copy()
    for col in avg_scaled.columns:
        avg_scaled[col] = scaler_avg.fit_transform(avg_scaled[col].values.reshape(-1, 1))
    avg_scaled.reset_index(inplace=True)
    final_features_df = pd.merge(final_features_df, avg_scaled, on='user_id', how='left').fillna(0)

    # --- 5. топ-3 жанра ---
    user_genres_counts = exploded.groupby(['user_id', 'genre'])['rating'].count().unstack(fill_value=0)
    genre_counts = (user_genres_counts > 0).sum(axis=0)
    kept_genres = genre_counts[genre_counts >= min_genre_users].index
    user_genres_filtered = user_genres_counts[kept_genres]
    top_genres = user_genres_filtered.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
    top_genres_df = pd.DataFrame(index=user_genres_filtered.index)
    for i in range(3):
        col_name = f'top_genre_{i+1}'
        top_genres_df[col_name] = top_genres.apply(lambda x: x[i] if len(x) > i else np.nan)
    top_genres_df = pd.get_dummies(top_genres_df, dummy_na=True, prefix='top_genre')
    top_genres_df.reset_index(inplace=True)
    final_features_df = pd.merge(final_features_df, top_genres_df, on='user_id', how='left').fillna(0)

    # --- 6. bias ---
    final_features_df['user_bias'] = 1.0

    return final_features_df


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




    
    
    
