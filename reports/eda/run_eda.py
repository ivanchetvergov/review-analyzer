# reports/eda/run_eda.py
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_loader import load_movies, load_reviews
from src.data_processor import create_dataframes, merge_datasets

ASSETS_DIR = "assets/eda"

def _ensure_assets_dir():
    os.makedirs(ASSETS_DIR, exist_ok=True)

def run_eda():
    """
    Главная функция для выполнения полного EDA.
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading data from files...")
    movies_list = load_movies("data/dataset/movies.csv", nrows=1_000_000)
    reviews_list = load_reviews("data/dataset/ratings.csv", nrows=3_000_000)


    movies_df, reviews_df, _ = create_dataframes(
        movies_list, reviews_list
    )
    
    full_dataset_df = merge_datasets(movies_df, reviews_df)

    logging.info("Full dataset created!")

    logging.info("Starting Exploratory Data Analysis...")

    print("\nShape of the merged DataFrame:", full_dataset_df.shape)
    print("\nColumns and their data types:")
    full_dataset_df.info()

    # визуализация распределения рейтингов
    plt.figure(figsize=(8, 6))
    sns.countplot(x='rating', data=full_dataset_df)
    plt.title('Распределения рейтингов', fontsize=16)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "ratings_distribution.png"))
    plt.close()

    # топ 10 фильмов
    top_movies = full_dataset_df['title'].value_counts().head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(y=top_movies.index, x=top_movies.values)
    plt.title('Top 10 Most Rated Movies', fontsize=16)
    plt.xlabel('Number of Ratings', fontsize=12)
    plt.ylabel('Movie Title', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "top10_movies.png"))
    plt.close()
    
    genre_counts = full_dataset_df["genres"].str.split("|").explode().value_counts()
    top_genres = genre_counts.head(15).index

    top_genre_ratings = (
        full_dataset_df["genres"]
        .str.split("|")
        .explode()
        .loc[lambda x: x.isin(top_genres)]
        .to_frame()
        .join(full_dataset_df["rating"])
        .groupby("genres")["rating"].mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10,6))
    sns.barplot(x=top_genre_ratings.index, y=top_genre_ratings.values)
    plt.title("Средний рейтинг по топ-15 жанрам")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "avg_rating_by_genre.png"))
    plt.close()

    # динамика по годам
    yearly_ratings = full_dataset_df.groupby("year")["rating"].mean().dropna()
    plt.figure(figsize=(12,6))
    sns.lineplot(x=yearly_ratings.index, y=yearly_ratings.values, marker="o")
    plt.title("Средний рейтинг фильмов по годам выпуска")
    plt.xlabel("год")
    plt.ylabel("средний рейтинг")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "avg_rating_by_year.png"))
    plt.close()

    # проверка уникальности отзывов
    duplicates = full_dataset_df.duplicated(subset=["user_id", "movie_id"]).sum()
    print(f"\nколичество дубликатов отзывов: {duplicates}")

if __name__ == "__main__":
    run_eda()