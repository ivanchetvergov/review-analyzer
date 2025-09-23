# eda.py
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_loader import load_movies, load_reviews, load_users
from src.data_processor import create_dataframes, merge_datasets

def run_eda():
    """
    Главная функция для выполнения полного EDA.
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading data from files...")
    movies_list = load_movies("data/dataset/u.item")
    reviews_list = load_reviews("data/dataset/u.data")
    users_list = load_users("data/dataset/u.user")

    movies_df, reviews_df, users_df = create_dataframes(
        movies_list, reviews_list, users_list
    )
    
    full_dataset_df = merge_datasets(movies_df, reviews_df, users_df)

    logging.info("Full dataset created!")

    logging.info("Starting Exploratory Data Analysis...")

    print("\nShape of the merged DataFrame:", full_dataset_df.shape)
    print("\nColumns and their data types:")
    full_dataset_df.info()

    # визуализация распределения рейтингов
    plt.figure(figsize=(8, 6))
    sns.countplot(x='rating_value', data=full_dataset_df)
    plt.title('Distribution of Movie Ratings', fontsize=16)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.show()

    # топ 10 фильмов
    top_movies = full_dataset_df['title'].value_counts().head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(y=top_movies.index, x=top_movies.values)
    plt.title('Top 10 Most Rated Movies', fontsize=16)
    plt.xlabel('Number of Ratings', fontsize=12)
    plt.ylabel('Movie Title', fontsize=12)
    plt.show()

if __name__ == "__main__":
    run_eda()