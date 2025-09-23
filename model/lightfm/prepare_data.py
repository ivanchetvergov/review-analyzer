# model/lightfm/prepare_data.py
import logging
import pandas as pd
from typing import Tuple
from src.data_loader import load_movies, load_reviews
from src.data_processor import create_dataframes, merge_datasets


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
    
    