# model/lightfm/prepare_data.py
import logging
import pandas as pd
from typing import Tuple
from src.data_loader import load_movies, load_reviews, load_users
from src.data_processor import create_dataframes, merge_datasets

def get_dataframes(
    path_movies="data/dataset/u.item",
    path_reviews="data/dataset/u.data",
    path_users="data/dataset/u.user"
) -> Tuple[pd.DataFrame, pd.DataFrame] :
    
    logging.info("Loading movies, reviews and users...")
    movies_list = load_movies(path_movies) 
    reviews_list = load_reviews(path_reviews) 
    users_list = load_users(path_users) 

    logging.info("Creating DataFrames")
    movies_df, reviews_df, users_df = create_dataframes(
        movies_list, reviews_list, users_list
    )
    
    logging.info("Merging DataFrames...")
    full_dataset_df = merge_datasets(
        movies_df, reviews_df, users_df
    )
    
    logging.info(f"DataFrames ready: {len(reviews_df)} reviews, {len(full_dataset_df)} full records")
    return reviews_df, full_dataset_df
    
    