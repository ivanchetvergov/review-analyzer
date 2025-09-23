# model/lightfm/train_model.py

import logging
from lightfm import LightFM  # type: ignore
from lightfm.evaluation import precision_at_k, auc_score # type: ignore
from model.lightfm.prepare_data import get_dataframes
from model.lightfm.build_matrices import build_interactions_matrix, build_features_matrices 

logging.basicConfig(level=logging.INFO)

# подготовка данных
reviews_df, full_dataset_df = get_dataframes()
interactions, user_to_idx, movie_to_idx = build_interactions_matrix(reviews_df)
user_features, item_features = build_features_matrices(full_dataset_df, user_to_idx, movie_to_idx)

# создаем модель
model = LightFM(
    no_components=30,
    loss='warp',
    random_state=42
)

# обучение
logging.info("Starting model training...")
model.fit(
    interactions,
    user_features=user_features,
    item_features=item_features,
    epochs=30,
    num_threads=4
)
logging.info("Training finished!")

# оценка модели
K = 5
train_precision = precision_at_k(
    model,
    interactions,
    user_features=user_features,
    item_features=item_features,
    k=K
).mean()

train_auc = auc_score(
    model,
    interactions,
    user_features=user_features,
    item_features=item_features
).mean()

logging.info(f"Train precision@{K}: {train_precision:.4f}")
logging.info(f"Train AUC: {train_auc:.4f}")