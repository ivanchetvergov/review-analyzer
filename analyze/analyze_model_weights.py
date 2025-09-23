# analyze/analyze_model_weights.py
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUD_DIR = 'assets/lightfm_weights'

def analyze_models_weights(model, user_features, item_features, user_feature_names, item_feature_names, top_n=10):
    """
    Анализирует веса признаков LightFM модели

    model: обученная LightFM модель
    user_features: матрица признаков пользователей (sparse)
    item_features: матрица признаков айтемов (sparse)
    user_feature_names: список имен user features
    item_feature_names: список имен item features
    top_n: сколько топовых признаков выводить
    """
    
    os.makedirs(OUTPUD_DIR, exist_ok=True)

    # получаем embeddings с учётом фичей
    user_emb, user_bias = model.get_user_representations()
    item_emb, item_bias = model.get_item_representations()

    # усредняем по компонентам (axis=1 — по embedding dimension)
    user_coef = np.mean(user_emb, axis=1) if user_emb.ndim > 1 else user_emb
    item_coef = np.mean(item_emb, axis=1) if item_emb.ndim > 1 else item_emb


    def plot_top_features(coefs, names, title="Top features", filename="plot.png"):
        idx = np.argsort(np.abs(coefs))[-top_n:][::-1]
        top_coefs = coefs[idx]
        top_names = [names[i] for i in idx]

        plt.figure(figsize=(10, 6))
        plt.barh(top_names[::-1], top_coefs[::-1], color='skyblue')
        plt.xlabel("Average weight")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUD_DIR, filename))
        plt.close()

    # строим графики
    plot_top_features(user_coef, user_feature_names, "Top User Features", "top_user_features.png")
    plot_top_features(item_coef, item_feature_names, "Top Item Features", "top_item_features.png")

    print("Top User Features:")
    for i in np.argsort(-np.abs(user_coef))[:top_n]:
        print(f"{user_feature_names[i]}: {user_coef[i]:.4f}")

    print("\nTop Item Features:")
    for i in np.argsort(-np.abs(item_coef))[:top_n]:
        print(f"{item_feature_names[i]}: {item_coef[i]:.4f}")

