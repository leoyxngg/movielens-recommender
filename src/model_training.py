from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from data_processing import rate_train, rate_val


def build_item_user_matrix(
    ratings_df: pd.DataFrame,
) -> tuple[csr_matrix, dict[int, int], dict[int, int], float]:
    """Build a movies x users sparse matrix and id-to-index maps."""
    movie_ids = np.sort(ratings_df["movieId"].unique())
    user_ids = np.sort(ratings_df["userId"].unique())
    movie_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}

    rows = ratings_df["movieId"].map(movie_index).to_numpy()
    cols = ratings_df["userId"].map(user_index).to_numpy()
    data = ratings_df["rating"].to_numpy(dtype=float)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(movie_ids), len(user_ids)),
    )
    global_mean = float(data.mean())
    return matrix, user_index, movie_index, global_mean


def predict_weighted_rating(
    user_idx: int,
    movie_idx: int,
    train_matrix: csr_matrix,
    knn: NearestNeighbors,
    k: int,
    global_mean: float,
) -> float:
    """
    Predict a rating from k similar movies the user has already rated.

    Cosine distance from sklearn is converted to similarity (1 - distance),
    then used as weights in a weighted average.
    """
    n_movies = train_matrix.shape[0]
    n_neighbors = min(k + 1, n_movies)
    distances, neighbor_indices = knn.kneighbors(
        train_matrix[movie_idx],
        n_neighbors=n_neighbors,
    )

    weighted_sum = 0.0
    similarity_sum = 0.0
    for distance, neighbor_idx in zip(distances[0], neighbor_indices[0]):
        if neighbor_idx == movie_idx:
            continue

        rating = train_matrix[neighbor_idx, user_idx]
        if rating == 0:
            continue

        similarity = 1.0 - distance
        if similarity <= 0:
            continue

        weighted_sum += similarity * rating
        similarity_sum += similarity

    if similarity_sum == 0:
        return global_mean
    return weighted_sum / similarity_sum


def rmse_on_ratings(
    ratings_df: pd.DataFrame,
    train_matrix: csr_matrix,
    user_index: dict[int, int],
    movie_index: dict[int, int],
    knn: NearestNeighbors,
    k: int,
    global_mean: float,
) -> float:
    squared_errors: list[float] = []
    for row in ratings_df.itertuples(index=False):
        user_idx = user_index.get(row.userId)
        movie_idx = movie_index.get(row.movieId)
        if user_idx is None or movie_idx is None:
            continue

        prediction = predict_weighted_rating(
            user_idx,
            movie_idx,
            train_matrix,
            knn,
            k,
            global_mean,
        )
        squared_errors.append((prediction - row.rating) ** 2)

    if not squared_errors:
        return float("inf")
    return float(np.sqrt(np.mean(squared_errors)))


def cross_validate_k(
    ratings_df: pd.DataFrame,
    k: int,
    cv_folds: int = 5,
    random_state: int = 42,
) -> float:
    """Average RMSE across K-fold splits of rating rows."""
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_scores: list[float] = []

    for train_idx, val_idx in kfold.split(ratings_df):
        fold_train = ratings_df.iloc[train_idx]
        fold_val = ratings_df.iloc[val_idx]

        train_matrix, user_index, movie_index, global_mean = build_item_user_matrix(
            fold_train
        )
        n_neighbors = min(k + 1, train_matrix.shape[0])
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(train_matrix)

        fold_scores.append(
            rmse_on_ratings(
                fold_val,
                train_matrix,
                user_index,
                movie_index,
                knn,
                k,
                global_mean,
            )
        )

    return float(np.mean(fold_scores))


def train_knn_model(
    train_ratings: pd.DataFrame,
    kvals: list[int],
    cv_folds: int = 5,
    metric: str = "cosine",
) -> tuple[int, dict[int, float], NearestNeighbors, csr_matrix, float]:
    """
    Tune k with cross-validation, then fit item-based KNN on all training ratings.

    Returns:
        best_k, cv_rmse_by_k, fitted_knn, train_matrix, global_mean
    """
    cv_rmse_by_k: dict[int, float] = {}
    for k in kvals:
        cv_rmse_by_k[k] = cross_validate_k(train_ratings, k, cv_folds=cv_folds)

    best_k = min(cv_rmse_by_k, key=cv_rmse_by_k.get)
    train_matrix, _, _, global_mean = build_item_user_matrix(train_ratings)
    n_neighbors = min(best_k + 1, train_matrix.shape[0])
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(train_matrix)

    return best_k, cv_rmse_by_k, knn, train_matrix, global_mean


def evaluate_holdout(
    val_ratings: pd.DataFrame,
    train_ratings: pd.DataFrame,
    k: int,
    knn: NearestNeighbors | None = None,
) -> float:
    """RMSE on a holdout set using a model trained on train_ratings."""
    train_matrix, user_index, movie_index, global_mean = build_item_user_matrix(
        train_ratings
    )
    if knn is None:
        n_neighbors = min(k + 1, train_matrix.shape[0])
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(train_matrix)

    return rmse_on_ratings(
        val_ratings,
        train_matrix,
        user_index,
        movie_index,
        knn,
        k,
        global_mean,
    )


if __name__ == "__main__":
    candidate_k = [10, 50, 100, 200, 400, 800]
    best_k, cv_scores, knn_model, _, _ = train_knn_model(rate_train, candidate_k)

    print("Item-based KNN — cross-validation RMSE by k:")
    for k, score in sorted(cv_scores.items()):
        marker = " <- best" if k == best_k else ""
        print(f"  k={k:>2}: {score:.4f}{marker}")

    holdout_rmse = evaluate_holdout(rate_val, rate_train, best_k, knn_model)
    print(f"\nBest k: {best_k}")
    print(f"Holdout validation RMSE: {holdout_rmse:.4f}")