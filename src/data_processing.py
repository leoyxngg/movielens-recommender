from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

# Load and process ratings data
ratings = pd.read_csv(DATA_DIR / "ratings.csv")
processed_ratings = ratings.dropna().drop_duplicates()
processed_ratings["timestamp"] = pd.to_datetime(
    processed_ratings["timestamp"],
    unit="s",
)

# Load and process movies data
movies = pd.read_csv(DATA_DIR / "movies.csv")
processed_movies = movies.dropna().drop_duplicates().copy()
stripped_titles = processed_movies["title"].str.strip()
extracted = stripped_titles.str.extract(r"^(.*)\s\((\d{4})\)$")
processed_movies["title"] = extracted[0].fillna(stripped_titles)
processed_movies["year"] = pd.to_numeric(extracted[1], errors="coerce").astype("Int64")
processed_movies["genres"] = processed_movies["genres"].str.split("|")

processed_data = processed_ratings.merge(processed_movies, on="movieId")

rate_build, rate_test = train_test_split(processed_ratings, test_size=0.2, random_state=42)
rate_train, rate_val = train_test_split(rate_build, test_size=0.2, random_state=42)