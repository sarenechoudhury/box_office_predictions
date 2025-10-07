import pandas as pd
import numpy as np
import random
import tensorflow as tf
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap

rng = np.random.default_rng(42)
random.seed(42)
tf.random.set_seed(42)

def safe_qcut(series, q=3, labels=None):
    try:
        result = pd.qcut(series, q=q, labels=labels, duplicates='drop')
    except ValueError:
        result = pd.cut(series, bins=q, labels=labels)
    return result

df = pd.read_csv("data/cleaned_movies_metadata.csv")

ratings = pd.read_csv("data/ratings.csv")

# Aggregate per movie
rating_agg = (
    ratings.groupby("movieId")
    .agg(
        avg_user_rating=("rating", "mean"),
        rating_count=("rating", "size"),
        rating_std=("rating", "std")
    )
    .reset_index()
)

links = pd.read_csv("data/links.csv")

df = df.merge(
    links[["tmdbId", "movieId"]],
    left_on="id", right_on="tmdbId", how="left"
)
df = df.merge(rating_agg, on="movieId", how="left")


# Rotten Tomatoes Movie Data
rt = pd.read_csv("data/rotten_tomatoes_movies.csv")

rt = rt.rename(columns={
    "tomatoMeter": "critic_score",
    "audienceScore": "audience_score",
    "genre": "main_genre_rt",
})

rt_cols = [
    "id", "title", "rating", "critic_score", "audience_score",
    "releaseDateTheaters", "runtimeMinutes", "boxOffice"
]
rt = rt[[c for c in rt_cols if c in rt.columns]]

df = df.merge(rt, how="left", on="title", suffixes=("", "_rt"))

# Rotten Tomatoes Movie Reviews Data
reviews = pd.read_csv("data/rotten_tomatoes_movie_reviews.csv")

# Normalize and filter sentiment
reviews["scoreSentiment"] = reviews["scoreSentiment"].str.lower().str.strip()
reviews = reviews[reviews["scoreSentiment"].isin(["positive", "negative"])]

# Aggregate positive/negative counts per movie
agg_reviews = (
    reviews.groupby("id")["scoreSentiment"]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

# Compute sentiment-derived features
total_reviews = agg_reviews[["positive", "negative"]].sum(axis=1).replace(0, np.nan)
agg_reviews["positive_review_ratio"] = agg_reviews["positive"] / total_reviews
agg_reviews["sentiment_gap"] = agg_reviews["positive"] - agg_reviews["negative"]

# Merge aggregated sentiment with RT Data
df = df.merge(
    agg_reviews[["id", "positive_review_ratio", "sentiment_gap"]],
    left_on="id_rt", right_on="id", how="left", suffixes=("", "_review")
)


# Credits and Director Data
credits = pd.read_csv("data/credits.csv")
df = df.merge(credits[["id", "cast", "crew"]], how="left", on="id")

def extract_directors(crew_str):
    try:
        crew_list = ast.literal_eval(crew_str)
        return [c["name"] for c in crew_list if isinstance(c, dict) and c.get("department") == "Directing"]
    except:
        return []

df["directors_list"] = df["crew"].apply(extract_directors)

# Compute mean revenue per director
director_revenue_map = (
    df.explode("directors_list")
      .dropna(subset=["directors_list"])
      .groupby("directors_list")["revenue"]
      .mean()
      .to_dict()
)

def mean_director_revenue(directors):
    if not directors:
        return np.nan
    values = [director_revenue_map.get(d, np.nan) for d in directors]
    values = [v for v in values if not pd.isna(v)]
    return np.mean(values) if values else np.nan

df["mean_director_revenue"] = df["directors_list"].apply(mean_director_revenue)
df["mean_director_revenue"] = df["mean_director_revenue"].fillna(df["mean_director_revenue"].median())

# Parse cast and crew features
def parse_cast(cast_str):
    try:
        cast_list = ast.literal_eval(cast_str)
        return len([c for c in cast_list if c.get('order', 999) < 5])
    except:
        return 0

def parse_crew(crew_str, department):
    try:
        crew_list = ast.literal_eval(crew_str)
        return sum(1 for c in crew_list if c.get('department') == department)
    except:
        return 0

genre_ohe = pd.get_dummies(df['main_genre'], prefix='genre', dummy_na=True)
df = pd.concat([df, genre_ohe], axis=1)
df["budget"] = df["budget"].clip(upper=df["budget"].quantile(0.99))
df["revenue"] = df["revenue"].clip(upper=df["revenue"].quantile(0.99))
df["log_revenue"] = np.log1p(df["revenue"])
df["budget"] = df["budget"].fillna(df["budget"].median())
df["rating"] = df["rating"].astype("category").cat.codes
df["belongs_to_collection"] = df["belongs_to_collection"].notna().astype(int)
df["rating_density"] = df["rating_count"] / df["popularity"].replace(0, np.nan)
df["polarization_index"] = df["rating_std"] / df["avg_user_rating"].replace(0, np.nan)
df['num_top_cast'] = df['cast'].apply(parse_cast)
df['num_directors'] = df['crew'].apply(lambda x: parse_crew(x, 'Directing'))
df['num_writers'] = df['crew'].apply(lambda x: parse_crew(x, 'Writing'))
df['num_producers'] = df['crew'].apply(lambda x: parse_crew(x, 'Production'))
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_quarter'] = df['release_date'].dt.quarter
df['is_holiday_season'] = df['release_month'].isin([11, 12]).astype(int)
df['is_summer_blockbuster'] = df['release_month'].isin([5, 6, 7, 8]).astype(int)
df['sentiment_gap'] = df['audience_score'] - df['critic_score']
df["budget_popularity"] = df["budget"] * df["popularity"]
df["budget_runtime"] = df["budget"] * df["runtime"]
df["popularity_squared"] = df["popularity"] ** 2
df["budget_to_runtime_ratio"] = df["budget"] / (df["runtime"] + 1)
df['genre_popularity'] = df['popularity'] * df['genre_Action']  
df['log_rating_count'] = np.log1p(df['rating_count'])

features = df.select_dtypes(include=[np.number]).columns.tolist()

drop_cols = ['revenue', 'log_revenue', 'id', 'movieId', 'tmdbId']
features = [f for f in features if f not in drop_cols]
target = 'log_revenue'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
# Fit/transform only the columns you truly want scaled:
cols_to_scale = ['popularity','runtime']
X_train_scaled_block = X_train[cols_to_scale].copy()
X_test_scaled_block  = X_test[cols_to_scale].copy()

X_train[cols_to_scale] = scaler.fit_transform(X_train_scaled_block)
X_test[cols_to_scale]  = scaler.transform(X_test_scaled_block)

X_train_aug = X_train.copy()
X_test_aug = X_test.copy()
y_resid = y_train.copy()

N_ITER = 3
for iteration in range(N_ITER):
    print(f"\n=== AugBoost Iteration {iteration + 1} ===")

    lgb_train = lgb.Dataset(X_train_aug, label=y_resid)
    lgb_model = lgb.train({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}, 
                          lgb_train, num_boost_round=100)

    pred_train = lgb_model.predict(X_train_aug)
    y_resid = y_train - pred_train

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test_aug)

    ann_model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)
    ])
    ann_model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
    ann_model.fit(X_train_scaled, y_resid, 
                  validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

    ann_train_features = ann_model.predict(X_train_scaled)
    ann_test_features = ann_model.predict(X_test_scaled)

    X_train_aug[f'ann_out_{iteration}'] = ann_train_features.flatten()
    X_test_aug[f'ann_out_{iteration}'] = ann_test_features.flatten()

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "lambda_l1": 1.0,
    "lambda_l2": 2.0,
    "num_leaves": 63,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "min_data_in_bin": 10,
    "min_gain_to_split": 0.01,
    "seed": 42,
    "deterministic": True,
    "verbosity": -1
}

final_model = LGBMRegressor(**params, n_estimators=1000)
final_model.fit(
    X_train_aug, y_train,
    eval_set=[(X_train_aug, y_train), (X_test_aug, y_test)],
    eval_names=["train", "valid"],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(25, verbose=True)],
)

final_preds = final_model.predict(X_test_aug)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
print(f"\n✅ Final AugBoost RMSE: {rmse:.4f}")

importances = final_model.feature_importances_
feature_names = final_model.feature_name_

imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
imp_df = imp_df.sort_values(by='Importance', ascending=False)
