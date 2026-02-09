import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor

DATA_FILE = "Airbnb_Open_Data.csv"
MODEL_BUNDLE_FILE = "model_bundle.joblib"


@dataclass
class Bundle:
    model: object
    metrics: dict
    features: list
    defaults: dict
    importances: pd.DataFrame


def _clean_dataframe(path: str = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    for money_col in ["price", "service fee"]:
        df[money_col] = df[money_col].replace("[\\$,]", "", regex=True)
        df[money_col] = pd.to_numeric(df[money_col], errors="coerce")

    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    return df


def _build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def _feature_importances(model, feature_names):
    reg = model.regressor_
    importances = reg.feature_importances_
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def train_model_bundle(path: str = DATA_FILE) -> Bundle:
    df = _clean_dataframe(path)

    max_rows = int(os.getenv("TRAIN_MAX_ROWS", "0"))
    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    features = [
        "neighbourhood group",
        "neighbourhood",
        "room type",
        "instant_bookable",
        "cancellation_policy",
        "lat",
        "long",
        "Construction year",
        "minimum nights",
        "number of reviews",
        "reviews per month",
        "review rate number",
        "calculated host listings count",
        "availability 365",
    ]
    target = "price"

    numeric_features = [
        "lat",
        "long",
        "Construction year",
        "minimum nights",
        "number of reviews",
        "reviews per month",
        "review rate number",
        "calculated host listings count",
        "availability 365",
    ]
    categorical_features = [
        "neighbourhood group",
        "neighbourhood",
        "room type",
        "instant_bookable",
        "cancellation_policy",
    ]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            max_depth=30,
            random_state=42,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=1,
            max_depth=None,
            random_state=42,
            n_jobs=1,
        ),
    }

    best = None
    best_metrics = None
    best_name = None

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", estimator)]
        )
        model = TransformedTargetRegressor(
            regressor=pipeline, func=np.log1p, inverse_func=np.expm1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        if best is None or r2 > best_metrics["r2"]:
            best = model
            best_name = name
            best_metrics = {"mae": float(mae), "r2": float(r2)}

    preprocess = best.regressor_.named_steps["preprocessor"]
    onehot_names = (
        preprocess.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )
    transformed_feature_names = numeric_features + onehot_names

    defaults = {
        "lat": float(df["lat"].mean()),
        "long": float(df["long"].mean()),
        "Construction year": int(df["Construction year"].median()),
        "reviews per month": float(df["reviews per month"].median()),
        "review rate number": float(df["review rate number"].median()),
        "calculated host listings count": float(
            df["calculated host listings count"].median()
        ),
    }

    importances = _feature_importances(best, transformed_feature_names)

    metrics = {
        **best_metrics,
        "best_model": best_name,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    return Bundle(
        model=best,
        metrics=metrics,
        features=features,
        defaults=defaults,
        importances=importances,
    )


def save_bundle(bundle: Bundle, bundle_path: str = MODEL_BUNDLE_FILE):
    payload = {
        "model": bundle.model,
        "metrics": bundle.metrics,
        "features": bundle.features,
        "defaults": bundle.defaults,
        "importances": bundle.importances,
    }
    joblib.dump(payload, bundle_path)


def load_or_train_bundle(bundle_path: str = MODEL_BUNDLE_FILE):
    if os.path.exists(bundle_path):
        return joblib.load(bundle_path)

    bundle = train_model_bundle()
    save_bundle(bundle, bundle_path)
    return joblib.load(bundle_path)


if __name__ == "__main__":
    trained_bundle = train_model_bundle()
    save_bundle(trained_bundle)
    print("Saved improved model bundle to model_bundle.joblib")
    print(trained_bundle.metrics)
