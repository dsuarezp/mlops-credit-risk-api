"""Train a credit risk classifier and persist it for serving."""

import json
import joblib
from sklearn.datasets import fetch_openml
from catboost import CatBoostClassifier

MODEL_PATH = "model.pkl"


def load_data():
    """Load the German Credit dataset (public, credit-risk domain)."""
    data = fetch_openml("credit-g", version=1, as_frame=True)
    X = data.data
    y = (data.target == "good").astype(int)  # 1 = good payer, 0 = bad payer

    # CatBoost handles categoricals natively; just declare them as strings
    cat_features = X.select_dtypes(exclude="number").columns.tolist()
    X[cat_features] = X[cat_features].astype(str)
    return X, y, cat_features


def train():
    X, y, cat_features = load_data()

    model = CatBoostClassifier(iterations=200, depth=4, verbose=0)
    model.fit(X, y, cat_features=cat_features)

    # Persist model + column order (critical: inference must use the same order)
    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Print one example request, ready to paste into the API test later
    sample = json.loads(X.iloc[[0]].to_json(orient="records"))[0]
    print("Sample request:", json.dumps({"features": sample}))


if __name__ == "__main__":
    train()