"""Step 2: cleaned table -> trained model bundle + test metrics.

Rebuilds the final models from 02_modelling.ipynb with the tuned hyperparameters in config/params.json:
a stacking ensemble (Random Forest + XGBoost + CatBoost, RidgeCV on top) for price per perch, and a
Random Forest classifier for the Low / Medium / High band. The hold-out row, train/test split and seed are
the same as in the notebook, so results are directly comparable.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor

from catboost_wrapper import CatBoostAreaRegressor  # top-level import so the pickle loads in the app

TARGET_RAW, TARGET = "Price per Perch", "log_price_per_perch"
CAT = ["Address"]
BANDS = ["Low", "Medium", "High"]


def split(df, seed, test_size):
    features = [c for c in df.columns if c not in ["Posted_Date_new", TARGET_RAW, TARGET]]
    holdout = df.sample(1, random_state=seed)
    data = df.drop(holdout.index).reset_index(drop=True)
    strata = pd.qcut(data[TARGET], 3, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data[TARGET], test_size=test_size, random_state=seed, stratify=strata)
    return features, holdout, data, X_train, X_test, y_train, y_test


def _prep(seed, target_type="continuous"):
    return ColumnTransformer([("area", TargetEncoder(target_type=target_type, random_state=seed), CAT)],
                             remainder="passthrough")


def build_regressor(p, seed):
    rf = Pipeline([("prep", _prep(seed)),
                   ("model", RandomForestRegressor(random_state=seed, n_jobs=1, **p["random_forest"]))])
    xgb = Pipeline([("prep", _prep(seed)), ("model", XGBRegressor(random_state=seed, n_jobs=1, **p["xgboost"]))])
    cat = CatBoostAreaRegressor(cat_features=tuple(CAT), random_seed=seed, thread_count=1, **p["catboost"])
    return StackingRegressor([("rf", rf), ("xgb", xgb), ("cat", cat)], final_estimator=RidgeCV(), cv=5, n_jobs=-1)


def regression_metrics(model, X_test, y_test):
    pred_log = model.predict(X_test)
    actual, pred = np.expm1(y_test), np.expm1(pred_log)
    return {
        "Test RMSE (log)": float(np.sqrt(mean_squared_error(y_test, pred_log))),
        "Test R2 (log)": float(r2_score(y_test, pred_log)),
        "Test MAE (Rs/perch)": float(mean_absolute_error(actual, pred)),
        "Test MAPE": float(mean_absolute_percentage_error(actual, pred)),
        "Test median APE": float(np.median(np.abs(pred - actual) / actual)),
    }


def train(df, p):
    seed = p["seed"]
    features, holdout, data, X_train, X_test, y_train, y_test = split(df, seed, p["test_size"])
    num = [c for c in features if c not in CAT]

    regressor = build_regressor(p, seed).fit(X_train, y_train)
    metrics = regression_metrics(regressor, X_test, y_test)
    metrics["stacking weights (rf, xgb, cat)"] = [round(float(w), 3) for w in regressor.final_estimator_.coef_]

    edges = np.expm1(y_train.quantile([1 / 3, 2 / 3])).values
    yc_train = np.digitize(np.expm1(y_train), edges)
    yc_test = np.digitize(np.expm1(y_test), edges)
    classifier = Pipeline([
        ("prep", _prep(seed, "multiclass")),
        ("model", RandomForestClassifier(random_state=seed, n_jobs=-1, **p["band_classifier"]))]).fit(X_train, yc_train)
    pc = classifier.predict(X_test)
    metrics["Band accuracy"] = float(accuracy_score(yc_test, pc))
    metrics["Band macro F1"] = float(f1_score(yc_test, pc, average="macro"))

    x_new = holdout[features]
    metrics["Holdout"] = {
        "area": holdout["Address"].iloc[0],
        "actual_rs_per_perch": float(holdout[TARGET_RAW].iloc[0]),
        "predicted_rs_per_perch": float(np.expm1(regressor.predict(x_new))[0]),
    }

    bundle = {
        "regressor": regressor,
        "regressor_name": "Stacking (RF + XGB + Cat)",
        "classifier": classifier,
        "classifier_name": "Random Forest classifier",
        "band_edges": edges,
        "band_labels": BANDS,
        "features": features,
        "land_types": [c for c in features if c.startswith("is_")],
        "area_defaults": data.groupby("Address")[num].median(),
        "test_metrics": {k: v for k, v in metrics.items() if k.startswith("Test")},
    }
    return bundle, metrics
