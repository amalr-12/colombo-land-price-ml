"""Explains one prediction of the land-price model with SHAP ("Why this price?" in the app).

SHAP splits the predicted log price into a baseline (a typical plot) plus one contribution per feature.
Because the model predicts log price, a contribution c multiplies the price by exp(c), so contributions are
reported as percentage effects on the price.

The final model is a stacking ensemble, so the model-agnostic PermutationExplainer is used (as in
04_explainability.ipynb). Address is text, so areas are mapped to integer codes for SHAP and back to names
before calling the model.
"""
import numpy as np
import pandas as pd
import shap
from scipy.stats import norm

LOCATION_PREFIXES = ("min_dist", "count", "Distance")


PLACES = {  # what each column's place token means
    "govtschools_a": "grade-A govt. school", "govtschools_A": "grade-A govt. schools",
    "govtschools_b": "grade-B govt. school", "govtschools_B": "grade-B govt. schools",
    "semigovtschools": "semi-govt. school", "intlschools": "international school", "uni": "university",
    "express": "expressway entrance", "railway": "railway station", "bank": "bank", "banks": "banks",
    "FinanceCompany": "finance company", "FinanceCompanies": "finance companies",
    "Govt_Hospital": "govt. hospital", "Govt_Hospitals": "govt. hospitals",
    "Pvt_Hospital": "private hospital", "Pvt_Med_center": "private medical centre",
    "Pvt_Med_Centers": "private medical centres", "Supermarket": "supermarket", "Supermarkets": "supermarkets",
    "Fuel_station": "fuel station", "Fuel_Stations": "fuel stations",
}
COUNT_PLURALS = {"semigovtschools": "semi-govt. schools", "intlschools": "international schools",
                 "uni": "universities", "Pvt_Hospital": "private hospitals"}


def pretty_name(col):
    """Turn a column name into something readable, e.g. min_dist_govtschools_a -> Distance to grade-A govt. school (km)."""
    if col == "Address":
        return "Area"
    if col == "Land_size(Perches)":
        return "Land size (perches)"
    if col == "Distance from fort":
        return "Distance from Colombo Fort (km)"
    if col.startswith("is_"):
        return "Land type: " + col.removeprefix("is_")
    if col.startswith("min_dist"):
        token = col.removeprefix("min_dist_").removeprefix("nearest_")
        return f"Distance to nearest {PLACES.get(token, token.replace('_', ' '))} (km)"
    if col.startswith("count"):
        token = col.removeprefix("count_")
        within = "_within_2km" in token or "_within2km" in token
        token = token.replace("_within_2km", "").replace("_within2km", "")
        place = COUNT_PLURALS.get(token) or PLACES.get(token, token.replace("_", " "))
        return f"No. of {place}" + (" within 2 km" if within else " nearby")
    return col.replace("_", " ")


def price_range(ngb_bundle, x, level=0.8):
    """NGBoost's predicted price per perch and its central `level` range (default 80%).

    NGBoost predicts a Normal distribution for log1p(price), i.e. a log-normal distribution for the price.
    """
    pipe = ngb_bundle["model"]
    dist = pipe[-1].pred_dist(pipe[:-1].transform(x[ngb_bundle["features"]]))
    mu, sd = float(dist.loc[0]), float(dist.scale[0])
    z = norm.ppf(0.5 + level / 2)
    return float(np.expm1(mu)), float(np.expm1(mu - z * sd)), float(np.expm1(mu + z * sd))


def feature_group(col):
    if col == "Address" or col.startswith(LOCATION_PREFIXES):
        return "Area & location"
    if col.startswith("is_"):
        return "Land type"
    return "Land size"


class PriceExplainer:
    def __init__(self, bundle, n_background=40, seed=42):
        self.model = bundle["regressor"]
        self.features = bundle["features"]
        defaults = bundle["area_defaults"]
        self.areas = sorted(defaults.index)
        self.code = {a: i for i, a in enumerate(self.areas)}

        # Background = a typical plot (median size and type) in 40 randomly chosen areas
        rng = np.random.default_rng(seed)
        picked = rng.choice(self.areas, size=min(n_background, len(self.areas)), replace=False)
        bg = defaults.loc[picked].reset_index(drop=True)
        bg["Address"] = picked
        self.background = self._to_numeric(bg[self.features])

        masker = shap.maskers.Independent(self.background.values, max_samples=len(self.background))
        self.explainer = shap.PermutationExplainer(self._predict_numeric, masker,
                                                   feature_names=self.features, seed=seed)

    def _to_numeric(self, X):
        X = X.copy()
        X["Address"] = X["Address"].map(self.code)
        return X.astype(float)

    def _predict_numeric(self, A):
        X = pd.DataFrame(A, columns=self.features)
        X["Address"] = [self.areas[int(round(c))] for c in X["Address"]]
        return self.model.predict(X)

    def explain(self, x):
        """x: one-row DataFrame in the model's feature order. Returns baseline, contributions and groups."""
        sv = self.explainer(self._to_numeric(x).values, silent=True)
        values, base = sv.values[0], float(sv.base_values[0])
        contrib = pd.DataFrame({
            "feature": self.features,
            "name": [pretty_name(f) for f in self.features],
            "group": [feature_group(f) for f in self.features],
            "value": x.iloc[0].values,
            "shap": values,
        })
        grouped = contrib.groupby("group")["shap"].sum().reindex(["Area & location", "Land type", "Land size"])
        return {
            "baseline_log": base,
            "prediction_log": base + float(values.sum()),
            "baseline_price": float(np.expm1(base)),
            "contributions": contrib,
            "grouped_log": grouped,
            "grouped_pct": (np.exp(grouped) - 1) * 100,  # % effect on price of each group
        }
