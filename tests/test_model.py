"""The deployed model loads and behaves sensibly (report Sections 5 and 7)."""
import numpy as np
import pandas as pd

SEED = 42


def _row(bundle, area, size=10.0, types=("Residential",)):
    # area defaults first, so the requested size isn't overwritten by the area's median size
    row = {**bundle["area_defaults"].loc[area].to_dict(), "Address": area, "Land_size(Perches)": size}
    for t in bundle["land_types"]:
        row[t] = int(t.removeprefix("is_") in types)
    return pd.DataFrame([row])[bundle["features"]]


def _price(bundle, x):
    return float(np.expm1(bundle["regressor"].predict(x))[0])


def test_bundle_has_expected_parts(bundle):
    for key in ["regressor", "classifier", "features", "band_edges", "band_labels", "area_defaults", "test_metrics"]:
        assert key in bundle
    assert len(bundle["features"]) == 33
    assert len(bundle["area_defaults"]) == 160


def test_held_out_listing_reproduces_notebook(bundle, clean_df):
    holdout = clean_df.sample(1, random_state=SEED)  # same hold-out row as 02_modelling.ipynb
    assert holdout["Address"].iloc[0] == "dampe"
    pred = _price(bundle, holdout[bundle["features"]])
    assert abs(pred - 332_620) / 332_620 < 0.01


def test_central_colombo_costs_more_than_suburbs(bundle):
    assert _price(bundle, _row(bundle, "kollupitiya")) > 10 * _price(bundle, _row(bundle, "kaduwela"))


def test_predictions_are_finite_and_positive(bundle):
    x = pd.concat([_row(bundle, a) for a in bundle["area_defaults"].index])
    pred = np.expm1(bundle["regressor"].predict(x))
    assert np.isfinite(pred).all() and (pred > 0).all()


def test_band_classifier_returns_valid_bands(bundle):
    x = pd.concat([_row(bundle, a) for a in ["kollupitiya", "nugegoda", "kaduwela"]])
    bands = bundle["classifier"].predict(x)
    assert set(bands) <= {0, 1, 2}
    assert bands[0] == 2 and bands[2] == 0  # central = High, suburban = Low


def test_explanation_adds_up_to_prediction(bundle):
    from explain import PriceExplainer
    x = _row(bundle, "battaramulla")
    e = PriceExplainer(bundle).explain(x)
    # SHAP: baseline + contributions = the model's own (log) prediction
    assert np.isclose(e["prediction_log"], bundle["regressor"].predict(x)[0], atol=1e-6)
    assert set(e["grouped_pct"].index) == {"Area & location", "Land type", "Land size"}
    assert e["grouped_pct"]["Area & location"] > 0  # Battaramulla is pricier than a typical plot


def test_ngboost_range_is_ordered_and_contains_its_prediction(bundle, ngb):
    from explain import price_range
    for area in ["battaramulla", "kaduwela", "kollupitiya"]:
        median, low, high = price_range(ngb, _row(bundle, area))
        assert 0 < low < median < high


def test_ngboost_ranges_are_calibrated_on_test_set(clean_df, ngb):
    from scipy.stats import norm
    from sklearn.model_selection import train_test_split
    import pandas as pd
    holdout = clean_df.sample(1, random_state=SEED)
    data = clean_df.drop(holdout.index).reset_index(drop=True)
    strata = pd.qcut(data["log_price_per_perch"], 3, labels=False)
    _, X_test, _, y_test = train_test_split(data[ngb["features"]], data["log_price_per_perch"],
                                            test_size=0.2, random_state=SEED, stratify=strata)
    pipe = ngb["model"]
    dist = pipe[-1].pred_dist(pipe[:-1].transform(X_test))
    inside = np.abs(y_test.values - dist.loc) <= norm.ppf(0.9) * dist.scale
    assert 0.70 <= inside.mean() <= 0.90  # an 80% range should hold about 80% of actual prices


def test_reported_accuracy_meets_quality_bar(bundle):
    assert bundle["test_metrics"]["Test R2 (log)"] >= 0.85
