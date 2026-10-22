"""The deployed model loads and behaves sensibly (report Sections 5 and 7)."""
import numpy as np
import pandas as pd

SEED = 42


def _row(bundle, area, size=10.0, types=("Residential",)):
    row = {"Address": area, "Land_size(Perches)": size, **bundle["area_defaults"].loc[area].to_dict()}
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


def test_reported_accuracy_meets_quality_bar(bundle):
    assert bundle["test_metrics"]["Test R2 (log)"] >= 0.85
