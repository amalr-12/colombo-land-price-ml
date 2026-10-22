"""Colombo land price estimator.

Run from the Assignment_2 folder:
    streamlit run app.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # so the pickle can find catboost_wrapper.py
MODEL_PATH = HERE / "land_price_model.pkl"


@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def label(col):
    """Turn a column name into something readable, e.g. min_dist_nearest_railway -> Distance to railway (km)."""
    name = col.replace("_", " ").replace("within2km", "within 2km").replace("within 2km", "(within 2 km)")
    if col.startswith("min_dist"):
        name = name.replace("min dist nearest ", "").replace("min dist ", "")
        return f"Distance to {name} (km)"
    if col.startswith("count"):
        return "No. of " + name.replace("count ", "")
    if col == "Distance from fort":
        return "Distance from Colombo Fort (km)"
    return name


bundle = load_bundle()
areas = bundle["area_defaults"]
location_cols = [c for c in areas.columns if c != "Land_size(Perches)" and not c.startswith("is_")]

st.set_page_config(page_title="Colombo Land Price Estimator", page_icon="🏡", layout="centered")
st.title("Colombo Land Price Estimator")
st.caption("Predicts the asking price per perch for land in the Colombo district, "
           "trained on 2021 listings.")

with st.sidebar:
    st.subheader("Model")
    st.write(f"**Price:** {bundle['regressor_name']}")
    st.write(f"**Price band:** {bundle['classifier_name']}")
    m = bundle["test_metrics"]
    st.write(f"Test R² (log scale): **{m['Test R2 (log)']:.3f}**")
    st.write(f"Median error: **{m['Test median APE']:.0%}**")
    st.caption("Trained on asking prices from Jul-Sep 2021. Treat results as a rough guide, "
               "not a valuation.")

area = st.selectbox("Area", sorted(areas.index), index=sorted(areas.index).index("nugegoda")
                    if "nugegoda" in areas.index else 0)
c1, c2 = st.columns(2)
size = c1.number_input("Land size (perches)", min_value=1.0, max_value=600.0, value=10.0, step=0.5)
types = c2.multiselect("Land type", [t.removeprefix("is_") for t in bundle["land_types"]],
                       default=["Residential"])

# In the dataset every listing in an area has the same location values (they were calculated per area,
# not per plot), so they're shown for information rather than edited.
location = areas.loc[area, location_cols].to_dict()
with st.expander(f"Location details used for {area}"):
    st.caption("These come from the area, not the individual plot, which is how the training data was built.")
    st.dataframe(
        pd.DataFrame({"Feature": [label(c) for c in location_cols],
                      "Value": [f"{location[c]:.0f}" if c.startswith("count") else f"{location[c]:.2f}"
                                for c in location_cols]}),
        hide_index=True, width="stretch")

if st.button("Estimate price", type="primary", width="stretch"):
    if not types:
        st.warning("Pick at least one land type.")
        st.stop()

    row = {"Address": area, "Land_size(Perches)": size, **location}
    for t in bundle["land_types"]:
        row[t] = int(t.removeprefix("is_") in types)
    x = pd.DataFrame([row])[bundle["features"]]

    per_perch = float(np.expm1(bundle["regressor"].predict(x))[0])
    band = bundle["band_labels"][int(bundle["classifier"].predict(x)[0])]
    lo, hi = bundle["band_edges"]

    a, b, c = st.columns(3)
    a.metric("Price per perch", f"Rs {per_perch:,.0f}")
    b.metric("Estimated total", f"Rs {per_perch * size / 1e6:,.2f} M")
    c.metric("Price band", band)
    st.caption(f"Bands: Low < Rs {lo:,.0f} ≤ Medium < Rs {hi:,.0f} ≤ High (per perch). "
               "The band comes from a separate classifier, so it can occasionally disagree with the price.")
