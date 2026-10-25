"""Colombo land price estimator.

Run from the Assignment_2 folder:
    streamlit run app.py
"""
import pickle
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # so the pickle can find catboost_wrapper.py
MODEL_PATH = HERE / "land_price_model.pkl"

from explain import PriceExplainer, pretty_name as label  # noqa: E402


@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_explainer():
    return PriceExplainer(load_bundle())


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

    # ---- Why this price? (SHAP explanation of the predicted price) ----
    st.subheader("Why this price?")
    with st.spinner("Explaining the prediction..."):
        e = load_explainer().explain(x)
    pct = e["grouped_pct"]

    def effect(v):
        return f"raises the price by about {v:.0f}%" if v >= 0 else f"lowers the price by about {-v:.0f}%"

    st.write(
        f"Compared with a typical plot (about **Rs {e['baseline_price']:,.0f}** per perch), "
        f"this plot's **area and location** {effect(pct['Area & location'])}, its **land type** "
        f"{effect(pct['Land type'])}, and its **size** {effect(pct['Land size'])}.")

    chart_df = pd.DataFrame({"Factor": pct.index, "Effect on price (%)": pct.values.round(1)})
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Effect on price (%):Q"),
        y=alt.Y("Factor:N", sort=None, title=None),
        color=alt.condition("datum['Effect on price (%)'] >= 0", alt.value("#1F4E79"), alt.value("#B45309")),
        tooltip=["Factor", "Effect on price (%)"],
    ).properties(height=150)
    st.altair_chart(chart, width="stretch")

    with st.expander("Top individual factors"):
        top = e["contributions"].reindex(e["contributions"]["shap"].abs().sort_values(ascending=False).index).head(6)
        st.dataframe(pd.DataFrame({
            "Factor": top["name"],
            "Value for this plot": [str(v) if isinstance(v, str) else f"{v:,.2f}".rstrip("0").rstrip(".") for v in top["value"]],
            "Effect on price": [f"{(np.exp(c) - 1) * 100:+.0f}%" for c in top["shap"]],
        }), hide_index=True, width="stretch")
        st.caption("SHAP values: how much each factor moves this prediction away from a typical plot. "
                   "Location features are fixed for each area, so read them together as 'location'.")
