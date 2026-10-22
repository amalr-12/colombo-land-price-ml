"""The cleaned modelling table follows the preprocessing rules (report Section 3)."""
import numpy as np


def test_row_count(clean_df):
    assert len(clean_df) == 4275


def test_no_duplicates(clean_df):
    assert not clean_df.duplicated().any()


def test_leaky_and_helper_columns_removed(clean_df):
    for col in ["Mentioned Price(Rs)", "Address_ID", "Price_Scale", "Land_type"]:
        assert col not in clean_df.columns


def test_land_type_split_into_four_flags(clean_df):
    flags = sorted(c for c in clean_df.columns if c.startswith("is_"))
    assert flags == ["is_Agricultural", "is_Commercial", "is_Other", "is_Residential"]
    assert (clean_df[flags].sum(axis=1) >= 1).all()  # every plot has at least one type


def test_log_target_matches_price(clean_df):
    assert np.allclose(clean_df["log_price_per_perch"], np.log1p(clean_df["Price per Perch"]))


def test_no_missing_values(clean_df):
    assert clean_df.isna().sum().sum() == 0


def test_price_range_is_plausible(clean_df):
    # After the per-acre fix and log-IQR outlier removal, no implausible prices remain
    assert clean_df["Price per Perch"].min() > 10_000
    assert clean_df["Price per Perch"].max() < 100_000_000
