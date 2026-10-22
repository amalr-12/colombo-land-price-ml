import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # catboost_wrapper.py and src/
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def clean_df():
    return pd.read_csv(ROOT / "land_prices_clean.csv")


@pytest.fixture(scope="session")
def bundle():
    with open(ROOT / "land_price_model.pkl", "rb") as f:
        return pickle.load(f)
