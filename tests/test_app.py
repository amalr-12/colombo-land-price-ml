"""The Streamlit app starts and produces a prediction (smoke test)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_and_predicts():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    at.selectbox[0].set_value("kollupitiya")
    at.button[0].click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels == ["Price per perch", "Estimated total", "Price band"]
    assert at.metric[2].value == "High"
