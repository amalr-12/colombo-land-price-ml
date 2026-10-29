# Colombo Land Price Estimator

Predicts the **asking price per perch** of land in the Colombo district, Sri Lanka, from its area, size and land
type. IT5514 Applied Machine Learning, final assignment.

**Live app:** https://colombo-land-price-ml-5jekcfuzve5ezgfgjjpqoh.streamlit.app

| | |
|---|---|
| Task | Regression (price per perch), plus Low / Medium / High price-band classification |
| Data | 4,494 Colombo land listings, Jul-Sep 2021 ([Kaggle](https://www.kaggle.com/datasets/ruchiraayeshmantha/land-prices-of-colombo-district)), 4,275 after cleaning |
| Final model | Stacking ensemble: Random Forest + XGBoost + CatBoost, RidgeCV meta-model |
| New algorithm | **NGBoost** (not covered in lectures): predicts a price distribution, used for the 80% price range |
| Test R² (log scale) | **0.866** |
| Median error | **11.8%** |
| Band accuracy | **90.5%** (Random Forest classifier) |
| Price ranges | NGBoost 80% ranges contain **81.9%** of actual test prices |

## Repository structure

```
01_data_preparation.ipynb   cleaning, preprocessing and EDA
02_modelling.ipynb          Ridge, RF, XGBoost, LightGBM, CatBoost, stacking: tuning, comparison, classification
03_unseen_areas.ipynb       generalisation to areas the model has never seen
04_explainability.ipynb     SHAP, LIME, CatBoost SHAP, partial dependence
05_ngboost.ipynb            NGBoost (new algorithm): tuning, accuracy, price-range calibration, SHAP
app.py                      Streamlit front-end: price, likely range, band, "Why this price?"
explain.py                  SHAP explanations and NGBoost price ranges for the app
catboost_wrapper.py         scikit-learn compatible CatBoost wrapper (needed to load the model)
ngboost_wrapper.py          scikit-learn compatible NGBoost wrapper (needed to load the range model)
land_price_model.pkl        deployed model bundle (stacking price model + band classifier)
ngboost_model.pkl           deployed NGBoost range model
src/                        retraining pipeline: prepare.py, train.py, promote.py
run_pipeline.py             runs the pipeline end to end
config/params.json          tuned hyperparameters and quality-gate thresholds
tests/                      data, model and app tests (pytest)
.github/workflows/          CI on every push; manual retraining workflow
Report.md / Report.docx     full write-up
figures/                    charts used in the report
```

## Running it

```bash
pip install -r requirements.txt          # app + tests
streamlit run app.py                     # front-end at http://localhost:8501

pip install -r requirements-dev.txt      # notebooks + pipeline
python -m pytest                         # 17 tests: data rules, model behaviour, explanations, ranges, app smoke test
python run_pipeline.py                   # retrain a candidate model into artifacts/
python run_pipeline.py --promote         # ...and replace the deployed model if it passes the quality gate
```

Python 3.12. Library versions are pinned exactly, because the pickled models must be loaded with the versions
they were trained with. pandas is 2.3.3 because NGBoost's dependency `lifelines` doesn't support pandas 3 yet.

## MLOps pipeline

```
raw Excel ──► src/prepare.py ──► clean data ──► src/train.py ──► candidate model + metrics.json
                                                                      │
                                         src/promote.py (quality gate)┘
                                   R² ≥ 0.85 and not worse than deployed?
                                     yes ──► replace land_price_model.pkl (old one backed up)
                                     no  ──► reject, deployed model unchanged
```

* **Reproducible training.** `config/params.json` holds the tuned hyperparameters from notebook 02, and the
  pipeline uses the same seed, hold-out listing and train/test split. A retrained model reproduces the
  notebook's results.
* **Quality gate.** A new model replaces the deployed one only if it meets the minimum accuracy and doesn't
  get worse.
* **Continuous integration** (`.github/workflows/ci.yml`). Every push runs the tests on Linux: data rules
  hold, the saved model loads with the pinned versions and predicts sensibly, and the app starts.
* **Retraining on demand** (`.github/workflows/retrain.yml`). Started from the Actions tab, it runs the
  pipeline in the cloud and uploads the candidate model and metrics for review.
* **Deployment.** Streamlit Community Cloud redeploys the app automatically when `main` changes.

## Model card

**Intended use.** A rough guide to the asking price of land in a known Colombo-district area, for buyers and
sellers comparing advertisements. It isn't a professional valuation.

**Training data.** Asking prices from online advertisements, posted 31 July to 21 September 2021, covering
160 areas. Cleaning fixed a per-acre price-scale error, removed 198 duplicates and 21 implausible prices
(IQR on the log price).

**Performance (test set, 855 listings).** R² 0.866 on log price, median absolute error 11.8%, MAE
Rs 554,620 per perch. On areas *not* seen in training, R² falls to about 0.69 (grouped cross-validation), so the
app only offers known areas.

**Uncertainty.** NGBoost gives an 80% price range for every prediction; 81.9% of actual test prices fall inside
it. Ranges typically span about 2x (asking prices are noisy) and are wider for areas with few listings.

**What drives predictions.** Location explains 87% of the model's decisions (SHAP), mainly distance from
Colombo Fort, the area itself and distance to grade-A schools. Land type explains 9%, land size 4%.

**Limitations.**
* Asking prices, not sale prices.
* 2021 data: absolute prices are out of date after the 2022 economic crisis; relative patterns are more
  likely to hold.
* Location features are recorded per area, not per plot, so two plots in the same area differ only by size and
  type.
* No information on road access, frontage, shape or flood risk.

## Data

*Land prices of Colombo district*, published on Kaggle by ruchiraayeshmantha:
https://www.kaggle.com/datasets/ruchiraayeshmantha/land-prices-of-colombo-district
