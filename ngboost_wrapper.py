"""scikit-learn compatible NGBoost regressor.

Recent scikit-learn versions check that a Pipeline's last step is fitted before predicting. NGBRegressor
stores its fitted trees in `base_models` (no trailing underscore), so the check fails and every
cross-validation score comes back as NaN. This subclass tells scikit-learn when the model is fitted.
It lives in its own module so the pickled model can be loaded by the app.
"""
from ngboost import NGBRegressor


class NGBoostRegressor(NGBRegressor):
    def __sklearn_is_fitted__(self):
        return bool(getattr(self, "base_models", None))
