"""scikit-learn compatible wrapper around CatBoostRegressor.

CatBoost's own estimator can't be cloned by recent scikit-learn versions when `cat_features` is set,
which breaks RandomizedSearchCV and StackingRegressor. This wrapper stores plain parameters and only
builds the CatBoost model inside fit(). It lives in its own module so the pickled model can be loaded
by the app.
"""
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class CatBoostAreaRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, cat_features=("Address",), iterations=1000, depth=6, learning_rate=0.06,
                 l2_leaf_reg=3, random_seed=42, thread_count=1):
        self.cat_features = cat_features
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            cat_features=list(self.cat_features), iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed, thread_count=self.thread_count, verbose=0)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
