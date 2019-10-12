# Training, optimizing and fitting models
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

class Param:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def optimizeLasso(self, scoring='r2', save=False):
        X = self.X
        y = self.y.y.values
        lasso_ = Lasso(max_iter=5000)
        par = {"alpha": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]}
        lasso_cv = GridSearchCV(lasso_, par, scoring=scoring, cv=5, verbose=1, n_jobs=-1)
        lasso_cv.fit(X, y)
        if save:
            par_df = pd.DataFrame(lasso_cv.best_params_, index=[0]).to_csv("data/lasso_par.csv")
        return lasso_cv.best_params_, lasso_cv.best_score_

    def optimizeRF(self, scoring='r2', iter=100, save=False):
        X = self.X
        y = self.y.y.values
        rf = RandomForestRegressor()
        par = {'bootstrap': [True, False],
               'max_depth': sp_randint(10, 80),
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': sp_randint(100, 800)}
        rf_cv = RandomizedSearchCV(estimator=rf, param_distributions=par, n_iter=iter,
                                   scoring=scoring, cv=5, verbose=1, n_jobs=-1)
        rf_cv.fit(X, y)
        if save:
            pd.DataFrame(rf_cv.best_params_, index=[0]).to_csv("data/rf_par.csv")
        return rf_cv.best_params_, rf_cv.best_score_

    def optimizeLGB(self, score='r2', iter=100, save=False):
        X_train, X_test, y_train, y_test  = train_test_split(self.X, self.y.y, test_size=.2)
        fit_params = {"early_stopping_rounds": 30,
                      "eval_metric": 'mse',
                      "eval_set": [(X_test, y_test)],
                      'eval_names': ['valid'],
                      "verbose": 0}
        param_test = {'num_leaves': sp_randint(6, 50),
                      'min_child_samples': sp_randint(100, 500),
                      'min_child_weight': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                      'subsample': sp_uniform(loc=0.2, scale=0.8),
                      'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                      'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                      'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
        lgbm = LGBMRegressor(max_depth=-1, silent=True, metric='None',
                                 n_estimators=5000)
        lgbm_cv = RandomizedSearchCV(estimator=lgbm, param_distributions=param_test, n_iter=iter,
                                     scoring='r2', cv=5, verbose=1, n_jobs=-1)
        lgbm_cv.fit(X_train, y_train, **fit_params)
        if save:
            pd.DataFrame(lgbm_cv.best_params_, index=[0]).to_csv("data/lgbm_par.csv")
        return lgbm_cv.best_params_, lgbm_cv.best_score_

class LGBM(BaseEstimator):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.model = LGBMRegressor(max_depth=-1, silent=True, n_estimators=5000, **self.__dict__)

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        fit_params = {"early_stopping_rounds": 30,
                  "eval_metric": 'mse',
                  "eval_set": [(X_test, y_test)],
                  'eval_names': ['valid'],
                  "verbose": 0}
        params = dict()
        self.model.fit(X_train, y_train, **fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {k: self.__dict__[k] for k in self.__dict__.keys() if k != 'model'}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

class RF(BaseEstimator):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for key, value in self.__dict__.items():
            self.__dict__[key] = value
        self.model = RandomForestRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {k: self.__dict__[k] for k in self.__dict__.keys() if k != 'model'}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

class LassoRegressor(BaseEstimator):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for key, value in self.__dict__.items():
            self.__dict__[key] = value
        self.model = Lasso(**self.__dict__)

    def fit(self, X, y):
        self.model.fit(X,y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {k: self.__dict__[k] for k in self.__dict__.keys() if k != 'model'}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

def r2_cv(model, X, y):
    kf = KFold(5, shuffle=True).get_n_splits(X, y)
    return cross_val_score(model, X, y, scoring='r2', cv=kf)