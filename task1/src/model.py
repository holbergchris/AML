# Training, optimizing and fitting models
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.base import BaseEstimator, clone
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense


class LGBM(BaseEstimator):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.model = LGBMRegressor(max_depth=-1, silent=True, **self.__dict__)

    def fit(self, X, y):
        params = dict()
        self.model.fit(X, y, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {k: self.__dict__[k] for k in self.__dict__.keys() if k != 'model'}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def tune(self, X, y, score='r2', iter=100, save=False):
        param_test = {'n_estimators': sp_randint(100, 1000),
                      'num_leaves': sp_randint(6, 60),
                      'min_child_samples': sp_randint(100, 500),
                      'min_child_weight': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                      'subsample': sp_uniform(loc=0.2, scale=0.8),
                      'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                      'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                      'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
        lgbm = LGBMRegressor(max_depth=-1, silent=True, metric='None')
        lgbm_cv = RandomizedSearchCV(estimator=lgbm, param_distributions=param_test, n_iter=iter,
                                     scoring='r2', cv=5, verbose=1, n_jobs=-1)
        lgbm_cv.fit(X, y, verbose=0)

        if save:
            pd.DataFrame(lgbm_cv.best_params_, index=[0]).to_csv("data/lgbm_par.csv")

        for key in lgbm_cv.best_params_:
            self.__dict__[key] = lgbm_cv.best_params_[key]

        print("Best score:", lgbm_cv.best_score_, "obtained for\n", lgbm_cv.best_params_)


class XGB(BaseEstimator):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.model = XGBRegressor(silent=True, **self.__dict__)

    def fit(self, X, y):
        params = dict()
        self.model.fit(X, y, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {k: self.__dict__[k] for k in self.__dict__.keys() if k != 'model'}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def tune(self, X, y, score='r2', iter=100, save=False):
        param_test = {'n_estimators': sp_randint(100, 1000),
                      'max_depth': sp_randint(4, 30),
                      'num_leaves': sp_randint(6, 60),
                      'min_child_samples': sp_randint(100, 500),
                      'min_child_weight': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                      'subsample': sp_uniform(loc=0.2, scale=0.8),
                      'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                      'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                      'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
        xgb = XGBRegressor(silent=True, metric='None')
        xgb_cv = RandomizedSearchCV(estimator=xgb, param_distributions=param_test, n_iter=iter,
                                    scoring='r2', cv=5, verbose=1, n_jobs=-1)
        xgb_cv.fit(X, y, verbose=0)

        if save:
            pd.DataFrame(xgb_cv.best_params_, index=[0]).to_csv("data/xgb_par.csv")

        for key in xgb_cv.best_params_:
            self.__dict__[key] = xgb_cv.best_params_[key]

        print("Best score:", xgb_cv.best_score_, "obtained for\n", xgb_cv.best_params_)



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

    def tune(self, X, y, scoring='r2', iter=100, save=False):
        par = {'bootstrap': [True, False],
               'max_depth': sp_randint(10, 80),
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': sp_randint(100, 800)}
        rf = RandomForestRegressor()
        rf_cv = RandomizedSearchCV(estimator=rf, param_distributions=par, n_iter=iter,
                                   scoring=scoring, cv=5, verbose=1, n_jobs=-1)
        rf_cv.fit(X, y)

        if save:
            pd.DataFrame(rf_cv.best_params_, index=[0]).to_csv("data/rf_par.csv")

        for key in rf_cv.best_params_:
            self.__dict__[key] = rf_cv.best_params_[key]

        print("Best score:", rf_cv.best_score_, "obtained for\n", rf_cv.best_params_)


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

    def tune(self, X, y, scoring='r2', save=False):
        par = {"alpha": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]}
        lasso = Lasso(max_iter=5000)
        lasso_cv = GridSearchCV(lasso, par, scoring=scoring, cv=5, verbose=1, n_jobs=-1)
        lasso_cv.fit(X, y)

        if save:
            pd.DataFrame(lasso_cv.best_params_, index=[0]).to_csv("data/lasso_par.csv")

        for key in lasso_cv.best_params_:
            self.__dict__[key] = lasso_cv.best_params_[key]

        print("Best score:", lasso_cv.best_score_, "obtained for\n", lasso_cv.best_params_)

class NN(BaseEstimator):
    def __init__(self, input=128, hidden=256, num_hidden=3):
        self.input = input
        self.hidden = hidden
        self.num_hidden = num_hidden
        self.model = Sequential()

    def fit(self, X, y):
        self.model.add(Dense(self.input, kernel_initializer='normal', input_dim=X.shape[1], activation='relu'))
        for i in range(self.num_hidden):
            self.model.add(Dense(self.hidden, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
        checkpoint = ModelCheckpoint('data/.nn_weights.hdf5', monitor='val_loss', verbose=0, save_best_only=True,
                                     mode='auto')
        callbacks_list = [checkpoint]
        self.model.fit(X, y, epochs=300, batch_size=64, validation_split=.2, verbose=0, callbacks=callbacks_list)
        self.model.load_weights('data/.nn_weights.hdf5')
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
        return self

    def predict(self, X):
        return self.model.predict(X)[:, 0]

    def get_params(self, deep=True):
        return {k: self.__dict__[k] for k in self.__dict__.keys() if k != 'model'}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def tune(self, X, y, scoring='r2', save=False):
        par = {
            "input": [64, 128],
            "hidden": [128, 256],
            "num_hidden": [2, 3, 4]
        }
        nn_cv = GridSearchCV(self, par, scoring=scoring, cv=5, verbose=1, n_jobs=1)
        nn_cv.fit(X, y)

        if save:
            pd.DataFrame(nn_cv.best_params_, index=[0]).to_csv("data/nn_par.csv")

        for key in nn_cv.best_params_:
            self.__dict__[key] = nn_cv.best_params_[key]

        print("Best score:", nn_cv.best_score_, "obtained for\n", nn_cv.best_params_)

class AveragingModels(BaseEstimator):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


def r2_cv(model, X, y):
    kf = KFold(5, shuffle=True).get_n_splits(X, y)
    return cross_val_score(model, X, y, scoring='r2', cv=kf)
