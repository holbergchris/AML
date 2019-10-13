# Class and methods for preprocessing data
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer

class Preprocessor:
  def __init__(self, X, X_test, y):
    self.X = X.copy()
    self.X_ = X_test.copy()
    self.y = y.copy()

  def impute(self, fill="random"):
    for name in self.X.columns:
      col = self.X[name]
      col_ = self.X_[name]
      missing = col.isna()
      missing_ = col_.isna()
      n = sum(missing)
      n_ = sum(missing_)
      if fill == "random":
        imp = np.random.choice(col[~missing], size=n)
        imp_ = np.random.choice(col[~missing], size=n_)
      if fill == "mean":
        imp = np.full((n,), np.mean(col[~missing]))
        imp_ = np.full((n_,), np.mean(col[~missing]))
      if fill == "median":
        imp = np.full((n,), np.median(col[~missing]))
        imp_ = np.full((n_,), np.median(col[~missing]))
      self.X.loc[missing, name] = imp
      self.X_.loc[missing_, name] = imp_
    return self.X, self.X_

  def processOutliers(self, method="if", contamination=.1, remove=True):
    if method == "if":
      train = self.X.merge(self.y, by='id')
      clf = IsolationForest(n_estimators=100, max_samples=.6, max_features=.8,
                            n_jobs=-1, bootstrap=True, contamination=contamination)
      clf.fit(train)
      outliers = clf.predict(self.X) < 0
      if remove:
        self.X = self.X.iloc[~outliers, :]
        self.y = self.y.iloc[~outliers, :]

    return self.X, self.X_, self.y

  def scaleData(self, scaler="normal"):
    if scaler == "normal":
      scaler = StandardScaler()
    elif scaler == "robust":
      scaler = RobustScaler()
    self.X.iloc[:, :] = scaler.fit_transform(self.X)
    self.X_.iloc[:, :] = scaler.transform(self.X_)
    return self.X, self.X_

  def transformResponse(self, method='box-cox'):
    if method == 'box-cox':
      transformer = PowerTransformer()
      self.y = transformer.fit_transform(self.y)
    return self.y, transformer

