# Class and methods for preprocessing data
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class Preprocessor:
  def __init__(self, X, X_test):
    self.X = X
    self.X_ = X_test

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

  def processOutliers(self, method="if", remove=True):
    if method == "if":
      clf = IsolationForest(n_estimators=300, max_samples=1000)
      clf.fit(self.X)
      outliers = clf.predict(self.X) < 0
      if remove:
        self.X = self.X.iloc[~outliers, :]
    return self.X, self.X_

  def scaleData(self, center=True, normal=True):
    if center & normal:
      scaler = StandardScaler()
      self.X.iloc[:,:] = scaler.fit_transform(self.X)
      self.X_.iloc[:,:] = scaler.transform(self.X_)
    return self.X, self.X_
