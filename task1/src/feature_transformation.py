# Class and methods for transforming feature space and learning new features
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.feature_selection import RFECV

class Transformer:
    def __init__(self, X, X_test, y):
        self.X = X.copy()
        self.X_ = X_test.copy()
        self.y = y.copy()

    def reduceDimension(self, threshold=.9):
        pca = PCA()
        self.X.iloc[:,:] = pca.fit_transform(self.X)
        cum_variance = np.cumsum(pca.explained_variance_ratio_)
        self.X = self.X.iloc[:, cum_variance < threshold]
        self.X_.iloc[:,:] = pca.transform(self.X_)
        self.X_ = self.X_.iloc[:, cum_variance < threshold]
        return self.X, self.X_

    def selectFeatures(self, method="RFECV", save=False):
        if method == "RFECV":
            rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=0)
            rfecv = RFECV(estimator=rf, step=20, cv=5, scoring='r2', verbose=1)
            rfecv.fit(self.X, self.y)
            self.X = self.X.iloc[:, rfecv.support_]
            self.X_ = self.X_.iloc[:, rfecv.support_]
            if save:
                pd.DataFrame({"features": rfecv.support_}).to_csv("data/best_features.csv")
            return self.X, self.X_

        if method == "SelectFrom":
            ranks = {}
            colnames = self.X.columns
            lr = LinearRegression(normalize=True)
            lr.fit(self.X, self.y)
            ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)
            ridge = Ridge(alpha=10)
            ridge.fit(self.X, self.y)
            ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)
            lasso = Lasso(alpha=.3)
            lasso.fit(self.X, self.y)
            ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)
            rf = RandomForestRegressor(n_jobs=-1, n_estimators=100, verbose=1)
            rf.fit(self.X, self.y)
            ranks["RF"] = ranking(rf.feature_importances_, colnames)

            r = {}
            for name in colnames:
                r[name] = np.round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
            mean_ranks = pd.DataFrame(list(r.items()), columns=['Feature', 'Mean Ranking'])
            mean_ranks = mean_ranks.sort_values('Mean Ranking', ascending=False)
            return mean_ranks.loc[:20, "Feature"]


def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))