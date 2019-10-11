# Class and methods for transforming feature space and learning new features
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class Transformer:
    def __init__(self, X, X_test):
        self.X = X.copy()
        self.X_ = X_test.copy()

    def reduceDimension(self, threshold=.9):
        pca = PCA()
        self.X.iloc[:,:] = pca.fit_transform(self.X)
        cum_variance = np.cumsum(pca.explained_variance_ratio_)
        self.X = self.X.iloc[:, cum_variance < threshold]
        self.X_.iloc[:,:] = pca.transform(self.X_)
        self.X_ = self.X_.iloc[:, cum_variance < threshold]
        return self.X, self.X_

