import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, r2_score
from sklearn.base import BaseEstimator, TransformerMixin


class BlendingClassifier():
    def __init__(self, clfs, n_class, folds=5, verbose=0):
        self.clfs = clfs
        self.kfold = KFold(folds)
        self.verbose = verbose
        self.n_class = n_class

    def __fit_one_fold(self, X, y):
        for clf in self.clfs:
            clf.fit(X, y)

    def __predict_one_fold(self, X):
        res = np.ones((X.shape[0], 1)) * (-1)
        for clf in self.clfs:
            res = np.column_stack((res, clf.predict_proba(X)))
        return np.array(res[:, 1:])

    def fit_transform_train(self, X, y):
        res = np.ones((X.shape[0], len(self.clfs) * self.n_class)) * (-1)
        X_train = X
        # k-fold for training set
        for (tr_idx, cv_idx) in self.kfold.split(X_train, y):
            X_tr, y_tr = X_train[tr_idx], y[tr_idx]
            X_cv, y_cv = X_train[cv_idx], y[cv_idx]
            self.__fit_one_fold(X_tr, y_tr)
            res[cv_idx, :] = self.__predict_one_fold(X_cv)
            if self.verbose > 0:
                print("Fold results (cv error):")
                # TODO - add different metrics
                for (idx, clf) in enumerate(self.clfs):
                    print(
                        "clf {:2d}: {:06.4f}".
                        format(idx, log_loss(y_cv, clf.predict_proba(X_cv))))
        return res

    def fit_transform_test(self, Xtr, ytr, Xts):
        # Xtr = Xtr.todense()
        # Xts = Xts.todense()
        self.__fit_one_fold(Xtr, ytr)
        return self.__predict_one_fold(Xts)


class BlendingRegressor():
    def __init__(self, rgrs=None, folds=5, verbose=0, scoring=r2_score):
        self.rgrs = rgrs
        self.kfold = KFold(folds)
        self.verbose = verbose
        self.hash = None
        self.target = None
        self.fitFullTrain = False
        self.scoring = scoring

    def fit_one_fold_(self, X, y):
        for rgr in self.rgrs:
            rgr.fit(X, y)

    def predict_one_fold_(self, X):
        res = np.ones((X.shape[0], 1)) * (-1)
        for rgr in self.rgrs:
            res = np.column_stack((res, rgr.predict(X)))
        return np.array(res[:, 1:])

    def fit_transform_train_(self, X, y):
        res = np.ones((X.shape[0], len(self.rgrs))) * (-1)
        X_train = X
        fold = 0
        # k-fold for training set
        for (tr_idx, cv_idx) in self.kfold.split(X_train, y):
            X_tr, y_tr = X_train[tr_idx], y[tr_idx]
            X_cv, y_cv = X_train[cv_idx], y[cv_idx]
            self.fit_one_fold_(X_tr, y_tr)
            res[cv_idx, :] = self.predict_one_fold_(X_cv)
            if self.verbose > 0:
                print("Fold {:1d} CV results:".format(fold))
                fold += 1
                # TODO - add different metrics
                for (idx, rgr) in enumerate(self.rgrs):
                    print(
                        "rgr {:2d}: {:06.4f}".
                        format(idx, r2_score(y_cv, rgr.predict(X_cv))))
        return res

    def fit_transform_test_(self, Xtr, ytr, Xts):
        self.fit_one_fold_(Xtr, ytr)
        return self.predict_one_fold_(Xts)

    def fit(self, X, y):
        self.hash = hash(X.data.tobytes())
        self.target = y
        self.fit_one_fold_(X, y)
        self.fitFullTrain = True

    def transform(self, X):
        if hash(X.data.tobytes()) == self.hash:
            return self.fit_transform_train_(X, self.target)
        else:
            return self.predict_one_fold_(X)

    def fit_transform(self, X, y):
        self.hash = hash(X.data.tobytes())
        self.target = y
        out = self.fit_transform_train_(X, y)
        self.fit_one_fold_(X, y)
        self.fitFullTrain = True
        return out


class PipeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, fields):
        self.fields = fields

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.fields]


class PipeShape(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("Pipeline output shape: ", X.shape)
        return X
