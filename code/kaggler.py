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


class BlendingRegressorTransformer():
    def __init__(self, base_models=None, n_splits=5,
                 verbose=0, scoring=r2_score):
        self.base_models = [[m for m in base_models] for s in n_splits]
        self.n_splits = n_splits
        self.folds = None
        self.verbose = verbose
        self.hash = None
        self.target = None
        self.isTrainFit = False
        self.scoring = scoring

    def fit_one_fold_(self, X, y):
        for rgr in self.base_models:
            rgr.fit(X, y)

    def predict_one_fold_(self, X):
        res = np.zeros((X.shape[0], 1))
        for rgr in self.base_models:
            res = np.column_stack((res, rgr.predict(X)))
        return np.array(res[:, 1:])

    def fit(self, X, y):
        self.hash = hash(X.data.tobytes())
        self.isTrainFit = True
        self.folds = list(KFold(n_splits=self.n_splits).split(X, y))

        for i, rgr in enumerate(self.base_models):
            for j, (train_idx, test_idx) in enumerate(self.folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]

                self.base_models[i, j].fit(X_train, y_train)
                y_pred = self.base_models[i, j].predict(X_holdout)[:]

                print ("Model %d fold %d score %f" %
                       (i, j, r2_score(y_holdout, y_pred)))
        return self

    def transform_train_(self, X):
        X_result = np.zeros((X.shape[0], len(self.base_models)))
        for i, rgr in enumerate(self.base_models):
            for j, (train_idx, test_idx) in enumerate(self.folds):
                X_holdout = X[test_idx]
                y_pred = self.base_models[i, j].predict(X_holdout)[:]
                X_result[test_idx, i] = y_pred
        return X_result

    def transform_test_(self, X):
        X_result = np.zeros((X.shape[0], len(self.base_models)))
        for i, rgr in enumerate(self.base_models):
            X_result_i = np.zeros((X.shape[0], self.n_splits))
            for j, (train_idx, test_idx) in enumerate(self.folds):
                X_result_i[:, j] = self.base_models[i, j].predict(X)[:]
            X_result[:, i] = X_result_i.mean(axis=1)
        return X_result

    def transform(self, X):
        if hash(X.data.tobytes()) == self.hash:
            return self.transform_train_(X)
        else:
            return self.transform_test_(X)

    def fit_transform(self, X, y):
        self.hash = hash(X.data.tobytes())
        self.fit(X, y)
        self.isTrainFit = True
        return self.transform_train_(X, y)


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
