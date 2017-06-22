import pandas as pd
import numpy as np
from kaggler import PipeExtractor
from datetime import datetime
from pickle import dump
from xgboost import XGBRegressor, DMatrix, train
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.linear_model import LassoLars, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.regressor import StackingCVRegressor
from sklearn.svm import SVR


np.random.seed(42)


class StackerTransformer():
    def __init__(self, base_models=None, n_splits=5,
                 verbose=0, scoring=r2_score):
        self.n_models = len(base_models)
        self.n_splits = n_splits
        self.base_models =\
            [[m for m in base_models] for s in np.arange(n_splits + 1)]
        self.folds = None
        self.verbose = verbose
        self.hash = None
        self.scoring = scoring

    def fit(self, X, y):
        self.hash = hash(X.data.tobytes())
        self.isTrainFit = True
        self.folds = list(
            KFold(n_splits=self.n_splits, shuffle=True).split(X, y))
        scores = np.zeros(self.n_splits)

        for i in np.arange(self.n_models):
            for j, (train_idx, test_idx) in enumerate(self.folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]

                self.base_models[j][i].fit(X_train, y_train)
                y_pred = self.base_models[j][i].predict(X_holdout)[:]

                if self.verbose > 1:
                    print ("Model %d fold %d score %f" %
                           (i, j, r2_score(y_holdout, y_pred)))
                if self.verbose == 1:
                    scores[j] = r2_score(y_holdout, y_pred)

            # full model fit
            self.base_models[self.n_splits][i].fit(X, y)

            if self.verbose == 1:
                print("Model %d: %0.4f (+/- %0.3f)" % (
                    i, np.mean(scores), np.std(scores)))
        return self

    def transform_train_(self, X):
        X_result = np.zeros((X.shape[0], self.n_models))
        for i in np.arange(self.n_models):
            for j, (train_idx, test_idx) in enumerate(self.folds):
                X_holdout = X[test_idx]
                y_pred = self.base_models[j][i].predict(X_holdout)[:]
                X_result[test_idx, i] = y_pred
        return X_result

    def transform_test_(self, X):
        X_result = np.zeros((X.shape[0], self.n_models))
        # for i in np.arange(self.n_models):
        #     X_result_i = np.zeros((X.shape[0], self.n_splits))
        #     for j, (train_idx, test_idx) in enumerate(self.folds):
        #         X_result_i[:, j] = self.base_models[j][i].predict(X)[:]
        #     X_result[:, i] = X_result_i.mean(axis=1)
        for i in np.arange(self.n_models):
            X_result[:, i] = self.base_models[self.n_splits][i].predict(X)[:]
        return X_result

    def transform(self, X):
        if hash(X.data.tobytes()) == self.hash:
            return self.transform_train_(X)
        else:
            return self.transform_test_(X)

    def fit_transform(self, X, y):
        self.hash = hash(X.data.tobytes())
        self.fit(X, y)
        return self.transform_train_(X)


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=42).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                print (
                    "Model %d fold %d score %f" %
                    (i, j, r2_score(y_holdout, y_pred)))

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        # results = cross_val_score(
        #     self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, rgrs=None, weights=None):
        self.rgrs = rgrs
        self.weights = weights

    def fit(self, X, y):
        for rgr in self.rgrs:
            rgr.fit(X, y)

    def predict(self, X):
        self.preds_ = list()
        for rgr in self.rgrs:
            self.preds_.append(rgr.predict(X))
        self.preds_ = np.array(self.preds_).transpose()
        return np.sum(self.preds_ * self.weights, axis=1)


class GroupedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, rgr):
        self.groups = [0, 1, 2, 3]
        self.rgr = rgr
        self.rgrs = np.array([self.rgr for i in self.groups])

    def fit(self, X, y):
        for grp in self.groups:
            idx = X[:, 0] == grp
            self.rgrs[grp].fit(X[idx], y[idx])
        return self

    def predict(self, X):
        self.preds_ = np.ones((X.shape[0], 1))
        for grp in self.groups:
            idx = X[:, 0] == grp
            self.preds_[idx] = self.rgrs[grp].predict(X[idx]).reshape(-1, 1)
        return self.preds_


def assign_X0_group(val):
    if val in ["bc", "az"]:
        return 0
    if val in ["ac", "am", "l", "b", "aq", "u", "ad", "e", "al", "s", "n", "y",
               "t", "ai", "k", "f", "z", "o", "ba", "m", "q"]:
            return 1
    if val in ["d", "ay", "h", "aj", "v", "ao", "aw"]:
        return 2
    if val in ["c", "ax", "x", "j", "w", "i", "ak", "g", "at", "ab", "af",
               "r", "as", "a", "ap", "au", "aa"]:
            return 3


def runXGB(train_X, train_y, test_X=None, test_y=None, feature_names=None,
           seed_val=0, num_rounds=2000, max_depth=6,
           eta=0.03, scale_pos_weight=1.0, verbose_eval=10, y_mean=None):

    param = {}
    param['n_trees'] = 520
    param['objective'] = 'reg:linear'
    param['eval_metric'] = 'rmse'
    param['eta'] = eta
    param['max_depth'] = max_depth
    param['silent'] = 1
    param['min_child_weight'] = 1
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val
    param['scale_pos_weight'] = scale_pos_weight
    param['verbose_eval'] = 50
    param['base_score'] = y_mean
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
        model = train(
            plst, xgtrain, num_rounds, watchlist,
            early_stopping_rounds=100, verbose_eval=verbose_eval)
    else:
        watchlist = [(xgtrain, 'train')]
        model = train(
            plst, xgtrain, num_rounds,
            watchlist, verbose_eval=verbose_eval)

    if test_X is None:
        pred_test_y = None
    else:
        xgtest = DMatrix(test_X)
        pred_test_y = model.predict(xgtest)
    return pred_test_y, model


def create_submission(score, pred, model):
    """
    Saving model, features and submission
    """
    ouDir = '../output/'

    now = datetime.now()
    scrstr = "{:0.4f}_{}".format(score, now.strftime("%Y-%m-%d-%H%M"))

    mod_file = ouDir + 'model_' + scrstr + '.model'
    if model:
        print('Writing model: ', mod_file)
        dump(model, open(mod_file, 'wb'))

    sub_file = ouDir + 'submit_' + scrstr + '.csv'
    print('Writing submission: ', sub_file)
    pred.to_csv(sub_file, index=False)


# Main part
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# train_df.loc[883, 'y'] = 169.91
joint = pd.concat([train_df, test_df], axis=0)

# define feature sets
cat_features = list(joint.select_dtypes(include=['object']).columns)
bin_features = list(joint.select_dtypes(include=[np.int64]).columns)
all_features = list(joint.select_dtypes(exclude=['float']).columns)

# get X0 group
joint['X0_group'] = joint['X0'].apply(assign_X0_group)
cat_features = ['X0_group'] + cat_features

# encode categorical features prior to pipeline
joint.loc[:, cat_features] =\
    joint[cat_features].apply(LabelEncoder().fit_transform)

# drop outlier and choose target feature
y_train = np.log1p(train_df['y'].values)
y_mean = np.mean(y_train)

# split back
train_df = joint[joint['y'].notnull()]
test_df = joint[joint['y'].isnull()]
joint = 0

# STD by categorical
kfold = KFold(5)
std_features = []
for col in cat_features:
    col_name = 'std_by_' + col
    train_df.iscopy = False
    for (tr_idx, cv_idx) in kfold.split(train_df.index):
        x_tr = train_df.loc[tr_idx, [col, 'y']]
        x_cv = train_df.loc[cv_idx, [col, 'y']]
        means = x_tr[[col, 'y']].groupby(col).\
            agg({'y': np.nanstd}).reset_index()
        means.columns = [col, col_name]
        x_cv = pd.merge(x_cv, means, how='left', on=col)
        train_df.loc[cv_idx, col_name] = x_cv[col_name].values
    # test
    means = train_df[[col, 'y']].groupby(col).\
        agg({'y': np.nanstd}).reset_index()
    means.columns = [col, col_name]
    test_df = pd.merge(test_df, means, how='left', on=col)
    std_features.append(col_name)

# Group size by categorical
size_features = []
for col in cat_features:
    col_name = 'size_by_' + col
    train_df.iscopy = False
    for (tr_idx, cv_idx) in kfold.split(train_df.index):
        x_tr = train_df.loc[tr_idx, [col, 'y']]
        x_cv = train_df.loc[cv_idx, [col, 'y']]
        means = x_tr[[col, 'y']].groupby(col).size().reset_index()
        means.columns = [col, col_name]
        x_cv = pd.merge(x_cv, means, how='left', on=col)
        train_df.loc[cv_idx, col_name] = x_cv[col_name].values
    # test
    means = train_df[[col, 'y']].groupby(col).size().reset_index()
    means.columns = [col, col_name]
    test_df = pd.merge(test_df, means, how='left', on=col)
    size_features.append(col_name)

# Median by categorical
median_features = []
for col in cat_features:
    col_name = 'median_by_' + col
    train_df.iscopy = False
    for (tr_idx, cv_idx) in kfold.split(train_df.index):
        x_tr = train_df.loc[tr_idx, [col, 'y']]
        x_cv = train_df.loc[cv_idx, [col, 'y']]
        means = x_tr[[col, 'y']].groupby(col).median().reset_index()
        means.columns = [col, col_name]
        x_cv = pd.merge(x_cv, means, how='left', on=col)
        train_df.loc[cv_idx, col_name] = x_cv[col_name].values
    # test
    means = train_df[[col, 'y']].groupby(col).median().reset_index()
    means.columns = [col, col_name]
    test_df = pd.merge(test_df, means, how='left', on=col)
    median_features.append(col_name)

train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)

# =========================================================
# Define stacking regressors
rfr = RandomForestRegressor(max_depth=3, n_estimators=700, n_jobs=-1)
gbr = GradientBoostingRegressor(
    learning_rate=0.01,
    max_depth=3, n_estimators=300)
clf3 = BaggingRegressor()
clf4 = ExtraTreesRegressor()
ridge = Ridge()
xgbr = XGBRegressor(
    n_estimators=500,
    max_depth=1, learning_rate=0.02,
    nthread=-1, objective='reg:linear')
mlpr = MLPRegressor(
    hidden_layer_sizes=(150, 150, 25), max_iter=1000, random_state=42,
    verbose=True, tol=0.01)
lasso = LassoLars()
hr = HuberRegressor(max_iter=1000)
lgbr = LGBMRegressor(
    max_depth=2, n_estimators=1050, learning_rate=0.0045, subsample=0.9,
    nthread=-1)
enetCV = ElasticNetCV(max_iter=4000, n_jobs=-1, selection="random")
enet = ElasticNet(alpha=0.03, l1_ratio=0.15)
knnr = KNeighborsRegressor(n_neighbors=200, weights='uniform', n_jobs=-1)
svr = SVR(kernel='rbf', C=1.0, epsilon=0.05)

# Define feature transformation pipeline
pipe = Pipeline([
    ('features', FeatureUnion([
        ('categorical', Pipeline([
            ('get', PipeExtractor(cat_features)),
        ])),
        ('cat_encoded', Pipeline([
            ('get', PipeExtractor(cat_features)),
            ('enc', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])),
        ('integer', Pipeline([
            ('get', PipeExtractor(bin_features)),
        ])),
        ('stds', Pipeline([
            ('get', PipeExtractor(std_features)),
        ])),
        ('sizes', Pipeline([
            ('get', PipeExtractor(size_features)),
        ])),
        ('tsvd', Pipeline([
            ('get', PipeExtractor(all_features)),
            ('fit', TruncatedSVD(n_components=12, random_state=42))
        ])),
        ('pca', Pipeline([
            ('get', PipeExtractor(all_features)),
            ('fit', PCA(n_components=12, random_state=42))
        ])),
        ('fastica', Pipeline([
            ('get', PipeExtractor(all_features, )),
            ('fit', FastICA(n_components=12, random_state=42, max_iter=400))
        ])),
        ('gauss', Pipeline([
            ('get', PipeExtractor(all_features)),
            ('fit', GaussianRandomProjection(
                n_components=12, eps=0.1, random_state=42))
        ])),
        ('sparserandom', Pipeline([
            ('get', PipeExtractor(all_features)),
            ('fit', SparseRandomProjection(
                n_components=12, dense_output=True, random_state=42))
        ])),
    ])),
    # ('scaler', RobustScaler()),
    # ('blender', StackerTransformer(
    #     base_models=[lgbr, rfr, xgbr, enet], verbose=1)),
    # ('rgr', lasso)
    # ('rgr', enet)
    ('rgr', StackingCVRegressor(
        regressors=(lgbr, rfr, xgbr, enet),
        meta_regressor=lasso, cv=5))
    # ('rgr', GroupedRegressor(
    #     rgr=LassoLarsCV(normalize=True)))
    # ('rgr', EnsembleRegressor(rgrs=[lgbr, rfr], weights=[.5, .5]))
])

# x_train = pipe.fit_transform(train_df, y_train)
# x_test = pipe.transform(test_df)

# blender = StackerTransformer(base_models=[lgbr, enet])
# Xn = blender.fit_transform(X, y_train)


# x_train = pipe.fit_transform(train_df, y_train)
# rgr = GroupedRegressor(LGBMRegressor())
# kfold = KFold(5)
# for (tr_idx, cv_idx) in kfold.split(x_train):
#     rgr.fit(x_train[tr_idx, :], y_train[tr_idx])
#     print(r2_score(y_train[cv_idx], rgr.predict(x_train[cv_idx, :])))

# fpipe1.fit(train_df, y_train)
# kk = fpipe1.predict(test_df)

mode = 'Val'

if mode == 'Val':
    cv = cross_val_score(pipe, train_df, y_train, cv=5)

    print("R^2 Score: %0.4f (+/- %0.3f) [%s]" % (
        cv.mean(), cv.std(), pipe.named_steps['rgr'].__class__))

elif mode == 'Grid':
    x_train = pipe.fit_transform(train_df, y_train)

    # params = {
    #     'meta-lgbmregressor__max_depth': [2, 3, 4],
    #     'meta-lgbmregressor__learning_rate': [0.005, 0.01, 0.03],
    #     'meta-lgbmregressor__n_estimators': [100, 300, 500]
    # }

    params = {
        'alpha': [0.9, 1, 3, 5, 10, 20],
        # 'max_iter': [1, 3, 5, 7, 10, 12, 14, 16, 18]
    }

    grid = GridSearchCV(
        ridge,
        param_grid=params,
        n_jobs=1, verbose=2, scoring='r2', cv=5)

    grid.fit(x_train, y_train)

elif mode == 'Stacking':
    # x_train, x_valid, y_train, y_valid =\
    #     train_test_split(train_df, y_train, test_size=0.2)

    x_train = pipe.fit_transform(train_df, y_train)
    x_test = pipe.transform(test_df)

    # for clf, label in zip(
    #     [clf1, clf2, clf3, clf4, clf5, clf6, clf7], [
    #         'RF', 'GradientBoosting', 'Bagging', 'ExtraTrees',
    #         'Ridge', 'XGBoost', 'MLP']):
    #             clf.fit(x_train, y_train)
    #             print("{}: {:06.4f}".format(
    #                 label, r2_score(y_valid, clf.predict(x_valid))))

    # # custom method
    # ens = BlendingRegressor(
    #     [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10],
    #     verbose=1)
    # x_tr2 = ens.fit_transform(x_train, y_train)
    # x_vl2 = ens.transform(x_valid)

    np.savetxt("../input/full_fpipe01_train.csv", x_train, delimiter=",")
    np.savetxt("../input/full_fpipe01_test.csv", x_test, delimiter=",")

elif mode == 'Submit':
    pipe.fit(train_df, y_train)
    predictions = pipe.predict(test_df)

    preds = pd.DataFrame()
    preds['ID'] = test_df['ID']
    preds['y'] = np.expm1(predictions)

    create_submission(0.6230, preds, None)
