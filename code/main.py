import pandas as pd
import numpy as np
import warnings
from kaggler import PipeExtractor, BlendingRegressor, PipeShape
from datetime import datetime
from pickle import dump
from sklearn.pipeline import Pipeline, FeatureUnion
from xgboost import XGBRegressor, DMatrix, train
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.cluster import KMeans

np.random.seed(42)


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

# train_df.drop(train_df.index[883], inplace=True)
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
clf2 = GradientBoostingRegressor(
    learning_rate=0.01,
    max_depth=3, n_estimators=300)
clf3 = BaggingRegressor()
clf4 = ExtraTreesRegressor()
clf5 = Ridge()
clf6 = XGBRegressor(
    n_estimators=500,
    max_depth=1, learning_rate=0.02,
    nthread=-1, objective='reg:linear')
clf7 = MLPRegressor(hidden_layer_sizes=(150, 25))
clf8 = LassoLarsCV(normalize=True)
lgbr = LGBMRegressor(
    max_depth=2, n_estimators=1050, learning_rate=0.0045, subsample=0.9,
    nthread=-1)
clf10 = ElasticNetCV()
clf11 = LGBMRegressor()
meta = LassoLarsCV()

# Define feature transformation pipeline
fpipe1 = Pipeline([
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
        # ('medians', Pipeline([
        #     ('get', PipeExtractor(median_features)),
        # ])),
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
            ('fit', FastICA(n_components=12, random_state=42))
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
    # ('rgr', BlendingRegressor(
    #     [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10],
    #     verbose=1))
    ('rgr', lgbr)
    # ('rgr', ElasticNetCV(normalize=True, verbose=0))
    # ('rgr', LassoLarsCV(normalize=True))
    # ('rgr', StackingRegressor(
    #     regressors=regressors, meta_regressor=LGBMRegressor(nthread=-1)))
    # ('rgr', GroupedRegressor(
    #     rgr=LassoLarsCV(normalize=True)))
    # ('rgr', EnsembleRegressor(rgrs=[lgbr, rfr], weights=[.5, .5]))
])


# x_train = fpipe1.fit_transform(train_df, y_train)
# rgr = GroupedRegressor(LGBMRegressor())
# kfold = KFold(5)
# for (tr_idx, cv_idx) in kfold.split(x_train):
#     rgr.fit(x_train[tr_idx, :], y_train[tr_idx])
#     print(r2_score(y_train[cv_idx], rgr.predict(x_train[cv_idx, :])))

# fpipe1.fit(train_df, y_train)
# kk = fpipe1.predict(test_df)

mode = 'Submit'

if mode == 'Val':
    cv = cross_val_score(fpipe1, train_df, y_train, cv=5)

    print(cv)
    print(np.mean(cv))

elif mode == 'Grid':
    # x_train = np.loadtxt("../input/full_fpipe01_train.csv", delimiter=",")
    x_train = fpipe1.fit_transform(train_df, y_train)

    # params = {
    #     'meta-lgbmregressor__max_depth': [2, 3, 4],
    #     'meta-lgbmregressor__learning_rate': [0.005, 0.01, 0.03],
    #     'meta-lgbmregressor__n_estimators': [100, 300, 500]
    # }

    weights = [[i, 1 - i] for i in np.arange(11) / 10]
    params = {
        'weights': weights
    }

    grid = GridSearchCV(
        EnsembleRegressor(rgrs=[lgbr, rfr]),
        param_grid=params,
        n_jobs=1, verbose=2, scoring='r2', cv=5)

    grid.fit(x_train, y_train)

elif mode == 'Stacking':
    # x_train, x_valid, y_train, y_valid =\
    #     train_test_split(train_df, y_train, test_size=0.2)

    x_train = fpipe1.fit_transform(train_df, y_train)
    x_test = fpipe1.transform(test_df)

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
    # x_train = fpipe1.fit_transform(train_df, y_train)
    # x_test = fpipe1.transform(test_df)

    # x_train = np.loadtxt("../input/full_fpipe01_train.csv", delimiter=",")
    # x_test = np.loadtxt("../input/full_fpipe01_test.csv", delimiter=",")

    # model = LGBMRegressor(
    #     max_depth=2, n_estimators=400, learning_rate=0.008,
    #     nthread=-1)
    # model.fit(x_train, y_train)

    fpipe1.fit(train_df, y_train)
    predictions = fpipe1.predict(test_df)

    preds = pd.DataFrame()
    preds['ID'] = test_df['ID']
    preds['y'] = np.expm1(predictions)

    create_submission(0.6176, preds, None)
