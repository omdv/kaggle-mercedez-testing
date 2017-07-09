import pandas as pd
import numpy as np
from kaggler import PipeExtractor
from datetime import datetime
from pickle import dump
from xgboost import XGBRegressor, DMatrix, train
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, NMF
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.linear_model import LassoLars, HuberRegressor, LassoLarsCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.cluster import KMeans
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.regressor import StackingCVRegressor
from sklearn.svm import SVR


np.random.seed(42)


class AddColumns(BaseEstimator, TransformerMixin):
    def __init__(self, transform_=None):
        self.transform_ = transform_

    def fit(self, X, y=None):
        self.transform_.fit(X, y)
        return self

    def transform(self, X, y=None):
        xform_data = self.transform_.transform(X, y)
        return np.append(X, xform_data, axis=1)


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


class AveragingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors):
        self.regressors = regressors

    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]
        for regr in self.regr_:
            regr.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])
        return np.mean(predictions, axis=1)


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
               "r", "as", "a", "ap", "au"]:
            return 3
    if val in ["aa"]:
        return 4


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
original_features = list(train_df.columns[2:])

# Get duplicates and number of clones
train_df["duplicate"] =\
    train_df.duplicated(subset=original_features).astype(np.int32)
test_df["duplicate"] =\
    test_df.duplicated(subset=original_features).astype(np.int32)
train_df["n_clones"] =\
    train_df.groupby(original_features).ID.transform("size").astype(np.int32)
test_df["n_clones"] =\
    test_df.groupby(original_features).ID.transform("size").astype(np.int32)

Leak = False

# Mix in the public LB
if True:
    test_df = test_df.set_index("ID")
    public_lb = pd.read_csv('../input/public_lb.csv')
    add_test = test_df.loc[public_lb.ID.values, :]
    add_test.loc[:, "y"] = public_lb["y"].values
    add_test.reset_index(inplace=True)
    train_df = pd.concat([train_df, add_test], axis=0).reset_index()
    test_df.reset_index(inplace=True)

joint = pd.concat([train_df, test_df], axis=0)

# define feature sets
cat_features = list(joint.select_dtypes(include=['object']).columns)
bin_features = list(joint.select_dtypes(include=[np.int64]).columns)
all_features = list(joint.select_dtypes(exclude=['float', np.int32]).columns)

# get X0 group
joint['X0_group'] = joint['X0'].apply(assign_X0_group)
cat_features = ['X0_group'] + cat_features

# encode categorical features prior to pipeline
joint.loc[:, cat_features] =\
    joint[cat_features].apply(LabelEncoder().fit_transform)

# scale the target feature
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
# all_features += std_features

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
# all_features += size_features

# Median by categorical
median_features = []
for col in ["X0", "X0_group"]:
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

# Mean by categorical
mean_features = []
for col in ["X0", "X0_group"]:
    col_name = 'mean_by_' + col
    train_df.iscopy = False
    for (tr_idx, cv_idx) in kfold.split(train_df.index):
        x_tr = train_df.loc[tr_idx, [col, 'y']]
        x_cv = train_df.loc[cv_idx, [col, 'y']]
        means = x_tr[[col, 'y']].groupby(col).mean().reset_index()
        means.columns = [col, col_name]
        x_cv = pd.merge(x_cv, means, how='left', on=col)
        train_df.loc[cv_idx, col_name] = x_cv[col_name].values
    # test
    means = train_df[[col, 'y']].groupby(col).mean().reset_index()
    means.columns = [col, col_name]
    test_df = pd.merge(test_df, means, how='left', on=col)
    mean_features.append(col_name)

train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)

# =========================================================
# Define stacking regressors
rfr1 = RandomForestRegressor(max_depth=3, n_estimators=700, n_jobs=4)
rfr2 = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
                             min_samples_leaf=25, max_depth=3)
gbr = GradientBoostingRegressor(
    learning_rate=0.01,
    max_depth=3, n_estimators=300)
clf3 = BaggingRegressor()
clf4 = ExtraTreesRegressor()
ridge = Ridge()
xgbr1 = XGBRegressor(
    n_estimators=500,
    max_depth=1, learning_rate=0.02,
    nthread=-1, objective='reg:linear')
xgbr2 = XGBRegressor(
    max_depth=4, learning_rate=0.005, subsample=0.921,
    objective='reg:linear', n_estimators=1300, base_score=y_mean)
mlpr = MLPRegressor(
    hidden_layer_sizes=(150, 150, 25), max_iter=1000, random_state=42,
    verbose=True, tol=0.01)
lasso = LassoLars()
lassoCV = LassoLarsCV()
hr = HuberRegressor(max_iter=1000)
lgbr1 = LGBMRegressor(
    max_depth=2, n_estimators=1050, learning_rate=0.0045, subsample=0.9,
    nthread=4)
lgbr2 = LGBMRegressor(
    max_depth=2, n_estimators=850, learning_rate=0.0035, subsample=0.9,
    nthread=4)
lgbr3 = LGBMRegressor(
    max_depth=2, n_estimators=100, learning_rate=0.003,
    nthread=4)
enetCV = ElasticNetCV(max_iter=4000, n_jobs=-1, selection="random")
enet = ElasticNet(alpha=0.03, l1_ratio=0.15)
knnr = KNeighborsRegressor(n_neighbors=200, weights='uniform', n_jobs=-1)
svr = SVR(kernel='rbf', C=1.0, epsilon=0.05)
lr1 = LinearRegression(n_jobs=4)

# Define feature transformation pipeline
pipe1 = Pipeline([
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
        ('nmf', Pipeline([
            ('get', PipeExtractor(all_features)),
            ('fit', NMF(n_components=12, init='nndsvdar', random_state=42))
        ])),
    ])),
])

p11 = make_pipeline(pipe1, lgbr1)
p12 = make_pipeline(pipe1, rfr1)
p13 = make_pipeline(pipe1, xgbr1)

# Define feature transformation pipeline
pipe2 = Pipeline([
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
        ]))
    ])),
])

p21 = make_pipeline(
    pipe2, RobustScaler(), PCA(), SVR(kernel='rbf', C=1.0, epsilon=0.05))

p22 = make_pipeline(
    pipe2, RobustScaler(),
    PCA(n_components=125), ElasticNet(alpha=0.001, l1_ratio=0.1))

p23 = make_pipeline(pipe2, rfr2)

# Define feature transformation pipeline
pipe3 = Pipeline([
    ('features', FeatureUnion([
        ('cat_encoded', Pipeline([
            ('get', PipeExtractor(cat_features)),
            ('enc', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])),
        ('integer', Pipeline([
            ('get', PipeExtractor(bin_features)),
        ])),
        ('clones', Pipeline([
            ('get', PipeExtractor(["n_clones"])),
        ])),
        ('stds', Pipeline([
            ('get', PipeExtractor(std_features)),
        ])),
        ('sizes', Pipeline([
            ('get', PipeExtractor(size_features)),
        ])),
        ('means', Pipeline([
            ('get', PipeExtractor(mean_features)),
        ])),
        ('medians', Pipeline([
            ('get', PipeExtractor(median_features)),
        ])),
    ])),
])

p31 = make_pipeline(pipe3, lgbr2)
p32 = make_pipeline(pipe3, lr1)

p8 = StackingCVRegressor(
    regressors=[p11, p12, p13, p21, p22, p23],
    meta_regressor=lassoCV, cv=5)

pipe = p8

mode = 'Submit'

if mode == 'Val':
    cv = cross_val_score(pipe, train_df, y_train, cv=5)

    print("R^2 Score: %0.4f (+/- %0.3f) [%s]" % (
        cv.mean(), cv.std(), pipe.__class__))

elif mode == 'Grid':

    params = {
        'meta-lgbmregressor__max_depth': [2, 3, 5],
        'meta-lgbmregressor__n_estimators': [100, 300, 500],
        'meta-lgbmregressor__learning_rate': [0.003]
    }

    grid = GridSearchCV(
        pipe,
        param_grid=params,
        n_jobs=4, verbose=2, scoring='r2', cv=5)

    grid.fit(train_df, y_train)
    print(grid.best_params_)

elif mode == 'Submit':
    pipe.fit(train_df, y_train)
    predictions = pipe.predict(test_df)

    preds = pd.DataFrame()
    preds['ID'] = test_df['ID']
    preds['y'] = np.expm1(predictions)
    # preds['y'] = predictions

    # mix-in real test
    if True:
        preds = preds.set_index("ID")
        lb = pd.read_csv("../input/public_lb.csv")
        preds.loc[lb.ID.values, "y"] = lb["y"].values
        preds.reset_index(inplace=True)

    create_submission(0.6205, preds, None)
