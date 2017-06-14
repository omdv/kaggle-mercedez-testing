import pandas as pd
import numpy as np
from kaggler import PipeExtractor, BlendingRegressor
from datetime import datetime
from pickle import dump
from sklearn.pipeline import Pipeline, FeatureUnion
from xgboost import XGBRegressor, DMatrix, train
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
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
from sklearn.linear_model import ElasticNet, LassoLars
from sklearn.model_selection import cross_val_score

np.random.seed(42)


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
joint = pd.concat([train_df, test_df], axis=0)

# define feature sets
cat_features = joint.select_dtypes(include=['object']).columns
int_features = joint.select_dtypes(include=[np.int64]).columns
all_features = joint.select_dtypes(exclude=['float']).columns

# encode categorical features prior to pipeline
joint.loc[:, cat_features] =\
    joint[cat_features].apply(LabelEncoder().fit_transform)

# choose target feature
y_train = train_df['y'].values
y_mean = np.mean(y_train)

# split back
train_df = joint[joint['y'].notnull()]
test_df = joint[joint['y'].isnull()]

# =========================================================
# Define stacking regressors
clf1 = RandomForestRegressor(max_depth=3, n_estimators=500)
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
clf8 = LassoLars(normalize=True)
clf9 = LGBMRegressor()
clf10 = ElasticNet()
meta = XGBRegressor()

# Define feature transformation pipeline
fpipe1 = Pipeline([
    ('features', FeatureUnion([
        ('categorical', Pipeline([
            ('get', PipeExtractor(cat_features)),
        ])),
        ('integer', Pipeline([
            ('get', PipeExtractor(int_features)),
        ])),
        ('tsvd', Pipeline([
            ('get', PipeExtractor(all_features)),
            ('fit', TruncatedSVD(n_components=12))
        ])),
        ('pca', Pipeline([
            ('get', PipeExtractor(all_features)),
            ('fit', PCA(n_components=12))
        ])),
        ('fastica', Pipeline([
            ('get', PipeExtractor(all_features)),
            ('fit', FastICA(n_components=12))
        ])),
        ('gauss', Pipeline([
            ('get', PipeExtractor(all_features)),
            ('fit', GaussianRandomProjection(n_components=12, eps=0.1))
        ])),
        ('sparserandom', Pipeline([
            ('get', PipeExtractor(all_features)),
            ('fit', SparseRandomProjection(n_components=12, dense_output=True))
        ])),
    ])),
    ('rgr', BlendingRegressor(
        [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10],
        verbose=1))
])


mode = 'Submit'

if mode == 'Val':
    x_train = fpipe1.fit_transform(train_df, y_train)
    # x_valid = fpipe1.transform(x_valid)

    # cv = cross_val_score(XGBRegressor(
    #     n_estimators=800,
    #     max_depth=1, learning_rate=0.01,
    #     nthread=-1, objective='reg:linear'), x_train, y_train, cv=5)

    cv = cross_val_score(MLPRegressor(
        hidden_layer_sizes=[300, 300]), x_train, y_train, cv=3)

    print(cv)
    print(np.mean(cv))

elif mode == 'Grid':
    x_train = np.loadtxt("../input/full_fpipe01_train.csv", delimiter=",")

    param_grid = dict(
        max_depth=[1, 2, 3],
        learning_rate=[0.005, 0.007, 0.01, 0.015, 0.02, 0.03],
        n_estimators=[300, 400, 500, 600, 700]
    )

    grid = GridSearchCV(
        XGBRegressor(nthread=-1, objective="reg:linear"), param_grid,
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

    x_train = np.loadtxt("../input/full_fpipe01_train.csv", delimiter=",")
    x_test = np.loadtxt("../input/full_fpipe01_test.csv", delimiter=",")

    model = XGBRegressor(
        nthread=-1, objective="reg:linear", max_depth=2, learning_rate=0.015,
        n_estimators=300)
    model.fit(x_train, y_train)

    preds = pd.DataFrame()
    preds['ID'] = test_df['ID']
    preds['y'] = model.predict(x_test)

    create_submission(0.5441, preds, model)
