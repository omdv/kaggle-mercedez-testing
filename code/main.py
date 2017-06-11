import pandas as pd
import numpy as np
from kaggler import PipeExtractor, BlendingRegressor
from datetime import datetime
from pickle import dump
from sklearn.pipeline import Pipeline, FeatureUnion
from xgboost import XGBRegressor, DMatrix, train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from brew.base import Ensemble, EnsembleClassifier
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from brew.combination.combiner import Combiner

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
])


mode = 'Stacking'

if mode == 'Val':
    x_train, x_valid, y_train, y_valid =\
        train_test_split(train_df, y_train, test_size=0.2)

    x_train = fpipe1.fit_transform(x_train, y_train)
    x_valid = fpipe1.transform(x_valid)
    preds, model = runXGB(
        x_train, y_train, x_valid, y_valid
        max_depth=2, num_rounds=2000, eta=0.005, y_mean=y_mean)
    print("r2: {:06.4f}".format(r2_score(y_valid, preds)))

elif mode == 'Grid':
    x_train, x_valid, y_train, y_valid =\
        train_test_split(train_df, y_train, test_size=0.2)

    x_train = fpipe1.fit_transform(x_train, y_train)
    x_valid = fpipe1.transform(x_valid)

    xgb = XGBRegressor(nthread=-1, objective='reg:linear')

    param_grid = dict(
        max_depth=[1, 3, 5],
        learning_rate=[0.005, 0.01, 0.02, 0.03],
        n_estimators=[300, 500, 700])

    grid = GridSearchCV(
        xgb, param_grid=param_grid, n_jobs=1, verbose=2, scoring='r2')
    grid.fit(x_train, y_train)

elif mode == 'Stacking':
    x_train, x_valid, y_train, y_valid =\
        train_test_split(train_df, y_train, test_size=0.2)

    x_train = fpipe1.fit_transform(x_train, y_train)
    x_valid = fpipe1.transform(x_valid)

    # Define stacking regressor
    clf1 = RandomForestRegressor(n_estimators=500)
    clf2 = GradientBoostingRegressor()
    clf3 = BaggingRegressor()
    clf4 = ExtraTreesRegressor()
    clf5 = Ridge()
    clf6 = XGBRegressor(
        n_estimators=900,
        max_depth=2, learning_rate=0.005,
        nthread=-1, objective='reg:linear')
    clf7 = MLPRegressor(hidden_layer_sizes=(200, 25))
    meta = XGBRegressor(base_score=np.mean(y_train))

    # for clf, label in zip(
    #     [clf1, clf2, clf3, clf4, clf5, clf6, clf7], [
    #         'RF',
    #         'GradientBoosting',
    #         'Bagging',
    #         'ExtraTrees',
    #         'Ridge',
    #         'XGBoost',
    #         'MLP']):
    #             clf.fit(x_train, y_train)
    #             print("{}: {:06.4f}".format(
    #                 label, r2_score(y_valid, clf.predict(x_valid))))

    # custom method
    ens = BlendingRegressor(
        [clf1, clf2, clf3, clf4, clf5, clf6, clf7], verbose=1)
    x_tr = ens.fit_transform_train(x_train, y_train)
    x_vl = ens.fit_transform_test(x_train, y_train, x_valid)
    meta.fit(x_tr, y_train)
    print("R2: {:06.4f}".format(r2_score(y_valid, meta.predict(x_vl))))

    # # brew method
    # layer_1 = Ensemble([clf1, clf2, clf3, clf4, clf5, clf6, clf7])
    # # layer_1 = Ensemble([clf1])
    # layer_2 = Ensemble([meta])
    # stack = EnsembleStack(cv=5, mode='labels')
    # stack.add_layer(layer_1)
    # stack.add_layer(layer_2)
    # sclf = EnsembleStackClassifier(stack)
    # sclf.fit(x_train, y_train)
    # r2_score(y_valid, sclf.predict(x_valid))


elif mode == 'Submit':
    x_train = fpipe1.fit_transform(train_df, y_train)
    x_test = fpipe1.transform(test_df)

    predictions, model = runXGB(
        x_train, y_train, x_test,
        max_depth=4, num_rounds=1250, eta=0.0045, y_mean=y_mean)

    preds = pd.DataFrame()
    preds['ID'] = test_df['ID']
    preds['y'] = predictions

    create_submission(0.6004, preds, model)
