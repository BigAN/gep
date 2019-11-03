import os
import numpy as np
import random as rn
import lightgbm as lgb
import gc


import numpy as np
np.random.seed()
import pandas as pd
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.wrappers.scikit_learn import KerasClassifier


def rank_gauss(x):
    from scipy.special import erfinv
    N = x.shape[0]*1.0
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

def LGBM_helper(_X_tr, _X_va, _X_te, label_name,predictors, cat_feats, params, seed=2018):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    X_tr = _X_tr[predictors]
    X_va = _X_va[predictors]
    X_te = _X_te[predictors]
    y_tr = _X_tr[label_name]
    y_va = _X_va[label_name]
    y_te = _X_te[label_name]
    params['feature_fraction_seed'] = seed
    params['bagging_seed'] = seed
    params['drop_seed'] = seed
    params['data_random_seed'] = seed
    params['num_leaves'] = int(params['num_leaves'])
    params['subsample_for_bin'] = int(params['subsample_for_bin'])
    params['max_depth'] = int(np.log2(params['num_leaves']) + 1.2)
    params['max_bin'] = int(params['max_bin'])
    print('*' * 50)
    for k, v in sorted(params.items()):
        print(k, ':', v)
    columns = X_tr.columns

    print('start for lgvalid')
    lgvalid = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_feats)
    _X_va.drop(predictors, axis=1)
    del _X_va, X_va, y_va
    gc.collect()

    print('start for lgtrain')
    lgtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_feats)
    _X_te.drop(predictors, axis=1)
    del _X_tr, X_tr, y_tr
    gc.collect()

    evals_results = {}
    # if get_opt('trainCheck', '-') == 'on':
    valid_names = ['train', 'valid']
    valid_sets = [lgtrain, lgvalid]
    # else:
    # valid_names = ['valid']
    # valid_sets = [lgvalid]
    # if get_opt('testCheck', '-') == 'on':
    #     valid_names.append('test')
    #     lgtest = lgb.Dataset(X_te, label=y_te, categorical_feature=cat_feats)
    #     valid_sets.append(lgtest)

    print('start training')
    bst = lgb.train(params,
                    lgtrain,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    evals_result=evals_results,
                    num_boost_round=2000,
                    early_stopping_rounds=100,
                    verbose_eval=50,
                    )

    importance = bst.feature_importance()
    print('importance (count)')
    tuples = sorted(zip(columns, importance), key=lambda x: x[1], reverse=True)
    for col, val in tuples:
        print(val, "\t", col)

    importance = bst.feature_importandiscount#price_avg_person \ce(importance_type='gain')
    print('importance (gain)')
    tuples = sorted(zip(columns, importance), key=lambda x: x[1], reverse=True)
    for col, val in tuples:
        print(val, "\t", col)

    n_estimators = bst.best_iteration
    metric = params['metric']
    auc = evals_results['valid'][metric][n_estimators - 1]
    _X_te[label_name] = bst.predict(X_te)

    return auc


