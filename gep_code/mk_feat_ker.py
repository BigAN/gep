import pandas as pd
import numpy as np
import math
import gep_lib.feature_lib as flb
import gep_lib.const as cst
import gep_lib.parse_cmd as pcmd
from sklearn import preprocessing

args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

key = 'kernel'

import gep_lib.utils as ut


def feature_func(train,test):
    train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])
    test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])

    # New feature - decimal part of the transaction amount.
    print "New feature - decimal part of the transaction amount."
    train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(
        int)
    test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

    # New feature - day of week in which a transaction happened.
    train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)
    test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

    # New feature - hour of the day in which a transaction happened.
    train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24
    test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24

    # Some arbitrary features interaction

    for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2',
                    'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:
        f1, f2 = feature.split('__')
        train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
        test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

        le = preprocessing.LabelEncoder()
        le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
        train[feature] = le.transform(list(train[feature].astype(str).values))
        test[feature] = le.transform(list(test[feature].astype(str).values))

    # Encoding - count encoding for both train and test
    for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_36']:
        train[feature + '_count_full'] = train[feature].map(
            pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
        test[feature + '_count_full'] = test[feature].map(
            pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

    # Encoding - count encoding separately for train and test
    for feature in ['id_01', 'id_31', 'id_33', 'id_36']:
        train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
        test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))
    rs = pd.concat([train,test])
    return rs


with ut.tick_tock('read_data'):
    print nrows
    train_base = pd.read_csv(cst.train_prefix + "deco_base.csv", ',', nrows=nrows)
    test_base = pd.read_csv(cst.test_prefix + "deco_base.csv", ',', nrows=nrows)

    train_key = pd.read_csv(cst.train_prefix + "key.csv", ',', nrows=nrows)
    test_key = pd.read_csv(cst.test_prefix + "key.csv", ',', nrows=nrows)

    ori_cols = train_base.columns.tolist()  # others use base
    train_index = len(train_key)
    feature = pd.concat([train_base, test_base])

with ut.tick_tock('cal fea'):
    out = feature_func(train_base,test_base)
    out_cols = list(set(out) - set(ori_cols))

    print out_cols

with ut.tick_tock("write data"):
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False)
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False)
