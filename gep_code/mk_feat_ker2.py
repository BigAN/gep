import pandas as pd
import numpy as np
import math
import gep_lib.feature_lib as flb
import gep_lib.const as cst
import gep_lib.parse_cmd as pcmd
from sklearn import preprocessing
import datetime
args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

key = 'k2_1'

import gep_lib.utils as ut


def feature_func(train,test):
    train_df = train
    test_df = test
    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

    for df in [train_df, test_df]:
        ########################### Device info
        df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
        df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
        # df['DeviceInfo_version'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

        ########################### Device info 2
        df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
        df['id_30_device'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
        # df['id_30_version'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

        ########################### Browser
        df['id_31'] = df['id_31'].fillna('unknown_device').str.lower()
        df['id_31_device'] = df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))

    for df in [train_df, test_df]:
        # Temporary
        df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        df['DT_M'] = (df['DT'].dt.year - 2017) * 12 + df['DT'].dt.month
        df['DT_W'] = (df['DT'].dt.year - 2017) * 52 + df['DT'].dt.weekofyear
        df['DT_D'] = (df['DT'].dt.year - 2017) * 365 + df['DT'].dt.dayofyear

        df['DT_hour'] = df['DT'].dt.hour
        df['DT_day_week'] = df['DT'].dt.dayofweek
        df['DT_day'] = df['DT'].dt.day

        # D9 column
        df['D9'] = np.where(df['D9'].isna(), 0, 1)

    i_cols = ['card1']

    for col in i_cols:
        valid_card = pd.concat([train_df[[col]], test_df[[col]]])
        valid_card = valid_card[col].value_counts()
        valid_card = valid_card[valid_card > 2]
        valid_card = list(valid_card.index)

        train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
        test_df[col] = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)

        train_df[col] = np.where(train_df[col].isin(valid_card), train_df[col], np.nan)
        test_df[col] = np.where(test_df[col].isin(valid_card), test_df[col], np.nan)

    i_cols = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']

    for df in [train_df, test_df]:
        df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)
        df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)

    # for col in ['ProductCD', 'M4']:
    #     temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(
    #         columns={'mean': col + '_target_mean'})
    #     temp_dict.index = temp_dict[col].values
    #     temp_dict = temp_dict[col + '_target_mean'].to_dict()
    #
    #     train_df[col + '_target_mean'] = train_df[col].map(temp_dict)
    #     test_df[col + '_target_mean'] = test_df[col].map(temp_dict)

    train_df['uid'] = train_df['card1'].astype(str) + '_' + train_df['card2'].astype(str)
    test_df['uid'] = test_df['card1'].astype(str) + '_' + test_df['card2'].astype(str)

    train_df['uid2'] = train_df['uid'].astype(str) + '_' + train_df['card3'].astype(str) + '_' + train_df[
        'card5'].astype(str)
    test_df['uid2'] = test_df['uid'].astype(str) + '_' + test_df['card3'].astype(str) + '_' + test_df['card5'].astype(
        str)

    train_df['uid3'] = train_df['uid2'].astype(str) + '_' + train_df['addr1'].astype(str) + '_' + train_df[
        'addr2'].astype(str)
    test_df['uid3'] = test_df['uid2'].astype(str) + '_' + test_df['addr1'].astype(str) + '_' + test_df['addr2'].astype(
        str)

    # Check if the Transaction Amount is common or not (we can use freq encoding here)
    # In our dialog with a model we are telling to trust or not to these values
    train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
    test_df['TransactionAmt_check'] = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)

    # For our model current TransactionAmt is a noise
    # https://www.kaggle.com/kyakovlev/ieee-check-noise
    # (even if features importances are telling contrariwise)
    # There are many unique values and model doesn't generalize well
    # Lets do some aggregations
    i_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3']

    for col in i_cols:
        for agg_type in ['mean', 'std']:
            new_col_name = col + '_TransactionAmt_' + agg_type
            temp_df = pd.concat([train_df[[col, 'TransactionAmt']], test_df[[col, 'TransactionAmt']]])
            # temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
            temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                columns={agg_type: new_col_name})

            temp_df.index = list(temp_df[col])
            temp_df = temp_df[new_col_name].to_dict()

            train_df[new_col_name] = train_df[col].map(temp_df)
            test_df[new_col_name] = test_df[col].map(temp_df)

    # Small "hack" to transform distribution
    # (doesn't affect auc much, but I like it more)
    # please see how distribution transformation can boost your score
    # (not our case but related)
    # https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
    train_df['TransactionAmt'] = np.log1p(train_df['TransactionAmt'])
    test_df['TransactionAmt'] = np.log1p(test_df['TransactionAmt'])

    i_cols = ['card1', 'card2', 'card3', 'card5',
              'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
              'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
              'addr1', 'addr2',
              'dist1', 'dist2',
              'P_emaildomain', 'R_emaildomain',
              'DeviceInfo', 'DeviceInfo_device',
              'id_30', 'id_30_device',
              'id_31_device',
              'id_33',
              'uid', 'uid2', 'uid3',
              ]



    for col in i_cols:
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
        train_df[col + '_fq_enc'] = train_df[col].map(fq_encode)
        test_df[col + '_fq_enc'] = test_df[col].map(fq_encode)

    for col in ['DT_M', 'DT_W', 'DT_D']:
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        fq_encode = temp_df[col].value_counts().to_dict()

        train_df[col + '_total'] = train_df[col].map(fq_encode)
        test_df[col + '_total'] = test_df[col].map(fq_encode)

    periods = ['DT_M', 'DT_W', 'DT_D']
    i_cols = ['uid']
    for period in periods:
        for col in i_cols:
            new_column = col + '_' + period

            temp_df = pd.concat([train_df[[col, period]], test_df[[col, period]]])
            temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
            fq_encode = temp_df[new_column].value_counts().to_dict()

            train_df[new_column] = (train_df[col].astype(str) + '_' + train_df[period].astype(str)).map(fq_encode)
            test_df[new_column] = (test_df[col].astype(str) + '_' + test_df[period].astype(str)).map(fq_encode)

            train_df[new_column] /= train_df[period + '_total']
            test_df[new_column] /= test_df[period + '_total']


    inp = pd.concat([train_df,test_df])

    for f in inp.columns:
        if inp[f].dtype == 'object' or inp[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(inp[f].values))
            inp[f] = lbl.transform(list(inp[f].values))

    return inp


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
    out_cols = list(set(out) - set(ori_cols) - set(['DT']))

    print out_cols

with ut.tick_tock("write data"):
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False)
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False)
