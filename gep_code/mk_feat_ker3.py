import pandas as pd
import numpy as np
import math
import gep_lib.feature_lib as flb
import gep_lib.const as cst
import gep_lib.parse_cmd as pcmd
from sklearn import preprocessing
import datetime
import gc
args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

key = 'k3_459'

import gep_lib.utils as ut
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

def feature_func(train,test):
    def addNewFeatures(data):
        data['uid'] = data['card1'].astype(str) + '_' + data['card2'].astype(str)

        data['uid2'] = data['uid'].astype(str) + '_' + data['card3'].astype(str) + '_' + data['card5'].astype(str)

        data['uid3'] = data['uid2'].astype(str) + '_' + data['addr1'].astype(str) + '_' + data['addr2'].astype(str)

        data['D9'] = np.where(data['D9'].isna(), 0, 1)

        return data

    train = addNewFeatures(train)
    test = addNewFeatures(test)
    i_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3']

    for col in i_cols:
        for agg_type in ['mean', 'std']:
            new_col_name = col + '_TransactionAmt_' + agg_type
            temp_df = pd.concat([train[[col, 'TransactionAmt']], test[[col, 'TransactionAmt']]])
            # temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
            temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                columns={agg_type: new_col_name})

            temp_df.index = list(temp_df[col])
            temp_df = temp_df[new_col_name].to_dict()

            train[new_col_name] = train[col].map(temp_df)
            test[new_col_name] = test[col].map(temp_df)

    train = train.replace(np.inf, 999)
    test = test.replace(np.inf, 999)
    train['TransactionAmt'] = np.log1p(train['TransactionAmt'])
    test['TransactionAmt'] = np.log1p(test['TransactionAmt'])

    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other',
              'optonline.net': 'other',
              'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
              'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol',
              'hotmail.de': 'microsoft',
              'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other',
              'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft',
              'protonmail.com': 'other',
              'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',
              'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
              'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',
              'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',
              'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
              'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
              'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
              'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft',
              'ptd.net': 'other',
              'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
    us_emails = ['gmail', 'net', 'edu']

    for c in ['P_emaildomain', 'R_emaildomain']:
        train[c + '_bin'] = train[c].map(emails)
        test[c + '_bin'] = test[c].map(emails)

        train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
        test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

        train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    p = 'P_emaildomain'
    r = 'R_emaildomain'
    uknown = 'email_not_provided'

    def setDomain(df):
        df[p] = df[p].fillna(uknown)
        df[r] = df[r].fillna(uknown)

        # Check if P_emaildomain matches R_emaildomain
        df['email_check'] = np.where((df[p] == df[r]) & (df[p] != uknown), 1, 0)

        df[p + '_prefix'] = df[p].apply(lambda x: x.split('.')[0])
        df[r + '_prefix'] = df[r].apply(lambda x: x.split('.')[0])

        return df

    train = setDomain(train)
    test = setDomain(test)

    def setTime(df):
        df['TransactionDT'] = df['TransactionDT'].fillna(df['TransactionDT'].median())
        # Temporary
        df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        df['DT_M'] = (df['DT'].dt.year - 2017) * 12 + df['DT'].dt.month
        df['DT_W'] = (df['DT'].dt.year - 2017) * 52 + df['DT'].dt.weekofyear
        df['DT_D'] = (df['DT'].dt.year - 2017) * 365 + df['DT'].dt.dayofyear

        df['DT_hour'] = df['DT'].dt.hour
        df['DT_day_week'] = df['DT'].dt.dayofweek
        df['DT_day'] = df['DT'].dt.day

        return df

    train = setTime(train)
    test = setTime(test)

    train["lastest_browser"] = np.zeros(train.shape[0])
    test["lastest_browser"] = np.zeros(test.shape[0])

    def setBrowser(df):
        df.loc[df["id_31"] == "samsung browser 7.0", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "opera 53.0", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "mobile safari 10.0", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "google search application 49.0", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "firefox 60.0", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "edge 17.0", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 69.0", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 67.0 for android", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 63.0 for android", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 63.0 for ios", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 64.0", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 64.0 for android", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 64.0 for ios", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 65.0", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 65.0 for android", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 65.0 for ios", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 66.0", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 66.0 for android", 'lastest_browser'] = 1
        df.loc[df["id_31"] == "chrome 66.0 for ios", 'lastest_browser'] = 1
        return df

    train = setBrowser(train)
    test = setBrowser(test)

    def setDevice(df):
        df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()

        df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]

        df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
        df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
        df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
        df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
        df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
        df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
        df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
        df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
        df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
        df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
        df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
        df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
        df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
        df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
        df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
        df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
        df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

        df.loc[df.device_name.isin(
            df.device_name.value_counts()[df.device_name.value_counts() < 200].index), 'device_name'] = "Others"
        df['had_id'] = 1
        gc.collect()

        return df

    train = setDevice(train)
    test = setDevice(test)

    i_cols = ['card1', 'card2', 'card3', 'card5',
              'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
              'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
              'addr1', 'addr2',
              'dist1', 'dist2',
              'P_emaildomain', 'R_emaildomain',
              'DeviceInfo', 'device_name',
              'id_30', 'id_33',
              'uid', 'uid2', 'uid3',
              ]

    for col in i_cols:
        temp_df = pd.concat([train[[col]], test[[col]]])
        fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
        train[col + '_fq_enc'] = train[col].map(fq_encode)
        test[col + '_fq_enc'] = test[col].map(fq_encode)

    for col in ['DT_M', 'DT_W', 'DT_D']:
        temp_df = pd.concat([train[[col]], test[[col]]])
        fq_encode = temp_df[col].value_counts().to_dict()

        train[col + '_total'] = train[col].map(fq_encode)
        test[col + '_total'] = test[col].map(fq_encode)

    periods = ['DT_M', 'DT_W', 'DT_D']
    i_cols = ['uid']
    for period in periods:
        for col in i_cols:
            new_column = col + '_' + period

            temp_df = pd.concat([train[[col, period]], test[[col, period]]])
            temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
            fq_encode = temp_df[new_column].value_counts().to_dict()

            train[new_column] = (train[col].astype(str) + '_' + train[period].astype(str)).map(fq_encode)
            test[new_column] = (test[col].astype(str) + '_' + test[period].astype(str)).map(fq_encode)

            train[new_column] /= train[period + '_total']
            test[new_column] /= test[period + '_total']


    inp = pd.concat([train,test])

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


    def get_too_many_null_attr(data):
        many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]
        return many_null_cols


    def get_too_many_repeated_val(data):
        big_top_value_cols = [col for col in data.columns if
                              data[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
        return big_top_value_cols


    def get_useless_columns(data):
        too_many_null = get_too_many_null_attr(data)
        print("More than 90% null: " + str(len(too_many_null)))
        too_many_repeated = get_too_many_repeated_val(data)
        print("More than 90% repeated value: " + str(len(too_many_repeated)))
        cols_to_drop = list(set(too_many_null + too_many_repeated))
        # cols_to_drop.remove('isFraud')
        return cols_to_drop


    cols_to_drop = get_useless_columns(out)

    print "cols_to_drop",cols_to_drop
    out_cols = list(set(out) - set(ori_cols) - set(['DT']) - set(cols_to_drop))

    print out_cols

with ut.tick_tock("write data"):
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False)
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False)
