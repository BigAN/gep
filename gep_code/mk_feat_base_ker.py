import pandas as pd
import numpy as np
import math
import gep_lib.feature_lib as flb
import gep_lib.const as cst
import gep_lib.parse_cmd as pcmd
from sklearn import preprocessing
import gc
args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

key = 'base_ker'

import gep_lib.utils as ut

no_cols = "V194,V173,V216,V213,V134,V125,V175,V40,V123,V253,V161,V181,V63,V84,V286,V197,V200,V16,V72,V304,id_34,V250,V243,V57,V336,V306,V22,V242,V179,V158,V255,V331,V154,V46,V47,V178,V202,V211,V32,V60,V183,V320,V177,id_04,V248,V59,V283,V227,V259,id_03,V245,id_15,V222,V18,V192,V214,V234,V229,V239,V333,V190,V204,V294,V2,V3,id_11,V318,V71,D7"
no_cols = no_cols.split(",")


def feature_func(dataframe):
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other',
              'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo',
              'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft',
              'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google',
              'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other',
              'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft',
              'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
              'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other',
              'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft',
              'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other',
              'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo',
              'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
              'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
              'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft',
              'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
    us_emails = ['gmail', 'net', 'edu']
    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]

    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]
    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]

    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]
    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]

    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]
    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]

    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]
    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]

    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[
                                                 dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    dataframe['had_id'] = 1
    gc.collect()

    columns_a = ['TransactionAmt', 'id_02', 'D15']
    columns_b = ['card1', 'card4', 'addr1']

    for col_a in columns_a:
        for col_b in columns_b:
            for df in [dataframe]:
                df['{}_to_mean_{}'.format(col_a,col_b)] = df[col_a] / df.groupby([col_b])[col_a].transform('mean')
                df['{}_to_std_{}'.format(col_a,col_b)] = df[col_a] / df.groupby([col_b])[col_a].transform('std')

    for c in ['P_emaildomain', 'R_emaildomain']:
        dataframe[c + '_bin'] = dataframe[c].map(emails)

        dataframe[c + '_suffix'] = dataframe[c].map(lambda x: str(x).split('.')[-1])

        dataframe[c + '_suffix'] = dataframe[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')


    inp = dataframe
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

    ori_cols = train_key.columns.tolist()  # others use base
    train_index = len(train_key)
    feature = pd.concat([train_base, test_base])

with ut.tick_tock('cal fea'):
    out = feature_func(feature)
    out_cols = list(set(feature) - set(ori_cols) -set(no_cols))

    print len(out_cols),out_cols

with ut.tick_tock("write data"):
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False)
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False)
