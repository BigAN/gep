import pandas as pd
import numpy as np
import math
import gep_lib.feature_lib as flb
import gep_lib.const as cst
import gep_lib.parse_cmd as pcmd
from sklearn import preprocessing

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

key = 'date'

import gep_lib.utils as ut


def feature_func(inp):
    # inp['uid'] = inp['card1'].astype(str) + '_' + inp['card2'].astype(str)
    #
    # inp['uid2'] = inp['uid'].astype(str) + '_' + inp['card3'].astype(str) + '_' + inp[
    #     'card5'].astype(str)
    #
    # inp['uid3'] = inp['uid2'].astype(str) + '_' + inp['addr1'].astype(str) + '_' + inp[
    #     'addr2'].astype(str)
    with ut.tick_tock("1"):
        inp=inp.fillna(inp.mean())
    with ut.tick_tock("1"):
        inp["timestamp"] = pd.to_datetime(inp["timestamp"])
        inp['hour'] = np.uint8(inp['timestamp'].dt.hour)
        inp['day'] = np.uint8(inp['timestamp'].dt.day)
    with ut.tick_tock("1"):

        inp['weekday'] = np.uint8(inp['timestamp'].dt.weekday)
        inp['month'] = np.uint8(inp['timestamp'].dt.month)
        inp['year'] = np.uint8(inp['timestamp'].dt.year - 1900)
    with ut.tick_tock("2"):

        inp['square_feet_log'] = np.log(inp['square_feet'])
        dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')
        us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
        inp['is_holiday'] = (inp['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(
            np.int8)

        inp.loc[(inp['weekday'] == 5) | (inp['weekday'] == 6), 'is_holiday'] = 1
    with ut.tick_tock("3"):

        def degToCompass(num):
            val = int((num / 22.5) + .5)
            arr = [i for i in range(0, 16)]
            return arr[(val % 16)]

        inp['wind_direction_deg'] = inp['wind_direction'].apply(degToCompass)
    with ut.tick_tock("4"):

        for f in inp.columns:
            if inp[f].dtype == 'object' or inp[f].dtype == 'object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(inp[f].values))
                inp[f] = lbl.transform(list(inp[f].values))
            # inp[f] = lbl.transform(list(inp[f].values))

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
    out = feature_func(feature)
    print out.columns
    print out.head()
    # no_cols = "V194,V173,V216,V213,V134,V125,V175,V40,V123,V253,V161,V181,V63,V84,V286,V197,V200,V16,V72,V304,id_34,V250,V243,V57,V336,V306,V22,V242,V179,V158,V255,V331,V154,V46,V47,V178,V202,V211,V32,V60,V183,V320,V177,id_04,V248,V59,V283,V227,V259,id_03,V245,id_15,V222,V18,V192,V214,V234,V229,V239,V333,V190,V204,V294,V2,V3,id_11,V318,V71,D7"
    # no_cols = no_cols.split(",")
    out_cols = list(set(out) - set(ori_cols))
    print out_cols

with ut.tick_tock("write data"):
    feat_key = key
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False,header=ut.deco_outcols(feat_key, out_cols))
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False,header=ut.deco_outcols(feat_key, out_cols))
