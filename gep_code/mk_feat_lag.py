import pandas as pd
import numpy as np
import math
import gep_lib.feature_lib as flb
import gep_lib.const as cst
import gep_lib.parse_cmd as pcmd
from sklearn import preprocessing
import os
import gep_lib.utils as ut

args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

key = ut.get_key(os.path.basename(__file__))

print key
def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
            'wind_direction', 'wind_speed']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        weather_df['{col}_mean_lag{window}'.format(**{"col":col,"window":window})] = lag_mean[col]
        weather_df['{col}_max_lag{window}'.format(**{"col":col,"window":window})] = lag_max[col]
        weather_df['{col}_min_lag{window}'.format(**{"col":col,"window":window})] = lag_min[col]
        weather_df['{col}_std_lag{window}'.format(**{"col":col,"window":window})] = lag_std[col]


def feature_func(inp,weather_df):
    # inp['uid'] = inp['card1'].astype(str) + '_' + inp['card2'].astype(str)
    #
    # inp['uid2'] = inp['uid'].astype(str) + '_' + inp['card3'].astype(str) + '_' + inp[
    #     'card5'].astype(str)
    #
    # inp['uid3'] = inp['uid2'].astype(str) + '_' + inp['addr1'].astype(str) + '_' + inp[
    #     'addr2'].astype(str)


    add_lag_feature(weather_df, window=3)
    add_lag_feature(weather_df, window=72)

    inp = inp.merge(weather_df, how='left', on=['site_id', 'timestamp'])

    return inp


with ut.tick_tock('read_data'):
    print nrows
    train_base = pd.read_csv(cst.train_prefix + "deco_base.csv", ',', nrows=nrows)
    test_base = pd.read_csv(cst.test_prefix + "deco_base.csv", ',', nrows=nrows)

    train_key = pd.read_csv(cst.train_prefix + "key.csv", ',', nrows=nrows)
    test_key = pd.read_csv(cst.test_prefix + "key.csv", ',', nrows=nrows)

    weather_train_df = pd.read_csv(cst.data_root + "weather_train.csv", ',', nrows=nrows)
    weather_test_df = pd.read_csv(cst.data_root + "weather_test.csv", ',', nrows=test_nrows)

    ori_cols = train_base.columns.tolist()  # others use base
    train_index = len(train_key)
    feature = pd.concat([train_base, test_base])
    weather_df = pd.concat([weather_train_df, weather_test_df])

with ut.tick_tock('cal fea'):
    out = feature_func(feature, weather_df)
    # no_cols = "V194,V173,V216,V213,V134,V125,V175,V40,V123,V253,V161,V181,V63,V84,V286,V197,V200,V16,V72,V304,id_34,V250,V243,V57,V336,V306,V22,V242,V179,V158,V255,V331,V154,V46,V47,V178,V202,V211,V32,V60,V183,V320,V177,id_04,V248,V59,V283,V227,V259,id_03,V245,id_15,V222,V18,V192,V214,V234,V229,V239,V333,V190,V204,V294,V2,V3,id_11,V318,V71,D7"
    # no_cols = no_cols.split(",")
    out_cols = list(set(out) - set(ori_cols))
    print out_cols

    ut.check(feature,out)
with ut.tick_tock("write data"):
    feat_key = key
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv.gz', index=False,
                                           header=ut.deco_outcols(feat_key, out_cols),compression='gzip', chunksize=100000)
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv.gz', index=False,
                                           header=ut.deco_outcols(feat_key, out_cols),compression='gzip', chunksize=100000)
