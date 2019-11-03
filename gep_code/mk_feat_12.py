import pandas as pd
import numpy as np
import math
import emc_lib.feature_lib as flb
import emc_lib.const as cst
import emc_lib.parse_cmd as pcmd
import emc_lib.utils as ut
import datetime

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

key = '12'

# lower by 0.003

def feature_func(out, feature):
    weidu = ['card_id']
    # for i in ['month', 'hour', 'weekofyear', 'dayofweek', 'year', 'subsector_id', 'merchant_id',
    #           'merchant_category_id']:
    #     out = flb.feat_nunique(out, feature, weidu, i)

    feature.sort_values(['card_id', 'purchase_date'], ascending=True, inplace=True)
    feature['purchase_data_diff'] = (feature['purchase_date'] -
                                     feature[['card_id', 'purchase_date']].groupby('card_id')[
                                         'purchase_date'].shift(1)).astype(np.int64) // 1e9 // 60 // 60

    feature = feature[feature.purchase_data_diff > -2562048]

    i = 'purchase_data_diff'
    out = flb.feat_sum(out, feature, weidu, i)
    out = flb.feat_max(out, feature, weidu, i)
    out = flb.feat_min(out, feature, weidu, i)
    out = flb.feat_mean(out, feature, weidu, i)
    out = flb.feat_median(out, feature, weidu, i)
    out = flb.feat_skew(out, feature, weidu, i)
    out = flb.feat_diff_mean(out, feature, weidu, i)

    return out


with ut.tick_tock('read_data'):
    hist_tran = pd.read_csv(cst.data_root + "deco_his_trans.csv", ',', nrows=nrows, parse_dates=['purchase_date'])

    print hist_tran.head()
    train_key = pd.read_csv(cst.train_prefix + "key.csv", ',', nrows=nrows)
    test_key = pd.read_csv(cst.test_prefix + "key.csv", ',', nrows=nrows)
    train_index = len(train_key)
    ori_columns = train_key.columns
    out = pd.concat([train_key, test_key])

with ut.tick_tock("cal fea"):
    out = feature_func(out, hist_tran)
    rm_feas = ['purchase_date_card_id_min', 'purchase_date_card_id_max']
    out_cols = list(set(out.columns) - set(ori_columns))

with ut.tick_tock("write data"):
    feat_key = key
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False,
                                       header=ut.deco_outcols(feat_key, out_cols))
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False,
                                       header=ut.deco_outcols(feat_key, out_cols))
