import pandas as pd
import numpy as np
import math
import gep_lib.feature_lib as flb
import gep_lib.const as cst
import gep_lib.parse_cmd as pcmd
from sklearn import preprocessing
import itertools as it
import gep_lib.utils as ut
import datetime

args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

key = 'cross_test'
print key
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

tgt_cols = [['card1'], ['card2'], ['card3'], ['card5'], ['uid'], ['uid2'], ['uid3'],['uid','ProductCD'], ['card1', 'card2', 'card3']]

uniq_cols = list(set(it.chain(*tgt_cols)))

print uniq_cols,"uniq_cols"

def feature_func(out, feature):
    # weidu = ['card_id']
    # for i in ['month', 'hour', 'weekofyear', 'dayofweek', 'year', 'subsector_id', 'merchant_id',
    #           'merchant_category_id']:
    #     out = flb.feat_nunique(out, feature, weidu, i)

    for weidu in tgt_cols:
        print weidu
        tgt_weidu = weidu + ['TransactionDT']

        feature.sort_values(tgt_weidu, ascending=True, inplace=True)
        feature['{}_diff'.format("_".join(weidu))] = (feature['TransactionDT'] -
                                                      feature[tgt_weidu].groupby(weidu)[
                                                          'TransactionDT'].shift(1)).fillna(-999).astype(np.int64)
        feature['{}_cross_diff'.format("_".join(weidu))] = (feature[tgt_weidu].groupby(weidu)[
                                                                'TransactionDT'].shift(-1) - feature['TransactionDT']
                                                            ).fillna(-999).astype(np.int64)

        # feature = feature[feature.purchase_data_diff > -2562048]

        # i = '{}_diff'.format(",".join(weidu))
        # out = flb.feat_sum(out, feature, weidu, i)
        # out = flb.feat_max(out, feature, weidu, i)
        # out = flb.feat_min(out, feature, weidu, i)
        # out = flb.feat_mean(out, feature, weidu, i)
        # out = flb.feat_median(out, feature, weidu, i)
        # out = flb.feat_skew(out, feature, weidu, i)
        # out = flb.feat_diff_mean(out, feature, weidu, i)
        #
        # i = '{}_cross_diff'.format(",".join(weidu))
        # out = flb.feat_sum(out, feature, weidu, i)
        # out = flb.feat_max(out, feature, weidu, i)
        # out = flb.feat_min(out, feature, weidu, i)
        # out = flb.feat_mean(out, feature, weidu, i)
        # out = flb.feat_median(out, feature, weidu, i)
        # out = flb.feat_skew(out, feature, weidu, i)
        # out = flb.feat_diff_mean(out, feature, weidu, i)
        # print feature.head()
        # out =out.merge(feature,how='left',on=weidu)
        # print out.head(),"out.head()"
    return feature




with ut.tick_tock('read_data'):
    print nrows
    train_key = pd.read_csv(cst.train_prefix + "deco_base2.csv", ',', nrows=nrows)
    test_key = pd.read_csv(cst.test_prefix + "deco_base2.csv", ',', nrows=nrows)

    # train_key = pd.read_csv(cst.train_prefix + "key.csv", ',', nrows=nrows)
    # test_key = pd.read_csv(cst.test_prefix + "key.csv", ',', nrows=nrows)

    ori_cols = train_key.columns.tolist()  # others use base
    train_index = len(train_key)
    feature = pd.concat([train_key, test_key])
    out = pd.concat([train_key, test_key])
    use_cols = uniq_cols
    print use_cols, "user_cols"
    out = out[use_cols]

with ut.tick_tock('cal fea'):
    out = feature_func(out, feature)
    out_cols = list(set(out) - set(ori_cols))

    print out_cols

with ut.tick_tock("write data"):
    feat_key = key

    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False, float_format='%.4f',
                                       header=ut.deco_outcols(feat_key, out_cols))
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False, float_format='%.4f',
                                       header=ut.deco_outcols(feat_key, out_cols))
