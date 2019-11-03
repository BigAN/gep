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

key = 'uid_stat2'
print key
import gep_lib.utils as ut


def feature_func(out,feature):

    for weidu in [['card1'], ['card2'], ['card3'], ['card5'], ['uid'], ['uid2'], ['uid3']]:
        for i in ['TransactionAmt','V258','C14','V201']:
            out = flb.feat_max(out, feature, weidu, i)
            out = flb.feat_min(out, feature, weidu, i)
            out = flb.feat_std(out, feature, weidu, i)
            out = flb.feat_sum(out, feature, weidu, i)
            out = flb.feat_count(out, feature, weidu, i)
            out = flb.feat_skew(out, feature, weidu, i)
            # out = flb.feat_kernelMedian(out,feature,weidu,i,name='kermean')
            out = flb.feat_kernelMedian(out, feature, weidu, i, ut.PrEp,
                                         'cross_{}_kernel_median_{}'.format("_".join(weidu), i))

    return out


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
    use_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3']
    out = out[use_cols]

with ut.tick_tock('cal fea'):
    out = feature_func(out,feature)
    out_cols = list(set(out) - set(ori_cols))

    print out_cols

with ut.tick_tock("write data"):
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False,float_format='%.4f')
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False,float_format='%.4f')
