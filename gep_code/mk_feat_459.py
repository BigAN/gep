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

key = 'base'

import gep_lib.utils as ut


def feature_func(inp):
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

    ori_cols = train_key.columns.tolist()  # others use base
    train_index = len(train_key)
    feature = pd.concat([train_base, test_base])

with ut.tick_tock('cal fea'):
    out = feature_func(feature)
    out_cols = list(set(feature) - set(ori_cols))

    print out_cols

with ut.tick_tock("write data"):
    feature[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False)
    feature[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False)
