import pandas as pd
import numpy as np
import math
import gep_lib.feature_lib as flb
import gep_lib.const as cst
import gep_lib.utils as ut

import gep_lib.parse_cmd as pcmd

args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

key = 'key'


with ut.tick_tock('read_data'):
    print nrows
    train = pd.read_csv(cst.train_prefix + "deco_base.csv", ',', nrows=nrows)
    test = pd.read_csv(cst.test_prefix + "deco_base.csv", ',', nrows=nrows)

    out_cols = cst.key

    train_index = len(train)
    feature = pd.concat([train, test])


with ut.tick_tock("write data"):
    feature[:train_index][out_cols + [cst.label]].to_csv(cst.train_prefix + key + '.csv', index=False)
    feature[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False)
