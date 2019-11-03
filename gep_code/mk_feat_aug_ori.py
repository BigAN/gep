import pandas as pd
import numpy as np
import math
import gep_lib.feature_lib as flb
import gep_lib.const as cst
import gep_lib.parse_cmd as pcmd

args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

key = 'aug_ori'

import gep_lib.utils as ut

with ut.tick_tock('read_data'):
    print nrows
    train_base = pd.read_csv(cst.train_prefix + "ori.csv", ',', nrows=nrows)
    test_base = pd.read_csv(cst.test_prefix + "ori.csv", ',', nrows=nrows)

    train_key = pd.read_csv(cst.train_prefix + "key.csv", ',', nrows=nrows)
    test_key = pd.read_csv(cst.test_prefix + "key.csv", ',', nrows=nrows)

    ori_cols = train_key.columns.tolist()  # others use base
    train_index = len(train_key)
    # feature = pd.concat([train_base, test_base])
    out_cols = train_base.columns.tolist()
    out_cols.remove(cst.label)
    out_cols.remove(cst.key)

    print out_cols, 'out cols'
    train_ori, train_y = ut.augment(train_base[out_cols].values, train_base[cst.label].values)

    train_ori = pd.DataFrame(train_ori, columns=out_cols)
    train_ori[cst.label] = train_y
    new_count = len(train_ori) - len(train_key)
    keys = np.concatenate([train_key[cst.key].values, ["aug_" + str(x) for x in range(new_count)]])
    train_ori[cst.key] = keys
    print train_ori.head()
    print train_ori.tail()
    print type(train_ori)
    print len(train_base), len(train_y)
    print len(train_ori), len(train_y)

    out_cols = list(set(train_ori.columns) - set(train_key.columns))
#
#     print out_cols
#
with ut.tick_tock("write data"):
    train_ori.to_csv(cst.train_prefix + key + '.csv', index=False)
    test_base.to_csv(cst.test_prefix + key + '.csv', index=False)
