import pandas as pd
from pandas.io.json import json_normalize
import json
import os
import numpy as np
import gep_lib.utils as ut
import gep_lib.const as cst
import datetime

import gep_lib.parse_cmd as pcmd

args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows


def func_feature(inp,out):
    inp = inp.merge(out,on=cst.key,how='left')
    return inp


with ut.tick_tock('read_data'):
    train = pd.read_csv(cst.data_root+"train_transaction.csv", ',', nrows=nrows)
    test = pd.read_csv(cst.data_root+"test_transaction.csv", ',', nrows=test_nrows)

    train_index = len(train)

    base = pd.concat([train, test])



with ut.tick_tock('cal fea'):
    out = base[['TransactionID']]


with ut.tick_tock('write data'):
    print out.columns
    out[:train_index].to_csv(cst.train_prefix + 'key.csv', index=False)
    out[train_index:].to_csv(cst.test_prefix + 'key.csv', index=False)
