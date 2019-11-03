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
    weather_train_df = pd.read_csv(cst.data_root+"weather_train.csv", ',', nrows=nrows)
    weather_test_df = pd.read_csv(cst.data_root+"weather_test.csv", ',', nrows=test_nrows)

    weather_train_df = weather_train_df.groupby('site_id').apply(
        lambda group: group.interpolate(limit_direction='both'))

    # ori_columns = train.columns.tolist()
    # print ori_columns
    train_index = len(train)
    base = pd.concat([train, test])
    id = pd.concat([trid,tid])



with ut.tick_tock('cal fea'):
    out = func_feature(base, id)

with ut.tick_tock('write data'):
    print out.columns
    out[:train_index].to_csv(cst.train_prefix + 'deco_base.csv', index=False)
    out[train_index:].to_csv(cst.test_prefix + 'deco_base.csv', index=False)
