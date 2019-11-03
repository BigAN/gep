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

key = "deco_base"

print test_nrows,"test_row"
def func_feature(inp,out):
    inp = inp.merge(out,on="site_id",how='left')
    return inp


with ut.tick_tock('read_data'):
    train_df = pd.read_csv(cst.data_root+"train.csv", ',', nrows=nrows)
    test_df = pd.read_csv(cst.data_root+"test.csv", ',', nrows=test_nrows)

    metadata_df = pd.read_csv(cst.data_root + "building_metadata.csv", ',', nrows=nrows)

    weather_train_df = pd.read_csv(cst.data_root + "weather_train.csv", ',', nrows=nrows)
    weather_test_df = pd.read_csv(cst.data_root + "weather_test.csv", ',', nrows=test_nrows)

with ut.tick_tock('process data'):

    weather_train_df = weather_train_df.groupby('site_id').apply(
        lambda group: group.interpolate(limit_direction='both'))

    train_df['meter_reading'] = np.log1p(train_df['meter_reading'])

    # ori_columns = train.columns.tolist()
    # print ori_columns
    train_index = len(train_df)
    base = pd.concat([train_df, test_df])
    weather = pd.concat([weather_train_df,weather_test_df])

    print len(weather),"len("


with ut.tick_tock('cal fea'):
    print len(base), 'len(base)'

    out = base.merge(metadata_df,how='left',on='building_id')
    print len(out), 'len(out)'
    out = out.merge(weather, how='left', on=['site_id','timestamp'])
    print out
    print len(out), 'len(out)'

    print len(out),'len(out)'


with ut.tick_tock('write data'):
    print out.columns
    out[:train_index].to_csv(cst.train_prefix + 'deco_base.csv', index=False)
    out[train_index:].to_csv(cst.test_prefix + 'deco_base.csv', index=False)
