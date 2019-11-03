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

key = 'rm_index'

import gep_lib.utils as ut




with ut.tick_tock('read_data'):
    print nrows


    train_prd = pd.read_csv(cst.stacking_root + "lgbm_cis_base_cv_590540_test8410570252823451322_0.97789206668_0.0203279908151_base_rm1_k2_1.csv", ',', nrows=nrows)
    test_prd = pd.read_csv(cst.stacking_root + "lgbm_cis_base_pred_590540_test8410570252823451322_0.97789206668_0.0203279908151_base_rm1_k2_1.csv", ',', nrows=nrows)


with ut.tick_tock('cal fea'):
    thred = 0.99
    tr_index = train_prd[train_prd.isFraud < thred]
    et_index = test_prd[train_prd.isFraud < thred]

    tr_index= tr_index.rename({"isFraud":"isFraud_x"},axis=1)
    et_index = et_index.rename({"isFraud": "isFraud_x"},axis=1)

with ut.tick_tock("write data"):
    tr_index.to_csv(cst.train_prefix + key + '.csv', index=False)
    et_index.to_csv(cst.test_prefix + key + '.csv', index=False)
