import pandas as pd
from pandas.io.json import json_normalize
import json
import os
import numpy as np
import gep_lib.utils as ut
import gep_lib.const as cst
import datetime

import gep_lib.parse_cmd as pcmd

a = pd.read_csv('../data/test_sctp_ori.csv')
b = pd.read_csv('../data/real_id.csv')

print len(a)
c = a.merge(b,how='inner',on = cst.key)
print len(c)

print c[cst.key].head()
c.to_csv()

with ut.tick_tock("write data"):
    c.to_csv(cst.test_prefix + "real" + '.csv', index=False)
