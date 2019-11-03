import pandas as pd
import gep_lib.utils as ut
import gep_lib.feature_lib as flb
import gep_lib.const as cst
# nrows = None
import gep_lib.parse_cmd as pcmd

# nrows = 10**3
fake = pd.read_csv('/home/dongjian/work/sctp/data/fake_id.csv')
# fake = pd.read_csv('/home/dongjian/work/sctp/data/fake_id.csv')

def rm_fake(feature):
    key = cst.key
    feature=feature.set_index(key)
    print "fake",fake.index
    print "feature",feature.index
    # feature = feature[~feature..isin(fake.index)]
    feature = feature.merge(fake,how='inner',on = key)
    # feature =feature.reset_index()
    return feature


