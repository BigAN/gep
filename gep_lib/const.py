import os

code_name = 'gep'
train_name = "train_" + code_name
test_name = "test_" + code_name
data_root = '../data/'
stacking_root = '../stacking/'

label = 'meter_reading'
key = ['building_id','meter','timestamp']
key_str = ",".join(key)
train_prefix = os.path.join(data_root,train_name+"_")
test_prefix = os.path.join(data_root,test_name+"_")

print train_prefix,test_prefix