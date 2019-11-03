import pandas as pd
import gep_lib.utils as ut
import gep_lib.feature_lib as flb
import gep_lib.const as cst
# nrows = None
import gep_lib.parse_cmd as pcmd

# nrows = 10**3
key = "tgt"
args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows




with ut.tick_tock("read data"):
    # use = ['deal_id', 'poi_id', 'barea_id']
    train_base = pd.read_csv(cst.train_prefix + "ori.csv", ',', nrows=nrows)
    test_base = pd.read_csv(cst.test_prefix + "ori.csv", ',', nrows=nrows)

    train_index = len(train_base)

    ori_columns = train_base.columns
    base = pd.concat([train_base, test_base])

    out = base[
        ["var_12","target"]]

    # for i in [["month", "elapsed_time_bin"], ["feature_2", "elapsed_time_bin"], ["feature_3", "elapsed_time_bin"]]:
    #     out["_".join(i)] = out.apply(lambda x: str(x[i[0]]) + "_" + str(x[i[1]]),axis=1)

    print out.columns
    out_train = out[:train_index]
    out_test = out[train_index:]

    ori_columns = out.columns

with ut.tick_tock("gene fea"):
    for f in ["var_12"
        # , "weekofyear", "month", "elapsed_time_bin", "feature_2", "feature_3", "feature_1",
              # "_".join(["month", "elapsed_time_bin"]),
              # "_".join(["feature_2", "elapsed_time_bin"]),
              # "_".join(["feature_3", "elapsed_time_bin"])
              ]:
        out_train[f + "_tef"], out_test[f + "_tef"] = flb.target_encode(trn_series=out_train[f],
                                                                        tst_series=out_test[f],
                                                                        target=out_train['target'],
                                                                        min_samples_leaf=10,
                                                                        smoothing=3,
                                                                        noise_level=0.03)

    new_columns = out_train.columns

with ut.tick_tock("write feas"):
    out_cols = list(set(new_columns) - set(ori_columns))
    print out_cols
    print out.shape

    feat_key = key
    print out.head()
    out_train[out_cols].to_csv(cst.train_prefix + key + '.csv', index=False,
                               header=ut.deco_outcols(feat_key, out_cols))
    out_test[out_cols].to_csv(cst.test_prefix + key + '.csv', index=False,
                              header=ut.deco_outcols(feat_key, out_cols))

    # out_train[out_cols].to_csv("../data/train_emc_train_tgt_encode.csv", index=False)
    # out_test[out_cols].to_csv("../data/test_tgt_encode.csv", index=False)
