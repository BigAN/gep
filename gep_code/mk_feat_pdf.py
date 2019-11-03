import pandas as pd
import gep_lib.utils as ut
import gep_lib.feature_lib as flb
import gep_lib.const as cst
# nrows = None
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import gep_lib.parse_cmd as pcmd
import utils as cut
from sklearn.neighbors import KernelDensity
from operator import itemgetter

# nrows = 10**3
key = "catcount3"
args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

from sklearn.model_selection import KFold
# fold_rs = folds.split(train, train[cst.label])
def calculate_pdf_difference(feat, df_feature,df_test, df_target, IQR_multiplier, bin_bandwidth_multiplier, print_number_bins):
    # Agreggating feature values in bin format using the Freedman-Diaconis rule
    IQR = df_feature[feat].quantile([0.75]).values - df_feature[feat].quantile(
        [0.25]).values  # Interquartile range (IQR)

    n = len(df_feature[feat])
    bin_size = IQR_multiplier * IQR / n ** (1 / 3)
    bin_number = int(np.round((df_feature[feat].max() - df_feature[feat].min()) / bin_size))
    #     bin_number = 50
    binvalues = pd.cut(df_feature[feat],retbins=True, bins=bin_number, labels=range(bin_number)).astype('float')
    binvalues = pd.cut(df_test[feat], bins=bin_number, labels=range(bin_number)).astype('float')

    print "IQR", IQR, "n", n, "bin_size", bin_size, "bin_number", bin_number, "binvalues", binvalues

    if print_number_bins:
        print('There are {} bins in the feature {}'.format(bin_number, feat))

    # Calculate the PDFs using the df_target
    pdf_0 = KernelDensity(kernel='gaussian', bandwidth=bin_size * bin_bandwidth_multiplier)
    pdf_0.fit(np.array(df_target[feat][df_target['target'] == 0]).reshape(-1, 1))

    pdf_1 = KernelDensity(kernel='gaussian', bandwidth=bin_size * bin_bandwidth_multiplier)
    pdf_1.fit(np.array(df_target[feat][df_target['target'] == 1]).reshape(-1, 1))

    # Creates an X array with the average feature value for each bin
    x = np.array(np.arange(min(df_feature[feat]) + bin_size / 2, max(df_feature[feat]), bin_size)).reshape(-1, 1)

    # gets the pdf values based on the X array
    log_pdf_0 = np.exp(pdf_0.score_samples(x))
    log_pdf_1 = np.exp(pdf_1.score_samples(x))

    # creates a dictionary that links the bin number with the PDFs value difference
    pdf_dict = dict()
    for i in range(bin_number):
        pdf_dict[i] = log_pdf_1[i] - log_pdf_0[i]

        # gets the PDF difference for each row of the dataset based on its equivalent bin.
    bin_pdf_values = np.array(itemgetter(*list(binvalues))(pdf_dict))

    return bin_pdf_values

def feature_func(out, feature):
    folds = KFold(n_splits=5, shuffle=True, random_state=4590)
    fold_rs = folds.split(feature, feature[cst.label])
    for i, (train_fold, validate) in enumerate(fold_rs):
        cat_feas = [x for x in out.columns.tolist() if "var_" in x]
        for i in cat_feas:
            weidu = [i]
            out = flb.feat_count(out, feature, weidu, 'ID_code')

    return out



with ut.tick_tock("read data"):
    # use = ['deal_id', 'poi_id', 'barea_id']
    train_base = pd.read_csv(cst.train_prefix + "ori.csv", ',', nrows=nrows)
    test_base = pd.read_csv(cst.test_prefix + "ori.csv", ',', nrows=nrows)
    test_real = pd.read_csv(cst.test_prefix + "real.csv", ',', nrows=nrows)

    train_index = len(train_base)

    ori_columns = train_base.columns
    out = pd.concat([train_base, test_base])
    feature = pd.concat([train_base, test_real])
    # out = base[
    #     ["ID_code","var_12","target"]]

    # for i in [["month", "elapsed_time_bin"], ["feature_2", "elapsed_time_bin"], ["feature_3", "elapsed_time_bin"]]:
    #     out["_".join(i)] = out.apply(lambda x: str(x[i[0]]) + "_" + str(x[i[1]]),axis=1)

    print out.columns

    ori_columns = out.columns

with ut.tick_tock("gene fea"):
    out = feature_func(out, feature)

    new_columns = out.columns

with ut.tick_tock("write feas"):
    out_cols = list(set(new_columns) - set(ori_columns))
    print out_cols
    print out.shape

    feat_key = key
    print out.head()
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False,
                                       header=ut.deco_outcols(feat_key, out_cols))
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False,
                                       header=ut.deco_outcols(feat_key, out_cols))

    # out_train[out_cols].to_csv("../data/train_emc_train_tgt_encode.csv", index=False)
    # out_test[out_cols].to_csv("../data/test_tgt_encode.csv", index=False)
