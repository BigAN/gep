# encoding=utf8
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def stack_feature_sum(inp, out, key, weidu, value, func, name_pre=None):
    #     func="sum"
    # print inp.head(50)
    # inp =  inp.head(50)
    # inp = inp.fillna(-999)
    a = inp.groupby(key + weidu).agg({value: func})
    # print "????a"
    # print a.head()
    b = a.unstack()
    # print "..."
    # print b.head(),"b ????"
    # print(b.columns,"b.columns")
    if name_pre:
        b.columns = list(map(lambda x: name_pre + str(x), b.columns))
    else:
        b.columns = b.columns.get_level_values(1)
        # print( b.columns, "????"
        b.columns = list(
            map(lambda x: "_".join(["_".join(key), "_".join(weidu), value]) + "_" + str(x) + "_" + func,
                b.columns))
        # print( b.columns, "????"

    grp = b.reset_index()
    out = out.merge(grp, how='left', on=key)
    return out


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def feat_most_freq(df, df_feature, fe, value, name=""):
    df_count = df_feature.groupby(fe)[value].agg(
        {value: {"mf_{}".format(value): lambda x: x.value_counts().index[0]}})

    df_count.columns = df_count.columns.get_level_values(1)
    df_count = df_count.reset_index()

    if not name:
        df_count.columns = fe + [value + "_%s_most_freq" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]

    df = df.merge(df_count, on=fe, how="left").fillna(-999)

    return df


def encode_onehot(df, column_name):
    feature_df = pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1), feature_df], axis=1)
    return all


def encode_count(df, column_name):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df


def merge_count(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_nunique(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_median(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_mean(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_sum(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_max(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_min(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_std(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_var(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].var()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add, on=columns, how="left")
    return df


def feat_count(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_count" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_nunique(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].nunique()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_nunique" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_mean(df, df_feature, fe, value, name=""):
    # print( df_feature.head()
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_mean_tmp(df, df_feature, fe, value, name=""):
    # print( df_feature.head()
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df,df_count


def feat_kernelMedian(df, df_feature, fe, value, pr, name=""):
    def get_median(a, pr=pr):
        a = np.array(a)
        x = a[~np.isnan(a)]
        n = len(x)
        weight = np.repeat(1.0, n)
        idx = np.argsort(x)
        x = x[idx]
        if n < pr.shape[0]:
            pr = pr[n, :n]
        else:
            scale = (n - 1) / 2.
            xxx = np.arange(-(n + 1) / 2. + 1, (n + 1) / 2., step=1) / scale
            yyy = 3. / 4. * (1 - xxx ** 2)
            yyy = yyy / np.sum(yyy)
            pr = (yyy * n + 1) / (n + 1)
        ans = np.sum(pr * x * weight) / float(np.sum(pr * weight))
        return ans

    df_count = pd.DataFrame(df_feature.groupby(fe)[value].apply(get_median)).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_kernel_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_std(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].std()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_std" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df

def feat_last(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].last()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_last" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_median(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].median()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_median" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_max(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]

    df = df.merge(df_count, on=fe, how="left").fillna(-999)

    return df


def feat_skew(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].skew()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_skew" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]

    df = df.merge(df_count, on=fe, how="left").fillna(-999)

    return df


def feat_kurt(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].apply(pd.DataFrame.kurt)).reset_index()
    if not name:

        df_count.columns = fe + [value + "_%s_kurt" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]

    df = df.merge(df_count, on=fe, how="left").fillna(-999)

    return df


def feat_min(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].min()).reset_index()
    # print(  fe, value
    if not name:
        # print( fe, value
        df_count.columns = fe + [value + "_%s_min" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df

def feat_ptp(df, df_feature, fe, value, name=""):
    group =df_feature.groupby(fe)[value]
    df_count = pd.DataFrame(group.max() - group.min()).reset_index()
    # print(  fe, value
    if not name:
        # print( fe, value
        df_count.columns = fe + [value + "_%s_ptp" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df

def feat_ptm(df, df_feature, fe, value, name=""):
    group = df_feature.groupby(fe)[value]
    df_count = pd.DataFrame(group.max() - group.median()).reset_index()
    # print(  fe, value
    if not name:
        # print( fe, value
        df_count.columns = fe + [value + "_%s_ptm" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df

def feat_ptq(df, df_feature, fe, value, name=""):
    group = df_feature.groupby(fe)[value]
    df_count = pd.DataFrame(group.max() - group.quantile(0.9)).reset_index()
    # print(  fe, value
    if not name:
        # print( fe, value
        df_count.columns = fe + [value + "_%s_ptq" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_sum(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].sum()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_sum" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]

    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_size(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].size()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_size" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]

    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_diff_mean(df, df_feature, fe, value, name=""):
    # print "???",df_feature.groupby(fe)[value].diff()
    # print df_feature.head()
    # print "--------",df_feature.sort_values(by=fe+["partition_date"]).head()
    df_feature['diff'] = - df_feature.groupby(fe)[value].diff()
    # print df_feature.sort_values(by=fe+["partition_date"]).head()
    df_count = pd.DataFrame(df_feature.groupby(fe)["diff"].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_diff_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_diff_median(df, df_feature, fe, value, name=""):
    # print "???",df_feature.groupby(fe)[value].diff()
    # print df_feature.head()
    # print "--------",df_feature.sort_values(by=fe+["partition_date"]).head()
    df_feature['diff'] = - df_feature.groupby(fe)[value].diff()
    # print df_feature.sort_values(by=fe+["partition_date"]).head()
    df_count = pd.DataFrame(df_feature.groupby(fe)["diff"].median()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_diff_median" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_diff_encode(df, df_feature, fe, value, name="", index=4):
    # print "???",df_feature.groupby(fe)[value].diff()
    # print df_feature.head()
    # print "--------",df_feature.sort_values(by=fe+["partition_date"]).head()
    df_feature['diff_encode'] = df_feature.groupby(fe)[value].diff().apply(lambda x: 1 if x > 0 else 0)
    # print df_feature.sort_values(by=fe+["partition_date"]).head()
    rs = df_feature[df_feature.date_gap_7 != 0].groupby(fe).diff_encode.apply(
        lambda x: int("".join(map(str, x)[:index]), 2))
    df_count = pd.DataFrame(rs).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_diff_encode" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_diff_max(df, df_feature, fe, value, name=""):
    # print "???",df_feature.groupby(fe)[value].diff()
    # print df_feature.head()
    # print "--------",df_feature.sort_values(by=fe+["partition_date"]).head()
    df_feature['diff'] = df_feature.groupby(fe)[value].diff()
    # print df_feature.sort_values(by=fe+["partition_date"]).head()
    df_count = pd.DataFrame(df_feature.groupby(fe)["diff"].max()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_diff_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_diff_min(df, df_feature, fe, value, name=""):
    # print "???",df_feature.groupby(fe)[value].diff()
    # print df_feature.head()
    # print "--------",df_feature.sort_values(by=fe+["partition_date"]).head()
    df_feature['diff'] = df_feature.groupby(fe)[value].diff()
    # print df_feature.sort_values(by=fe+["partition_date"]).head()
    df_count = pd.DataFrame(df_feature.groupby(fe)["diff"].min()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_diff_min" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_diff_skew(df, df_feature, fe, value, name=""):
    # print "???",df_feature.groupby(fe)[value].diff()
    # print df_feature.head()
    # print "--------",df_feature.sort_values(by=fe+["partition_date"]).head()
    df_feature['diff'] = df_feature.groupby(fe)[value].diff()
    # print df_feature.sort_values(by=fe+["partition_date"]).head()
    df_count = pd.DataFrame(df_feature.groupby(fe)["diff"].skew()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_diff_skew" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_diff_std(df, df_feature, fe, value, name=""):
    df_feature['diff'] = df_feature.groupby(fe)[value].diff()
    # print df_feature.sort_values(by=fe+["partition_date"]).head()
    df_count = pd.DataFrame(df_feature.groupby(fe)["diff"].std()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_diff_std" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_var(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].var()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_var" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def feat_quantile(df, df_feature, fe, value, n, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].quantile(n)).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_quantile" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(-999)
    return df


def interaction_features(df, fea1, fea2):
    df['inter_{}*{}'.format(fea1, fea2)] = df[fea1] * df[fea2]
    df['inter_{}/{}'.format(fea1, fea2)] = df[fea1] / df[fea2]
    df['inter_{}+{}'.format(fea1, fea2)] = df[fea1] + df[fea2]
    df['inter_{}-{}'.format(fea1, fea2)] = df[fea1] - df[fea2]

    # test['inter_{}*'.format(prefix)] = test[fea1] * test[fea2]
    # test['inter_{}/'.format(prefix)] = test[fea1] / test[fea2]

    return df


PrOriginalEp = np.zeros((2000, 2000))
PrOriginalEp[1, 0] = 1
PrOriginalEp[2, range(2)] = [0.5, 0.5]
for i in range(3, 2000):
    scale = (i - 1) / 2.
    x = np.arange(-(i + 1) / 2. + 1, (i + 1) / 2., step=1) / scale
    y = 3. / 4. * (1 - x ** 2)
    y = y / np.sum(y)
    PrOriginalEp[i, range(i)] = y
PrEp = PrOriginalEp.copy()
for i in range(3, 2000):
    PrEp[i, :i] = (PrEp[i, :i] * i + 1) / (i + 1)
