import pandas as pd
import numpy as np
import math
import gep_lib.feature_lib as flb
import gep_lib.const as cst
import gep_lib.parse_cmd as pcmd
import gep_lib.utils as ut
import datetime
import multiprocessing as mp

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
import glob
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows

key = 'svd'



with ut.tick_tock("read data"):
    train_base = pd.read_csv(cst.train_prefix + "ori.csv", ',', nrows=nrows)
    test_base = pd.read_csv(cst.test_prefix + "ori.csv", ',', nrows=nrows)
    test_real = pd.read_csv(cst.test_prefix + "real.csv", ',', nrows=nrows)

    train_index = len(train_base)

    ori_columns = train_base.columns
    out = pd.concat([train_base, test_base])
    feature = pd.concat([train_base, test_real])
    feature = feature.reset_index()
    print out.columns

    ori_columns = out.columns

with ut.tick_tock("gene fea"):
    print len(feature),"feature length"
    feature = feature.astype(str)
    all_cols = filter(lambda x:"var_" in x,feature.columns.tolist())
    print "feature",feature.columns.tolist()
    print "all_cols",all_cols
    feature['all'] = feature[all_cols].apply(lambda x: x.add(' ')).sum(axis=1).str.strip()
    print feature.head()
    a = feature['all']
    key_df = feature[[cst.key]]

    print "abs????"
    print a[0],"???"
    tfidf = TfidfVectorizer(ngram_range=(1, 1),min_df= 1,token_pattern=r"(?u)\b\w+\b")
    # print a.values
    values = tfidf.fit_transform(a.values)
    #
    # print values
    # print tfidf.get_feature_names()
    print "len(tfidf.get_feature_names())",len(tfidf.get_feature_names())

    print values[:10]

    # tfidf = CountVectorizer( min_df=1,token_pattern=r"(?u)\b\w+\b")
    # values = tfidf.fit_transform(a.values)
    # print values

    n_components = 100
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    components = svd.fit_transform(values)

    df_cp = pd.DataFrame(components, columns=["svd_" + str(x) for x in range(0, n_components)])
    df_cp.reset_index(drop=True, inplace=True)

    df_cp[cst.key] = key_df
    out = out.merge(df_cp, on=cst.key, how='left')

    new_columns = out.columns

    out_cols = list(set(out.columns) - set(ori_columns))
    # out_cols = out.columns.tolist()
    # out_cols += ['card_id']
    # print out_cols
    with ut.tick_tock("write data"):
        feat_key = key

    print out.head()
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False,
                                       header=ut.deco_outcols(feat_key, out_cols))
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False,
                                       header=ut.deco_outcols(feat_key, out_cols))
