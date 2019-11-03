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

key = 'uid_stat2'
print key
import gep_lib.utils as ut


def feature_func(out,feature):
    to_w2v = filter(lambda x:feature[x].dtype=='object',feature.columns)
    print to_w2v,"to_w2v"
    # for f in fea.columns:
    #     if inp[f].dtype == 'object' or inp[f].dtype == 'object':

    for weidu in [['card1'], ['card2'], ['card3'], ['card5'], ['uid'], ['uid2'], ['uid3']]:
        for i in ['TransactionAmt','V258','C14','V201']:
            out = flb.feat_max(out, feature, weidu, i)
            out = flb.feat_min(out, feature, weidu, i)
            out = flb.feat_std(out, feature, weidu, i)
            out = flb.feat_sum(out, feature, weidu, i)
            out = flb.feat_count(out, feature, weidu, i)
            out = flb.feat_skew(out, feature, weidu, i)
            # out = flb.feat_kernelMedian(out,feature,weidu,i,name='kermean')
            out = flb.feat_kernelMedian(out, feature, weidu, i, ut.PrEp,
                                         'cross_{}_kernel_median_{}'.format("_".join(weidu), i))

    return out


with ut.tick_tock('read_data'):
    print nrows
    train_key = pd.read_csv(cst.train_prefix + "deco_base2.csv", ',', nrows=nrows)
    test_key = pd.read_csv(cst.test_prefix + "deco_base2.csv", ',', nrows=nrows)

    # train_key = pd.read_csv(cst.train_prefix + "key.csv", ',', nrows=nrows)
    # test_key = pd.read_csv(cst.test_prefix + "key.csv", ',', nrows=nrows)

    ori_cols = train_key.columns.tolist()  # others use base
    train_index = len(train_key)
    feature = pd.concat([train_key, test_key])
    out = pd.concat([train_key, test_key])
    use_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3']
    out = out[use_cols]

with ut.tick_tock('cal fea'):
    out = feature_func(out,feature)
    out_cols = list(set(out) - set(ori_cols))

    print out_cols

with ut.tick_tock("write data"):
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False,float_format='%.4f')
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False,float_format='%.4f')

    import pandas as pd
    import numpy as np
    import math
    import emc_lib.feature_lib as flb
    import emc_lib.const as cst
    import emc_lib.parse_cmd as pcmd
    import emc_lib.utils as ut
    import datetime
    import multiprocessing as mp

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    import glob
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    import multiprocessing

    from gensim.models import Word2Vec

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

    args = pcmd.init_arguments_feature()

    nrows = args.nrows
    test_nrows = nrows

    key = 'w2v'
    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer


    # df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    #     df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
    #     for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\
    #                      'new_hist_purchase_date_min']:
    #         df[f] = df[f].astype(np.int64) * 1e-9
    #     df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']
    #     df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']

    def read_one(file_):
        df = pd.read_csv(file_, ',', nrows=nrows)
        for i in ['purchase_date', 'first_active_month']:
            df[i] = pd.to_datetime(df[i])
        # print df.dtypes
        return df


    def read_data(path):
        allFiles = glob.glob(path + "wholetran_*")
        list_ = []

        list_ = pool.map(read_one, allFiles)
        #     for file_ in allFiles:
        #         df = pd.read_csv(file_, index_col=None, names=['label','uuid','poi_id','tag_id','length','mid_his','cat_his','mask'],nrows=100,sep='\t')
        #         list_.append(df)

        frame = pd.concat(list_, axis=0, ignore_index=True)
        return frame


    with ut.tick_tock('read_data'):
        pool = mp.Pool(12)
        agg_key = ['feature_1', "feature_2", 'feature_3']
        hist_tran = read_data(cst.data_root)
        # hist_tran = pd.read_csv(cst.data_root + "deco_new_his_trans.csv", ',', nrows=nrows, parse_dates=['purchase_date'])
        # hist_tran = pd.read_csv(cst.data_root + "deco_his_trans.csv", ',', nrows=nrows,
        #                         parse_dates=['purchase_date', 'first_active_month'])
        print hist_tran.dtypes, "hist_tran.dtypes"

        print hist_tran.head()

        train_key = pd.read_csv(cst.train_prefix + "deco_base.csv", ',', nrows=nrows)
        test_key = pd.read_csv(cst.test_prefix + "deco_base.csv", ',', nrows=nrows)
        train_index = len(train_key)
        ori_columns = train_key.columns
        out = pd.concat([train_key, test_key])
        use_cols = ["card_id"] + agg_key
        out = out[use_cols]
        feature = hist_tran

        print out.head()

    with ut.tick_tock("gene fea"):

        for i in ['month_lag']:
            feature[i] = feature[i].astype(int).astype(str)

        a = feature.groupby(agg_key).apply(lambda x: " ".join(x.month_lag))
        # n = 50
        vectorizer = CountVectorizer(min_df=3)
        values = vectorizer.fit_transform(a.values)

        w2v_model = Word2Vec(min_count=20,
                             window=2,
                             size=50,
                             sample=6e-5,
                             alpha=0.03,
                             min_alpha=0.0007,
                             negative=20,
                             workers=cores - 1)
        w2v_model.build_vocab(sentences, progress_per=10000)
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

        n_components = 6
        lda = LatentDirichletAllocation(n_components=n_components, learning_method='batch', random_state=71, n_jobs=-1)
        components = lda.fit_transform(values)1

        df_cp = pd.DataFrame(components, columns=["lda_" + str(x) for x in range(0, n_components)])
        df_cp.reset_index(drop=True, inplace=True)
        fea = feature[agg_key].drop_duplicates().reset_index(drop=True)

        print df_cp.head(), "before"

        df_cp = pd.concat([df_cp, fea], axis=1)
        print df_cp.head(), "df_cp"
        print out.head(), "out"
        out = out.merge(df_cp, on=agg_key, how='left')

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

