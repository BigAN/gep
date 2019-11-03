import pandas as pd
import numpy as np
import math
import gep_lib.feature_lib as flb
import gep_lib.const as cst
import gep_lib.parse_cmd as pcmd

# nrows = 10000
key = "inter"
args = pcmd.init_arguments_feature()

nrows = args.nrows
test_nrows = nrows
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


def deco_deal(df):
    def sp(x):
        # i
        # print x,type(x),x==np.nan
        if isinstance(x, str):
            return x.split("|")[0]
        else:
            return 99

    df['dish_tag'] = df.dish_tag.apply(sp)

    return df


import gep_lib.utils as ut

with ut.tick_tock("read data"):
    # use = ['deal_id', 'poi_id', 'barea_id']
    train_base = pd.read_csv(cst.train_prefix + "ori.csv", ',', nrows=nrows)
    test_base = pd.read_csv(cst.test_prefix + "ori.csv", ',', nrows=nrows)

    train_index = len(train_base)

    ori_columns = train_base.columns
    out = pd.concat([train_base, test_base])
    base = out
    # out = base[
    #     ["ID_code","var_12","target"]]

    # for i in [["month", "elapsed_time_bin"], ["feature_2", "elapsed_time_bin"], ["feature_3", "elapsed_time_bin"]]:
    #     out["_".join(i)] = out.apply(lambda x: str(x[i[0]]) + "_" + str(x[i[1]]),axis=1)

    print out.columns

    ori_columns = out.columns
# deal_id,poi_id,sales,market_price,price,deal_max_num,deal_max_num,deal_avg_num,time_available,day_unavailable
# weekday_unavailable,begin_date,deal_max_num,beg_weekday,day,month,day2,days_to_side,open_hours,is_mid,is_night,is_midnight,av_5,av_6,av_7,av_days,mt_poi_cate2_name,price_person,has_parking_area,has_booth,is_dining,barea_id,mt_score,dp_score,dp_evn_score,dp_taste_score,dp_service_score,dp_star,poi_zlf,poi_rank
with ut.tick_tock('cal fea'):
    deal_poi = base[["ID_code"]]

    ori_columns = base.columns

    deal_poi.head()
    from itertools import combinations

    for e, (x, y) in enumerate(combinations(
            ['var_81', 'var_139', 'var_12', 'var_6', 'var_53', 'var_110', 'var_146',
             'var_26', 'var_174', 'var_99', 'var_166', 'var_76', 'var_80', 'var_165',
             'var_21', 'var_22'], 2)):
        base = flb.interaction_features(base, x, y)

    assert len(base) == len(deal_poi)
    deal_poi = deal_poi.merge(base)
    new_columns = deal_poi.columns

    out_cols = sorted(list(set(new_columns) - set(ori_columns)))

# print new_columns
# print ori_columns
print out_cols
#
# # new_fea = [['discount',]]
with ut.tick_tock("write data"):
    # fea_key = "intersect"
    print "len(out_cols)", len(out_cols)
    feat_key = key
    out[:train_index][out_cols].to_csv(cst.train_prefix + key + '.csv', index=False,
                                       header=ut.deco_outcols(feat_key, out_cols))
    out[train_index:][out_cols].to_csv(cst.test_prefix + key + '.csv', index=False,
                                       header=ut.deco_outcols(feat_key, out_cols))

