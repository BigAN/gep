import gep_lib.utils as ut
# encoding utf8
import pandas as pd
from collections import Counter
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import auc
import gep_lib.utils as  ut
from gep_lib import const as cst
import numpy as np
import pandas as pd
from gep_lib.parse_cmd import init_arguments, add_feature, parse_paras, parse_label_tran_cmd

with ut.tick_tock("lgbm"):
    data_root = '../data/'
    args = init_arguments()
    paras = parse_paras(args)
    shift = args.shift
    md5 = args.md5

    label = cst.label
    train = pd.read_csv(data_root + 'train_{}_key.csv'.format(cst.code_name))
    test = pd.read_csv(data_root + 'test_{}_key.csv'.format(cst.code_name))

    NFOLDS = paras.pop('nfolds', 5)
    save_csv = paras.pop("save_csv", 0)
    rm_high = paras.pop('rm_high',0)

    before_len = len(train)
    train, test, predictors, cat = add_feature(args, train, test)

    if rm_high == 1:
        print len(train),"before rm "
        tr_index = pd.read_csv(cst.train_prefix + 'rm_index.csv')
        et_index = pd.read_csv(cst.test_prefix + 'rm_index.csv')

        train = train.merge(tr_index,on='TransactionID',how='inner')
        # test = test.merge(tr_index, on='TransactionID', how='inner')
        print len(train), "before rm2 "

    # cst.code_name = 'outlier'
    # train = train.sample(frac=0.2)
    sample_rate = paras.pop("sample",0.5)
    train_pos = train[train['isFraud'] == 1]
    train_neg = train[train['isFraud'] == 0]
    train_neg = train_neg.sample(frac=sample_rate,random_state=42)
    # train_neg = train_neg.sample(int(train.shape[0] * 0.2), random_state=42)
    train = pd.concat([train_pos, train_neg]).sort_index()
    with ut.tick_tock("fill na"):
        # train = train.fillna(train.mean())
        if save_csv == 0:

            train = train.fillna(-999)
    # # with ut.tick_tock("filter "):
    #     train[label] = train[label].fillna(0)
    #     # train = train[~(train.sales > filter_thred)]
    #     #
    #     # print "filter {}".format(before_len - len(train))
    # train[label] = train[label].apply(lambda x: tran_to_train(x))

    print "len(predictors),len(cat)", len(predictors), len(cat)

    X = train[predictors + cat]
    X_test = test[predictors + cat]
    print save_csv, "save_csv"
    fea_list = "_".join(args.add_feas.split(",")) if args.add_feas else "base"
    if save_csv != 0:
        X.to_csv('../data/train_{}_{}_{}.csv'.format(cst.code_name, fea_list, args.intro),index=False)
        X_test.to_csv('../data/test_{}_{}_{}.csv'.format(cst.code_name, fea_list, args.intro),index=False)



    print " train label distribution "
    print train[label].describe()

    train_label = train[label].copy()
    print " X.shape", X.shape
    # with ut.tick_tock("out train.csv"):
    #     train[predictors].to_csv('../data/for_vif.csv', index=False)

    print "check inf"
    for i in predictors:
        if X[i].std() == np.nan or X[i].min == -np.inf or X[i].max == np.inf:
            print "!" * 100
            print i

    # X =
    # min_data_in_leaf = 100
    num_boost_round = paras.pop('num_boost_round', 2000)
    evals_results = {}
    # {'num_leaves': 100, 'nthread': 12, 'verbosity': 0, 'max_depth': 15, 'min_child_samples': 100, 'objective': 'regression',
    #  'metric': ['MAE'], 'feature_fraction': 0.7, 'learning_rate': 0.15, 'boosting_type': 'gbdt'}
    # params = paras
    # params = {'num_leaves': 31,
    #      'min_data_in_leaf': 30,
    #      'objective':'regression',
    #      'max_depth': -1,
    #      'learning_rate': 0.01,
    #      "min_child_samples": 20,
    #      "boosting": "gbdt",
    #      "feature_fraction": 0.9,
    #      "bagging_freq": 1,
    #      "bagging_fraction": 0.9 ,
    #      "bagging_seed": 11,
    #      "metric": 'rmse',
    #      "lambda_l1": 0.1,
    #      "verbosity": -1,}
    # params = {
    #     'num_leaves': 100,
    #     'min_data_in_leaf': 20,
    #     'objective': 'regression',
    #     'max_depth': 15,
    #     'learning_rate': 0.01,
    #     # "min_child_samples": 5,
    #     "boosting": "gbdt",
    #     "feature_fraction": 0.9,
    #     "bagging_freq": 3,
    #     "bagging_fraction": 0.85,
    #     "bagging_seed": 11,
    #     "metric": 'rmse',
    #     # "lambda_l1": 0.1,
    #     "verbosity": -1,
    #     "nthread": 12,
    #     "random_state": 4590}
    # last
    # params = {'num_leaves': 50, 'bagging_seed': 11, 'learning_rate': 0.002, 'min_data_in_leaf': 100,
    #           'boosting': 'gbdt', 'bagging_fraction': 0.85, 'metric': ['auc'], 'bagging_freq': 3,
    #           'verbosity': -1,
    #           'nthread': paras.pop("nthread", 12), 'random_state': 4590, 'objective': 'binary', 'max_depth': 7,
    #           'lambda_l2': paras.pop("lambda_l2", 8),
    #           'feature_fraction': 0.4}

    params = {
        'nthread': paras.pop("nthread", 12),
        'lambda_l2': paras.pop("lambda_l2", 0.5),
        # 'bagging_freq': paras.pop("bagging_freq", 5),
        # 'bagging_fraction': paras.pop("bagging_fraction", 1.0),
        'boost_from_average': 'false',
        'boost': 'gbdt',
        'feature_fraction': paras.pop("feature_fraction", 0.7),
        'learning_rate': paras.pop("learning_rate", 0.01),
        'max_depth': paras.pop("max_depth", -1),
        'metric': 'auc',
        'subsample_freq': paras.pop("subsample_freq", 1),
        'subsample': paras.pop("subsample", 0.7),
        'min_data_in_leaf': paras.pop("min_data_in_leaf", 20),
        'min_sum_hessian_in_leaf': paras.pop("min_sum_hessian_in_leaf", 0.001),
        'num_leaves': paras.pop("num_leaves", 2**8),
        'tree_learner': 'serial',
        'objective': 'binary',
        'max_bin':255,
        'verbosity': 1}

    # params = {'num_leaves': 491,
    #           'min_child_weight': 0.03454472573214212,
    #           'feature_fraction': 0.3797454081646243,
    #           'bagging_fraction': 0.4181193142567742,
    #           'min_data_in_leaf': 106,
    #           'objective': 'binary',
    #           'max_depth': -1,
    #           'learning_rate': 0.006883242363721497,
    #           "boosting_type": "gbdt",
    #           "bagging_seed": 11,
    #           "metric": 'auc',
    #           "verbosity": -1,
    #           'reg_alpha': 0.3899927210061127,
    #           'reg_lambda': 0.6485237330340494,
    #           'random_state': 47,
    #           }

    # params = {
    #     'objective': 'binary',
    #     'boosting_type': 'gbdt',
    #     'metric': 'auc',
    #     'n_jobs': 8,
    #     'learning_rate': 0.01,
    #     'num_leaves': 2 ** 8,
    #     'max_depth': -1,
    #     'tree_learner': 'serial',
    #     'colsample_bytree': 0.7,
    #     'subsample_freq': 1,
    #     'subsample': 0.7,
    #     'n_estimators': 1800,
    #     'max_bin': 255,
    #     'verbose': -1,
    #     'seed': 1111,
    #     'early_stopping_rounds': 100,
    # }


    # params = {
    #     'task': 'train',
    #     'boosting': 'gbdt',
    #     'objective': 'binary',
    #     'metric': ['binary_logloss'],
    #     # 'metric': 'rmse',
    #     'learning_rate': 0.01,
    #     'subsample': 0.9855232997390695,
    #     'max_depth': 7,
    #     'top_rate': 0.9064148448434349,
    #     'num_leaves': 63,
    #     'min_child_weight': 41.9612869171337,
    #     'other_rate': 0.0721768246018207,
    #     'reg_alpha': 9.677537745007898,
    #     'colsample_bytree': 0.5665320670155495,
    #     'min_split_gain': 9.820197773625843,
    #     'reg_lambda': 8.2532317400459,
    #     'min_data_in_leaf': 21,
    #     'verbose': -1,
    #     # 'seed': int(2 ** n_fold),
    #     # 'bagging_seed': int(2 ** n_fold),
    #     # 'drop_seed': int(2 ** n_fold)
    # }
    # params = {
    #     'num_leaves': 30,
    #     'min_data_in_leaf': 20,
    #     'objective': 'binary',
    #     'max_depth': 7,
    #     'learning_rate': 0.01,
    #     # "min_child_samples": 5,
    #     # "is_unbalance": True,
    #     "boosting": "gbdt",
    #     "feature_fraction": 0.9,
    #     # "bagging_freq": 3,
    #     # "bagging_fraction": 0.85,
    #     # "bagging_seed": 11,
    #     "metric": ['auc', 'binary_logloss'],
    #     # "lambda_l1": 0.1,
    #     "verbosity": -1,
    #     "nthread": 12,
    #     "random_state": 4590}
    # params = {}  # no para work
    params.update(paras)
    # print "final param {} ".format(params)
    # params = {
    #     # "objective": "fair",
    #     # "fair_c":0.7,
    #     "objective": "regression_l2",
    #     # "objective": "poisson",
    #
    #     # "objective": "huber",
    #
    #     "metric": "None",
    #     "boosting_type": "gbdt",
    #     "learning_rate": 0.15,
    #     "num_leaves": 100,
    #     "max_depth": 15,  # 2**7=128
    #     # "bagging_freq":10,
    #     # "max_bin": 127,
    #     # "min_data_in_bin":5,
    #     'feature_fraction': 0.7,
    #     # 'max_bin': 511,
    #     # 'lambda_l1':10,
    #     # 'bagging_fraction': 0.9,
    #     # "bagging_freq": 5,
    #     # "drop_rate": 0.1,
    #     # "is_unbalance": False,
    #     # "max_drop": 50,
    #     "min_child_samples": 100,
    #     # "min_gain_to_split": 10,
    #     # "lambda_l2": 1,
    #     # "drop_rate": 0.1,
    #     # "max_drop": 50,
    #     # "min_child_weight": 10,
    #     # "min_split_gain": 0,
    #     # "subsample": 0.9,
    #     # "device_type":"gpu",
    #     'nthread': 12,
    # }
    print "final params"

    print params

    x_score = []

    final_cv_train = np.zeros(len(X))
    final_cv_pred = np.zeros(len(X_test))

    cv_only = True
    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=4590)
    # folds = GroupKFold(n_splits=NFOLDS)
    # print "GroupKFold","??????????"
    # folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=4590)

    # fold_rs = folds.split(train, train['outliers'])

    # kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
    # kf = KFold(train.shape[0], n_folds=folds, shuffle=True, random_state=seed)

    '''
    fairobj <- function(preds, dtrain) {
      labels <- getinfo(dtrain, "label")
      con <- 2
      x <-  preds-labels
      grad <- con*x / (abs(x)+con)
      hess <- con^2 / (abs(x)+con)^2
      return(list(grad = grad, hess = hess))
    }
    '''

    fair_constant = 0.1


    def fair_obj(preds, dtrain):
        labels = dtrain.get_label()
        x = (preds - labels)
        den = abs(x) + fair_constant
        grad = fair_constant * x / (den)
        hess = fair_constant * fair_constant / (den * den)
        return grad, hess


    def logregobj(preds, dtrain):
        labels = dtrain.get_label()
        con = 0.7
        x = preds - labels
        grad = con * x / (np.abs(x) + con)
        hess = con ** 2 / (np.abs(x) + con) ** 2
        return grad, hess


    def smoteAdataset(Xig_train, yig_train, Xig_test, yig_test):
        sm = SMOTE(random_state=2, n_jobs=4)
        Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())

        return Xig_train_res, pd.Series(yig_train_res), Xig_test, pd.Series(yig_test)


    BAG_NUM = params.pop("bag_num", 1)
    print "BAG_NUM", BAG_NUM
    fold_scores_bag = []
    fold_logloss_bag = []
    top100_list_bag = []
    permutation_dict = Counter()
    for s in xrange(BAG_NUM):
        fold_rs = folds.split(train, train[cst.label])

        params['seed'] = int(s) + 2018
        with ut.tick_tock("bag round {}".format(s)):
            cv_train = np.zeros(len(X))
            cv_pred = np.zeros(len(X_test))

            if cv_only:
                print "cv_train[:50]", cv_train[:50]
                print "final_cv_train", final_cv_train[:50]

                best_trees = []
                fold_scores = []
                fold_logloss = []
                top100_list = []
                for i, (train_fold, validate) in enumerate(fold_rs):
                    params['seed'] = params['seed'] + i * 10
                    print params['seed'], "params['seed']"
                    # print "np.max(label_validate)",np.max(label_validate)
                    with ut.tick_tock("round {}".format(i)):

                        # use_smote = params.pop("use_smote", 1)
                        # if use_smote:
                        #     X.fillna(0, inplace=True)
                        #     X_train, X_validate, label_train, label_validate = \
                        #         X.iloc[train_fold, :], X.iloc[validate, :], train_label.iloc[train_fold], train_label.iloc[
                        #             validate]
                        #
                        #     trn_xa, trn_y, val_xa, val_y = smoteAdataset(X_train.values, label_train.values, X_validate.values,
                        #                                                  label_validate.values)
                        #     trn_x = pd.DataFrame(data=trn_xa, columns=trn_x.columns)
                        #
                        #     val_x = pd.DataFrame(data=val_xa, columns=val_x.columns)
                        # else:
                        X_train, X_validate, label_train, label_validate = \
                            X.iloc[train_fold, :], X.iloc[validate, :], train_label.iloc[train_fold], train_label.iloc[
                                validate]

                        # print pd.DataFrame(label_train, columns=['label_train']).describe()
                        # print pd.DataFrame(label_validate, columns=['label_validate']).describe()

                        dtrain = lgbm.Dataset(X_train.values.astype(np.float32), label_train, categorical_feature=cat)
                        dvalid = lgbm.Dataset(X_validate.values.astype(np.float32), label_validate, reference=dtrain,
                                              categorical_feature=cat)
                        # num_boost_round / 20,/
                        early_stopping_rounds = 5000
                        bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=[dvalid],
                                         # feval=xg_eval_genhao_mae,
                                         verbose_eval= early_stopping_rounds/5,
                                         early_stopping_rounds=early_stopping_rounds, valid_names=["valid"],
                                         evals_result=evals_results,
                                         categorical_feature=cat)

                        best_trees.append(bst.best_iteration)
                        train_pred = bst.predict(X_train, num_iteration=bst.best_iteration)
                        val_pred = bst.predict(X_validate, num_iteration=bst.best_iteration)



                        # np.exp(allpredictions.mean(axis=1).values) - shift
                        print " label_train", label_train.to_frame().describe()
                        print " label_validate", label_validate.to_frame().describe()

                        print "train pred "
                        print pd.DataFrame(train_pred, columns=['train_pred']).describe()

                        print "cv pred mean"
                        print pd.DataFrame(val_pred, columns=['val_pred']).describe()

                        pred = bst.predict(X_test, num_iteration=bst.best_iteration)

                        cv_pred += pred
                        # mae = evals_results['valid']["auc"][bst.best_iteration - 1]
                        binary_logloss = evals_results['valid']["auc"][bst.best_iteration - 1]
                        mae = binary_logloss
                        fold_logloss.append(binary_logloss)
                        val_mae = mae
                        print "test pred mean"
                        print pd.DataFrame(pred, columns=['val_pred']).describe()
                        print "!!! auc is", val_mae

                        cv_train[validate] += bst.predict(X_validate, num_iteration=bst.best_iteration)

                        importance = bst.feature_importance()

                        print('importance score')
                        df = pd.DataFrame({'feature': X_train.columns, 'importances': bst.feature_importance('gain')})
                        df['fscore'] = df['importances'] / df['importances'].sum()

                        # print(df)
                        print(df.sort_values('importances', ascending=False).to_string(index=False))
                        print  "-" * 20 + " features" + "-" * 20
                        print ",".join([str(x.strip()) for x in
                                        df.sort_values('importances', ascending=False).feature])
                        print "feature", len(X_train.columns)
                        fold_scores.append(val_mae)
                        def imp_metric(model, X_val, y_val):
                            pred = model.predict(X_val)
                            return ut.fast_auc(y_val, pred)

                        print "X_validate columns ",len(X_validate.columns.tolist())
                        p_inportance = ut.permutation_importances(bst, X_validate, label_validate, imp_metric)

                        print p_inportance,">>>>>",len(p_inportance)
                        permutation_dict.update(p_inportance)
                        print permutation_dict, "permutation_dict >>>>>", len(permutation_dict)
                        print ",".join(permutation_dict.keys())

                        del dtrain
                        del dvalid
                        del X_train
                        del X_validate
                        del label_train
                        del label_validate
                        import gc

                        gc.collect()
                        # if do_valid:
                        #     print "!!!!!!!!!!!!!!!!!!!!!!" + "this is valid mode!!!!!" + "!!!!!!!!!!!!!!!!!!!!"
                        #     exit()

                cv_pred /= NFOLDS
                final_cv_train += cv_train
                final_cv_pred += cv_pred
                print 'cv_train monitor'
                print pd.DataFrame({"cv_tar": cv_train}).describe()
                print 'cv_pred monitor'
                print pd.DataFrame({"cv_pred": cv_pred}).describe()

                print("cv score:")
                print(fold_scores)
                print("cv logloss:")
                print(fold_logloss)

                print(np.mean(fold_scores), 'fold_scores')
                print(np.mean(fold_logloss), "fold_logloss")
                fold_scores_bag.append(np.mean(fold_scores))
                fold_logloss_bag.append(np.mean(fold_logloss))

                print(best_trees, np.mean(best_trees))

    # print(fold_scores_bag)
    print(" fold_scores_bag cv logloss:")
    print(fold_logloss_bag)

    print(np.mean(fold_scores_bag), 'bag_fold_scores')
    print(np.mean(fold_logloss_bag), "bag_fold_logloss")

    bag_cv_pred = final_cv_pred / BAG_NUM
    bag_cv_train = final_cv_train / BAG_NUM
    # print 'final_cv_train monitor'
    # print pd.DataFrame({"final_cv_train_tar": final_cv_train}).describe()
    # print 'final_cv_pred monitor'
    # print pd.DataFrame({"final_cv_pred_tar": final_cv_pred}).describe()

    print 'bag_cv_train monitor'
    print pd.DataFrame({"bag_cv_train_tar": bag_cv_train}).describe()
    print 'bag_cv_pred monitor'
    print pd.DataFrame({"bag_cv_pred_tar": bag_cv_pred}).describe()

    test[label] = bag_cv_pred
    print "sales mean ", test[label].mean()

    # train[cst.key] = train[cst.key].astype(str).str.cat(train.poi_id.astype(str), sep="_")
    # test[cst.key] = test[cst.key].astype(str).str.cat(test.poi_id.astype(str), sep="_")

    # test[['deal_poi', 'sales']].to_csv('../out/lgbm_out_CV5_{}_{}_{}_{}.csv.gz'.format(args.intro, np.mean(fold_scores),test.sales.mean(),args.valid),
    #                                    index=False, float_format='%.4f', compression='gzip')

    test[label] = np.clip(test[label], -100, 100)
    print "test describe "
    print test[label].describe()
    # hash_sign = hash("{}_{}_{}".format(args.intro, np.mean(fold_logloss_bag),
    #                                    "_".join(args.add_feas.split(",")) if args.add_feas else "base"))
    sign = "{}_{}_{}_{}_{}".format(args.intro, md5,
                                  round(np.mean(fold_logloss_bag),6),
                                  round(test[label].mean(), 4),
                                  "_".join(args.add_feas.split(",")) if args.add_feas else "base")[:200]

    # cst.code_name = 'outlier'

    if rm_high==1:
        with ut.tick_tock("a"):
            test = test.merge(et_index,how='left',on=cst.key)
        with ut.tick_tock("a"):
            test.loc[test['isFraud_x']>0,"isFraud"] = test['isFraud_x']

    print "?????"
    with ut.tick_tock("a"):
        test[[cst.key, label]].to_csv(
            '../out/lgbm_{}_base_out_{}.csv.gz'.format(cst.code_name, sign),
            index=False, float_format='%.8f', compression='gzip')
    print "len(train.deal_poi),len(bag_cv_train)", len(train[cst.key]), len(bag_cv_train)
    print "len(test.deal_poi),len(bag_cv_pred)", len(test[cst.key]), len(bag_cv_pred)

    cv_sign = "{}".format(len(train[cst.key]))
    pd.DataFrame({cst.key: train[cst.key], label: bag_cv_train}).to_csv(
        '../stacking/lgbm_{}_base_cv_{}_{}.csv'.format(cst.code_name, cv_sign, sign), index=False)

    pd.DataFrame({cst.key: test[cst.key], label: bag_cv_pred}).to_csv(
        '../stacking/lgbm_{}_base_pred_{}_{}.csv'.format(cst.code_name, cv_sign, sign), index=False)

    print "all done", "{}{}".format(args.intro, md5)
