# encoding=utf8
import time
import pandas as pd
import numpy as np

import datetime

import glob
import feature_lib as flb
from sklearn.neighbors import KernelDensity
from operator import itemgetter

# promo_2017_train = df_2017.set_index(
#     ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
#         level=-1).fillna(False)

def calculate_pdf_difference(feat, df_feature, df_target, IQR_multiplier, bin_bandwidth_multiplier, print_number_bins):
    # Agreggating feature values in bin format using the Freedman-Diaconis rule
    IQR = df_feature[feat].quantile([0.9]).values - df_feature[feat].quantile(
        [0.1]).values  # Interquartile range (IQR)

    n = len(df_feature[feat])
    bin_size = IQR_multiplier * IQR / n ** (1 / 3)
    bin_number = int(np.round((df_feature[feat].max() - df_feature[feat].min()) / bin_size))
    #     bin_number = 50
    binvalues = pd.cut(df_feature[feat], bins=bin_number, labels=range(bin_number)).astype('float')
    print " bin_size", bin_size,"bin_number",bin_number

    # print "IQR", IQR, "n", n, "bin_size", bin_size, "bin_number", bin_number, "binvalues", binvalues[:1000]

    if print_number_bins:
        print('There are {} bins in the feature {}'.format(bin_number, feat))

    # Calculate the PDFs using the df_target
    pdf_0 = KernelDensity(kernel='gaussian', bandwidth=0.3)
    pdf_0.fit(np.array(df_target[feat][df_target['target'] == 0]).reshape(-1, 1))

    pdf_1 = KernelDensity(kernel='gaussian', bandwidth=0.3)
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

    # print "pdf_dict", pdf_dict
    return bin_pdf_values

def calculate_mean_encoding(feat, df_feature, df_target, IQR_multiplier, bin_bandwidth_multiplier, print_number_bins):
    key = 'target_{}_mean'.format(feat)
    df_feature,df_target = flb.feat_mean_tmp(df_feature,df_target,[feat],'target')
    # print a.head()
    feat = key
    print df_feature[feat].describe()
    # Agreggating feature values in bin format using the Freedman-Diaconis rule
    IQR = df_feature[feat].quantile([0.9]).values - df_feature[feat].quantile(
        [0.1]).values  # Interquartile range (IQR)

    n = len(df_feature[feat])
    bin_size = 0.00333333
    print (df_feature[feat].max() - df_feature[feat].min()),"(df_feature[feat].max() - df_feature[feat].min())"
    bin_number = int(np.round((df_feature[feat].max() - df_feature[feat].min()) / bin_size))
    #     bin_number = 50
    binvalues = pd.cut(df_feature[feat], bins=bin_number, labels=range(bin_number)).astype('float')
    print " bin_size", bin_size,"bin_number",bin_number

    # print "IQR", IQR, "n", n, "bin_size", bin_size, "bin_number", bin_number, "binvalues", binvalues[:1000]


    if print_number_bins:
        print('There are {} bins in the feature {}'.format(bin_number, feat))

    # Calculate the PDFs using the df_target
    pdf_0 = KernelDensity(kernel='gaussian', bandwidth=0.3)
    pdf_0.fit(np.array(df_target[feat]).reshape(-1, 1))


    # Creates an X array with the average feature value for each bin
    x = np.array(np.arange(min(df_feature[feat]) + bin_size / 2, max(df_feature[feat]), bin_size)).reshape(-1, 1)

    # gets the pdf values based on the X array
    log_pdf_0 = np.exp(pdf_0.score_samples(x))
    # log_pdf_1 = np.exp(pdf_1.score_samples(x))

    # creates a dictionary that links the bin number with the PDFs value difference
    pdf_dict = dict()
    for i in range(bin_number):
        pdf_dict[i] = log_pdf_0[i]

        # gets the PDF difference for each row of the dataset based on its equivalent bin.
    bin_pdf_values = np.array(itemgetter(*list(binvalues))(pdf_dict))

    # print "pdf_dict", pdf_dict
    return bin_pdf_values

def deco_outcols(fea_key, outcols, exclude_cols=[]):
    return map(lambda x: "_".join([fea_key, str(x)]) if x not in exclude_cols else x, outcols)


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


def augment(x, y, t=2):
    xs, xn = [], []
    for i in range(t):
        mask = y > 0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:, c] = x1[ids][:, c]
        xs.append(x1)

    for i in range(t // 2):
        mask = y == 0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:, c] = x1[ids][:, c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x, xs, xn])
    y = np.concatenate([y, ys, yn])
    return x, y


def check_features(train, test, out_cols):
    base_cols = out_cols
    std = abs((train[out_cols].mean() - test[out_cols].mean())) / test[base_cols].mean()
    rs = zip(out_cols, std)
    print("mean", sorted(rs, key=lambda x: -x[1]))
    out_cols = map(lambda x: x[0], filter(lambda x: x[1] < 0.31, rs))
    print(len(out_cols), out_cols)
    mean_rm = list(set(base_cols) - set(out_cols))
    print("rm feature {}".format(mean_rm))

    std = abs((train[out_cols].std() - test[out_cols].std())) / test[base_cols].std()
    rs = zip(out_cols, std)
    out_cols = map(lambda x: x[0], filter(lambda x: x[1] < 0.3, rs))

    print("std", sorted(rs, key=lambda x: -x[1]))
    print(len(out_cols), out_cols)

    std_rm = list(set(base_cols) - set(out_cols))
    print("rm feature {}".format(std_rm))
    print('final rm {}'.format(",".join(list(set(mean_rm) & set(std_rm)))))

    return out_cols


import pandas as pd
from pandas.core.algorithms import unique1d
from sklearn.base import BaseEstimator, TransformerMixin


class PandasLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, y):
        self.classes_ = unique1d(y)
        return self

    def transform(self, y):
        s = pd.Series(y).astype('category', categories=self.classes_)
        return s.cat.codes

    def fit_transform(self, y, **fit_params):
        self.fit(y)
        return self.transform(y)


class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print("*" * 50 + " {} START!!!! ".format(self.process_name) + "*" * 50)
            self.begin_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            duration_seconds = end_time - self.begin_time
            duration = str(datetime.timedelta(seconds=duration_seconds))

            print("#" * 50 + " {} END... time lapsing {}  ". \
                  format(self.process_name, duration) + "#" * 50)


dates = {}


def check_missing_data(df):
    flag = df.isna().sum().any()
    if flag == True:
        total = df.isnull().sum()
        percent = (df.isnull().sum()) / (df.isnull().count() * 100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return (np.transpose(output))
    else:
        return (False)


def lookup(s):
    global dates
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    for date in s.unique():
        dates[date] = dates[date] if date in dates else pd.to_datetime(date)
    # dates = {date: pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)


def log_for_fileds(df, field_list):
    # df = df.replace(0,0.00001)
    for f in field_list:
        df[f] = np.log(df[f] + 1)
    return df


from collections import Counter


def permutation_importances(rf, X_train, y_train, metric):
    baseline = metric(rf, X_train, y_train)
    print "baseline is ", baseline
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        imp.append((baseline - m) / baseline * 100)
    cols = X_train.columns.tolist()
    return Counter(dict(zip(cols, imp)))
    # return np.array(imp)


def get_batch(batch_size, index, *args):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(args[0]) else len(args[0])
    return [x[start:end] for x in args]


def gene_tran_lable(method, value, base=1):
    def log_tran(mode="to_train"):
        if mode == 'to_train':
            return lambda x: np.log(x + value)
        else:
            shift = value
            return lambda x: np.exp(x) - shift

    def expk_tran(mode="to_train"):
        if mode == 'to_train':
            # print            value, type(value), "????"

            tran_expk = 1.0 / value
            return lambda x: np.power(x + base, tran_expk)
        else:
            expk = value
            return lambda x: np.power(x, expk) - base

    def daoshu(mode="to_train"):
        if mode == 'to_train':
            return lambda x: value / (x + 0.001)
        else:
            return lambda x: float(value) / x - 0.001

    if method == 'expk':
        return expk_tran

    if method == 'log':
        return log_tran

    if method == 'daoshu':
        return daoshu


# coding:utf8
import pandas as pd
import numpy as np
import sys, os


# @from: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65/code
# @liscense: Apache 2.0
# @author: weijian
def reduce_mem_usage(props):
    # 计算当前内存
    start_mem_usg = props.memory_usage().sum() / 1024.0 ** 2
    print("Memory usage of the dataframe is :", start_mem_usg, "MB")

    # 哪些列包含空值，空值用-999填充。why：因为np.nan当做float处理
    NAlist = []
    for col in props.columns:
        # 这里只过滤了objectd格式，如果你的代码中还包含其他类型，请一并过滤
        if (props[col].dtypes != object):

            # print("**************************")
            # print("columns: ", col)
            # print("dtype before", props[col].dtype)

            # 判断是否是int类型
            isInt = False
            mmax = props[col].max()
            mmin = props[col].min()

            # Integer does not support NA, therefore Na needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].replace([np.inf, -np.inf], np.nan)
                props[col].fillna(-999, inplace=True)  # 用-999填充
                # props[col].repa(-999, inplace=True) # 用-999填充

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = np.fabs(props[col] - asint)
            result = result.sum()
            if result < 0.01:  # 绝对误差和小于0.01认为可以转换的，要根据task修改
                isInt = True

            # make interger / unsigned Integer datatypes
            if isInt:
                if mmin >= 0:  # 最小值大于0，转换成无符号整型
                    if mmax <= 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mmax <= 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mmax <= 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:  # 转换成有符号整型
                    if mmin > np.iinfo(np.int8).min and mmax < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mmin > np.iinfo(np.int16).min and mmax < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mmin > np.iinfo(np.int32).min and mmax < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mmin > np.iinfo(np.int64).min and mmax < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)
            else:
                # 注意：这里对于float都转换成float16，需要根据你的情况自己更改
                if mmax >= 32768 or mmin <= -32768:
                    props[col] = props[col].astype(np.float32)
                else:
                    props[col] = props[col].astype(np.float32)

                    # print("dtype after", props[col].dtype)
                    # print("********************************")
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024.0 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props


def scan_column_stats(props):
    for col in props.columns:
        mmax = props[col].max()
        mmin = props[col].min()
        print
        "col:", col, ", max:", mmax, ", min:", mmin


# encoding:utf8
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from keras import backend as K


# from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear

class QAAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(QAAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = (1. - att_score) * state + att_score * c
        return new_h, new_h


class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h


def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc


import numpy as np
from numba import jit


@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


import time


def approximate_auc_1(labels, preds):
    """
        近似方法，将预测值分桶(n_bins)，对正负样本分别构建直方图，再统计满足条件的正负样本对
        复杂度 O(N)
    """
    # gmv值 换成0、1值
    # labels = np.where(labels > 0.0001, 1.0, 0.0)
    # print("----approximate_auc_1 开始--- %s" % time.asctime(time.localtime(time.time())))
    n_bins = 500000
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    total_pair = n_pos * n_neg

    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 25.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i] / bin_width)
        if nth_bin > n_bins:
            nth_bin = n_bins - 1
        if labels[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1

    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
        accumulated_neg += neg_histogram[i]
    # print("----approximate_auc_1 结束--- %s" % time.asctime(time.localtime(time.time())))
    return satisfied_pair / float(total_pair + 0.01)


def attention(query, facts, attention_size, mask, stag='null', mode='LIST', softmax_stag=1, time_major=False,
              return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])

    mask = tf.equal(mask, tf.ones_like(mask))
    hidden_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    input_size = query.get_shape().as_list()[-1]

    # Trainable parameters
    w1 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    w2 = tf.Variable(tf.random_normal([input_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    v = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `tmp` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        tmp1 = tf.tensordot(facts, w1, axes=1)
        tmp2 = tf.tensordot(query, w2, axes=1)
        tmp2 = tf.reshape(tmp2, [-1, 1, tf.shape(tmp2)[-1]])
        tmp = tf.tanh((tmp1 + tmp2) + b)

    # For each of the timestamps its vector of size A from `tmp` is reduced with `v` vector
    v_dot_tmp = tf.tensordot(tmp, v, axes=1, name='v_dot_tmp')  # (B,T) shape
    key_masks = mask  # [B, 1, T]
    # key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(v_dot_tmp) * (-2 ** 32 + 1)
    v_dot_tmp = tf.where(key_masks, v_dot_tmp, paddings)  # [B, 1, T]
    alphas = tf.nn.softmax(v_dot_tmp, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    # output = tf.reduce_sum(facts * tf.expand_dims(alphas, -1), 1)
    output = facts * tf.expand_dims(alphas, -1)
    output = tf.reshape(output, tf.shape(facts))
    # output = output / (facts.get_shape().as_list()[-1] ** 0.5)
    if not return_alphas:
        return output
    else:
        return output, alphas


def din_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                  return_alphas=False):
    print "query.get_shape()", query.get_shape()
    print "fact.get_shape()", facts.get_shape()
    print "tf.shape(facts)[1].get_shape()", tf.shape(facts)[1].get_shape()
    print "tf.shape(facts)", tf.shape(facts)
    print "tf.shape(facts)[1]", tf.shape(facts)[1]
    # print t(tf.shape(facts)[1])

    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print ("querry_size mismatch")
        query = tf.concat(values=[
            query,
            query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.relu, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.relu, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    print("scores", scores.get_shape())
    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]
        print("scores", scores.get_shape())

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output


def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                      return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output


def self_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch, output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:, i, :], batch[:, 0:i + 1, :],
                                               ATTENTION_SIZE, mask[:, 0:i + 1], softmax_stag=1, stag=stag,
                                               mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(i, self_attention_tmp)
        return batch, output, i + 1

    output_ta = tf.TensorArray(dtype=tf.float32,
                               size=0,
                               dynamic_size=True,
                               element_shape=(facts[:, 0, :].get_shape()))
    _, output_op, _ = tf.while_loop(cond, body, [facts, output_ta, 0])
    self_attention = output_op.stack()
    self_attention = tf.transpose(self_attention, perm=[1, 0, 2])
    return self_attention


def self_all_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch, output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:, i, :], batch,
                                               ATTENTION_SIZE, mask, softmax_stag=1, stag=stag,
                                               mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(i, self_attention_tmp)
        return batch, output, i + 1

    output_ta = tf.TensorArray(dtype=tf.float32,
                               size=0,
                               dynamic_size=True,
                               element_shape=(facts[:, 0, :].get_shape()))
    _, output_op, _ = tf.while_loop(cond, body, [facts, output_ta, 0])
    self_attention = output_op.stack()
    self_attention = tf.transpose(self_attention, perm=[1, 0, 2])
    return self_attention


def din_fcn_shine(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                  return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1_trans_shine' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, facts_size, activation=tf.nn.sigmoid, name='f1_shine_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, facts_size, activation=tf.nn.sigmoid, name='f2_shine_att' + stag)
    d_layer_2_all = tf.reshape(d_layer_2_all, tf.shape(facts))
    output = d_layer_2_all
    return output

def get_key(inp):
    la = inp.split("_")[-1].split(".")[0]
    return la

def check(a,b):
    assert len(a) == len(b)

if __name__ == "__main__":
    # filename = sys.argv[1]
    # data = pd.read_csv(filename, header=None)
    # data2 = reduce_mem_usage(data)
    # scan_column_stats(data)
    a = np.array([range(10) + range(10 ** 4, 10 ** 5, 3)])
    tran = gene_tran_lable('expk', 2)
    tran_to_train = tran(mode='to_train')
    tran_to_predict = tran(mode='to_predict')
    r = tran_to_train(a)
    print
    a
    print
    r
    print
    tran_to_predict(r)
    print
    "#" * 50

    a = np.array([range(10) + range(10 ** 4, 10 ** 5, 3)])
    tran = gene_tran_lable('log', 10)
    tran_to_train = tran(mode='to_train')
    tran_to_predict = tran(mode='to_predict')
    r = tran_to_train(a)
    print
    a
    print
    r
    print
    tran_to_predict(r)
    print
    "#" * 50

    a = np.array([range(10) + range(10 ** 4, 10 ** 5, 3)])
    tran = gene_tran_lable('daoshu', 1)
    tran_to_train = tran(mode='to_train')
    tran_to_predict = tran(mode='to_predict')
    r = tran_to_train(a)
    print
    a
    print
    r
    print
    tran_to_predict(r)

    # tran = tran_label(a, "expk", 2, 'to_train')
    #
    # print a
    # print "expk", tran
    # print "expk", tran_label(tran, "expk", 2, 'to_pred')
    #
    # a = np.array([range(10) + range(10 ** 4, 10 ** 5, 3)])
    # tran = tran_label(a, "log", 1, 'to_train')
    # print a
    # print "log", tran
    # print "log", tran_label(tran, "log", 1, 'to_pred')
    #
    # a = np.array([range(10) + range(10 ** 4, 10 ** 5, 3)])
    # tran = tran_label(a, "daoshu", 1, 'to_train')
    #
    # print a
    # print "daoshu", tran
    # print "daoshu", tran_label(tran, "daoshu", 1, 'to_pred')
