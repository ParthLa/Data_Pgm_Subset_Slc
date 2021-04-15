import pickle
import numpy as np
#from .config import flags as config

from my_data_types import *

#reduce_x_features = config.w_network == 'textcnn'
reduce_x_features = False
seq_len = 25


def load_data(fname, num_load=None):
    print('Loading from hoff ', fname)
    with open(fname, 'rb') as f:
        x = pickle.load(f)
        l = pickle.load(f).astype(np.int32)
        m = pickle.load(f).astype(np.int32)
        L = pickle.load(f).astype(np.int32)
        d = pickle.load(f).astype(np.int32)
        r = pickle.load(f).astype(np.int32)

        len_x = len(x)
        assert len(l) == len_x
        assert len(m) == len_x
        assert len(L) == len_x
        assert len(d) == len_x
        assert len(r) == len_x

        L = np.reshape(L, (L.shape[0], 1))
        d = np.reshape(d, (d.shape[0], 1))

        if reduce_x_features:
            x = np.concatenate([x[:, 0:seq_len], x[:, 75:(seq_len + 75)],
                x[:, 150:(150 + seq_len)]], axis=-1)

        if num_load is not None and num_load < len_x:
            x = x[:num_load]
            l = l[:num_load]
            m = m[:num_load]
            L = L[:num_load]
            d = d[:num_load]
            r = r[:num_load]

        return F_d_U_Data(x, l, m, L, d, r)


def get_rule_classes(l, num_classes):
    num_rules = l.shape[1]
    rule_classes = []
    for rule in range(num_rules):
        labels = l[:, rule]
        rule_class = num_classes
        for lbl in labels:
            if lbl != num_classes:
                assert lbl < num_classes
                if rule_class != num_classes:
                    #print('rule is: ', rule, 'Rule class is: ', rule_class, 'newly found label is: ', lbl, 'num_classes is: ', num_classes)
                    assert(lbl == rule_class)
                else:
                    rule_class = lbl

        if rule_class == num_classes:
            print('No valid label found for rule: ', rule)
            # ok if a rule is just a label (i.e. it does not fire at all)
            #input('Press a key to continue')
        rule_classes.append(rule_class)

    return rule_classes


def extract_rules_satisfying_min_coverage(m, min_coverage):
    num_rules = len(m[0])
    coverage = np.sum(m, axis=0)
    satisfying_threshold = coverage >= min_coverage
    not_satisfying_threshold = np.logical_not(satisfying_threshold)
    all_rules = np.arange(num_rules)
    satisfying_rules = np.extract(satisfying_threshold, all_rules)
    not_satisfying_rules = np.extract(not_satisfying_threshold, all_rules)

    # Assert that the extraction is stable
    assert np.all(np.sort(satisfying_rules) == satisfying_rules)
    assert np.all(np.sort(not_satisfying_rules) == not_satisfying_rules)

    rule_map_new_to_old = np.concatenate([satisfying_rules,
            not_satisfying_rules])
    rule_map_old_to_new = np.zeros(num_rules, dtype=all_rules.dtype) - 1
    for new, old in enumerate(rule_map_new_to_old):
        rule_map_old_to_new[old] = new

    return satisfying_rules, not_satisfying_rules, rule_map_new_to_old, rule_map_old_to_new


def remap_2d_array(arr, map_old_to_new):
    old = np.arange(len(map_old_to_new))
    arr[:, old] = arr[:, map_old_to_new]
    return arr


def remap_1d_array(arr, map_old_to_new):
    old = np.arange(len(map_old_to_new))
    arr[old] = arr[map_old_to_new]
    return arr


def modify_d_or_U_using_rule_map(raw_U_or_d, rule_map_old_to_new):
    remap_2d_array(raw_U_or_d.l, rule_map_old_to_new)
    remap_2d_array(raw_U_or_d.m, rule_map_old_to_new)


def shuffle_F_d_U_Data(data):
    idx = np.arange(len(data.x))
    np.random.shuffle(idx)
    x = np.take(data.x, idx, axis=0)
    l = np.take(data.l, idx, axis=0)
    m = np.take(data.m, idx, axis=0)
    L = np.take(data.L, idx, axis=0)
    d = np.take(data.d, idx, axis=0)
    r = np.take(data.r, idx, axis=0)

    return F_d_U_Data(x, l, m, L, d, r)


def oversample_f_d(x, labels, sampling_dist):
    x_list = []
    L_list = []
    #print('Sampling distribution: ', sampling_dist)
    #print('labels: ', labels[0:4])
    for xx, L in zip(x, labels):
        for i in range(sampling_dist[L]):
            x_list.append(np.array(xx))
            L_list.append(np.array(L))

    return np.array(x_list), np.array(L_list)

def oversample_d(raw_d, sampling_dist):
    '''
    Func Desc:
    performs oversampling on the raw labelled data using the given distribution

    Input:
    raw_d - raw labelled data
    sampling_dist - the given sampling dist

    Output:
    F_d_U_Data
    '''
    x_list = []
    l_list = []
    m_list = []
    L_list = []
    d_list = []
    r_list = []
    #print('Sampling distribution: ', sampling_dist)
    #print('labels: ', raw_d.L[0:4])
    for x, l, m, L, d, r in zip(raw_d.x, raw_d.l, raw_d.m, raw_d.L, raw_d.d, raw_d.r):
        L1 = np.squeeze(L)
        for i in range(sampling_dist[L1]):
            x_list.append(np.array(x))
            l_list.append(np.array(l))
            m_list.append(np.array(m))
            L_list.append(np.array(L))
            d_list.append(np.array(d))
            r_list.append(np.array(r))

    return F_d_U_Data(np.array(x_list),
            np.array(l_list),
            np.array(m_list),
            np.array(L_list),
            np.array(d_list),
            np.array(r_list))