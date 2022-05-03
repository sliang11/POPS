import numpy as np
from collections import OrderedDict

def next_integer(val, eq):
    if float(val).is_integer():
        return int(val) if eq else int(val) + 1
    else:
        return int(np.ceil(val))

def last_integer(val, eq):
    if float(val).is_integer():
        return int(val) if eq else int(val) - 1
    else:
        return int(np.floor(val))


def cumsum(arr, pad_0, dtype=float):
    if pad_0:
        return np.cumsum([0] + list(arr), dtype=dtype)
    else:
        return np.cumsum(arr, dtype=dtype)


def intersect1d(*all_arr):
    ret = all_arr[0]
    for arr in all_arr[1:]:
        ret = np.intersect1d(ret, np.array(arr))
    return ret


def union1d(*all_arr):
    ret = all_arr[0]
    for arr in all_arr[1:]:
        ret = np.union1d(ret, np.array(arr))
    return ret


def select_kth(arr, k, min_k_is_0):
    if not min_k_is_0:
        k -= 1

    if k == 0:
        return np.min(arr)
    if k == len(arr):
        return np.max(arr)
    return np.partition(arr, k)[k]


def percentile(arr, p):
    if p > 1 or p < 0:
        print('Error! p must be in [0, 1]!')
        exit(1)

    n = len(arr)
    k = int(np.floor(1 / 3 + p * (n + 1 / 3)))
    if k < 1:
        return np.min(arr)
    if k + 1 > n:
        return np.max(arr)

    gamma = p * n - k + (p + 1) / 3
    return (1 - gamma) * select_kth(arr, k, False) + gamma * select_kth(arr, k + 1, False)


def var(sum1, sum2, num):
    return max(0, sum2 / num - (sum1 * sum1) / (num * num))


def std(sum1, sum2, num):
    return np.sqrt(var(sum1, sum2, num))


def is_iterable(obj, exclude_str=True, exclude_dict=True, exclude_tuple=False):
    try:
        iter(obj)
        ex_flg = (exclude_str and type(obj) == str) or \
                 (exclude_dict and type(obj) in (dict, OrderedDict)) or \
                 (exclude_tuple and type(obj) == tuple)
        return not ex_flg
    except:
        return False


def to_iterable(obj, exclude_str=True, exclude_dict=True, exclude_tuple=False, to_ndarry=True):
    return obj if is_iterable(obj, exclude_str=exclude_str, exclude_dict=exclude_dict,
                              exclude_tuple=exclude_tuple) else (np.array([obj]) if to_ndarry else [obj])
