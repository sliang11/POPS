# -*- coding: UTF-8 -*-

from util.util import is_iterable, to_iterable
import pickle
import numpy as np
from collections import OrderedDict
import gc
import os


class FileReader(object):

    @staticmethod
    def load_pickle(fname, path=None):

        # 新增，待检查：直接粘下来的
        if path is not None:
            fname = os.path.join(path, fname)

        gc.disable()
        with open(fname, 'rb') as f:
            ret = pickle.load(f)
        gc.enable()
        return ret


class FileWriter(object):

    @staticmethod
    def dump_pickle(obj, fname, path=None):

        if path is not None:
            fname = os.path.join(path, fname)

        gc.disable()
        with open(fname, 'wb') as f:
            pickle.dump(obj, f, protocol=-1)
        gc.enable()


class DirProcessor(object):

    @staticmethod
    def create_dir(dir):
        if not os.path.exists(dir):
            os.mkdir(dir)


class FileNameProcessor(object):

    @staticmethod
    def var2str(var, primary_sep='_', secondary_sep='-'):

        simple_types = (str, int, float, bool, np.int32, np.int64, np.float32, np.float64)
        if type(var) in simple_types or var is None:
            return str(var)
        elif type(var) in (list, tuple, np.ndarray):
            for elm in var:
                if type(elm) not in simple_types or elm is None:
                    print(f'Error! Data type {type(elm)} of element {elm} currently not supported!')
                    exit(1)
            return primary_sep.join((FileNameProcessor.var2str(elm) for elm in var))
        elif type(var) == OrderedDict:  # 注意：不允许使用dict，否则无法判断顺序！
            return primary_sep.join((
                primary_sep.join(
                    (key,
                     secondary_sep.join((FileNameProcessor.var2str(elm) for elm in to_iterable(val, to_ndarry=False))))
                ) for key, val in var.items()
            ))
        else:
            print(f'Error! Data type {type(var)} currently not supported!')
            exit(1)

    @staticmethod
    def create_fname(elements, ext=None, path=None, sep='_', primary_sep='_', secondary_sep='-'):
        fname = sep.join((FileNameProcessor.var2str(elm, primary_sep, secondary_sep)
                          for elm in elements))

        if ext is not None:
            if not ext.startswith('.'):
                ext = '.' + ext
            fname += ext
        if path is not None:
            fname = os.path.join(path, fname)
        return fname


class VariablePrinter(object):

    @staticmethod
    def print_variables(s_variables, variables, pfx_msg=None, sfx_msg=None):

        if not is_iterable(s_variables):
            s_vars, vars = [s_variables], [variables]
        else:
            s_vars, vars = s_variables, variables

        print_str = ', '.join(
            (s_var + ' = ' + str(vars[i]).replace('\n', '') for i, s_var in enumerate(s_vars))
        )
        if pfx_msg is not None:
            print_str = pfx_msg + ' ' + print_str
        if sfx_msg is not None:
            print_str += ' ' + sfx_msg
        print_str = 'print(' + repr(print_str) + ')'
        exec(print_str)

    @staticmethod
    def gen_cmd_print_variables(str_all_vars, separator=', ', pfx_msg=None, sfx_msg=None):

        var_names = str_all_vars.split(separator)
        for i in range(len(var_names)):
            var_names[i].replace(' ', '')

        if len(var_names) == 1:
            str_s_variables = '\'' + var_names[0] + '\''
            str_variables = 'eval(' + str_s_variables + ')'
        else:
            str_s_variables = '(' + ', '.join(('\'' + var_name + '\'' for var_name in var_names)) + ')'
            str_variables = 'list(map(eval, ' + str_s_variables + '))'
        to_exec = 'VariablePrinter.print_variables(' + ', '.join((str_s_variables, str_variables))
        if pfx_msg is not None:
            to_exec += ', pfx_msg=\'' + pfx_msg + '\''
        if sfx_msg is not None:
            to_exec += ', sfx_msg=\'' + sfx_msg + '\''
        to_exec += ')'
        return to_exec
