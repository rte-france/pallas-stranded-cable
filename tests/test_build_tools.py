# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import strdcable.build_tools as build_tools
import numpy as np
import pandas as pd


def test_testvar_validtype():
    """
    test for testvar function in build_tools.py
    - tests if valid data types (some useful python and numpy data types) do not raise an assertion error

    """
    type_list = [int, float, str, list, dict, bool,
                 np.ndarray]
    var_list = [1, 1.0, 'test', [1, 1.0, 'test'], {'test': 1}, True,
                np.array([1, 2, 3])]

    for i, dtype in enumerate(type_list):
        var = var_list[i]

        try:
            build_tools.testvar(var, dtype)
            assert True
        except AssertionError:
            assert False


def test_testvar_invalidtype():
    """
    test for testvar function in build_tools.py
    - tests if an invalid requested data type raises a type error

    """
    a = int(1)
    b = 2.0
    c = np.array([1, 2, 3])

    invalid_list = [a, b, c]

    for val in invalid_list:
        try:
            build_tools.testvar(val, val)
            assert False
        except TypeError:
            assert True


def test_convertvars_validinputs():
    """
    test for convertvars function in build_tools.py
    - tests if valid inputs produce a valid output

    """
    dscal = {'test1': 1, 'test2': 2.0}
    ddict = {'test3': {'a': 'a', 1: 1}}
    darray = {'test4': np.array([1, 2, 3])}

    format_list = ['dict', 'dataframe']

    for i, tested_format in enumerate(format_list):
        with pytest.raises(AssertionError):
            out = build_tools.convertvars(dscal, ddict, darray, tested_format)
            assert isinstance(out, dict)
            assert isinstance(out, pd.DataFrame)


def test_convertvars_invalidformat():
    """
    test for convertvars function in build_tools.py
    - tests if an invalid requested format raises an assertion error

    """
    dscal = {'test1': 1, 'test2': 2.0}
    ddict = {'test3': {'a': 'a', 1: 1}}
    darray = {'test4': np.array([1, 2, 3])}

    invalid_format = 'array'

    try:
        _ = build_tools.convertvars(dscal, ddict, darray, invalid_format)
        assert False
    except AssertionError:
        assert True


def test_get_input_from_data_validinputs():
    """
    test for get_input_from_data in build_tools.py
    - tests if valid inputs produce valid outputs

    """
    data_dict = {'colonne1': [3, 2, 1], 'colonne2': np.array([4, 5, 6])}
    df = pd.DataFrame.from_dict(data_dict)

    test_arr = np.array([True, False, True])

    headers = ['column1', 'colonne2']
    drename = {'column1': 'colonne1'}
    sortcols = ['colonne1']

    dtest, dinput = build_tools.get_input_from_data(df, test_arr, headers, drename, sortcols)

    assert len(dtest.index) == sum(test_arr)
    assert dtest['colonne1'].is_monotonic_increasing
    assert 'column1' in dinput
    assert 'colonne2' in dinput
