"""Functions relative to data."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Tuple, Union

# %%


def testvar(var, dtype) -> None:
    """Check the format of a given variable.

    Parameters
    ----------
    var : object
        Given variable.
    dtype
        Type expected for the given variable.

    Returns
    -------
    None.

    """
    if not isinstance(var, dtype) and var is not None:
        raise AssertionError("wrong type ", var)
    return

# %%


def convertvars(dscal: dict, ddict: dict, darray: dict, format: str) -> Union[dict, pd.DataFrame]:
    """Convert three nominal dict in a given format.

    Parameters
    ----------
    dscal : dict
        Dictionary with only scalar for each key.
    ddict : dict
        Dictionary with only dict for each key.
    darray:  dict
        Dictionary with only array for each key
    format : str
        Extracted format of output data. The value must be in
        ['dict', 'dataframe']. The default is 'dict'.

    Returns
    -------
    dict or pandas.DataFrame
        Output data. The type depends on the the given format.

    """
    if format == 'dict':
        out = {**dscal, **ddict, **darray}
    elif format == 'dataframe':
        out = pd.DataFrame.from_dict(darray)
        for k in dscal.keys():
            out[k] = dscal[k]
        for k in ddict.keys():
            for n in ddict[k].keys():
                out["%s_%s" % (k, n)] = ddict[k][n]
    else:
        raise AssertionError("unknown format '%s'" % format)
    return out

# %%


def get_input_from_data(df: pd.DataFrame, test: np.ndarray,
                        headers: list, drename: dict, sortcols: list) -> Tuple[pd.DataFrame, dict]:
    """Transform a dataframe into an dictionary for cable initialization.

    Parameters
    ----------
    df : pandas.DataFrame
        Initial dataframe containing basic geometrical data on conductors.
    test : numpy.ndarray
        Array of imposed test to get relevant index (same size as df).
    headers : list
        List of strings defining columns needed to initialize
        strdcable.cable.strandedcable.
    drename : dict
        Dictionary used to rename columns from strdcable.cable.strandedcable.
    sortcols : list
        List of strings defining columns used to sort lines relative to one
        conductor.

    Returns
    -------
    dtest : pandas.DataFrame
        Dataframe extracted of the initial dataframe according to test.
    dinput : dict
        Dictionary with keys corresponding to headers and values included
        in initial dataframe.

    """
    if test is None:
        dtest = df.copy()
    else:
        dtest = df.loc[test].copy()
    if sortcols is not None:
        dtest = dtest.sort_values(by=sortcols, ignore_index=True)
    dtmp = dtest.to_dict('list')
    dinput = {}
    for k in headers:
        if k in drename.keys():
            lbl = drename[k]
        else:
            lbl = k
        dinput[k] = np.array(dtmp[lbl])
    return dtest, dinput
