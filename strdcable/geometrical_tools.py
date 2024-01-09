"""Functions relative to geometrical properties of cables."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple

# %%


def get_radial_positions(dw: np.ndarray) -> np.ndarray:
    """Evaluate the radial position of each layer.

    Parameters
    ----------
    dw : numpy.ndarray
        Array of wires diameter (m).

    Returns
    -------
    numpy.ndarray
        Array of radial positions (m).

    """
    add = np.zeros(dw.shape)
    add[1:] = 0.5 * (dw[:-1] + dw[1:])
    return np.cumsum(add)


def evaluate_rugosity(dw: np.ndarray, rw: np.ndarray, formul: str) -> float:
    """Evaluate the external rugosity of the cable.

    Parameters
    ----------
    dw : numpy.ndarray
        Array of wires diameter (m).
    rw : numpy.ndarray
        Array of radial positions (m).
    formul : str
        Specified formulation. the value must be in ['RTE'].

    Returns
    -------
    float
        External rugosity (no unit).

    """
    # Dcable = 2*rw[-1] + dw[-1]
    if formul == 'RTE':  # NT-RD-CNER-DL-SLA-20-00215 -- page 32/62
        return 0.5*dw[-1]/(2.*rw[-1])
    else:
        raise AssertionError("unknown formulation '%s'" % formul)


def evaluate_sections(Aw: np.ndarray, nw: np.ndarray, mat: np.ndarray,
                      lcond: list = ['AL1', 'AL4']) -> Tuple[dict, float, float]:
    """Evaluate the area per material and in total.

    Parameters
    ----------
    Aw : numpy.ndarray
        Array of wires section (m2).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    mat : numpy.ndarray
        Array of strings defining the wires material.

    Returns
    -------
    smat : dict
        Dictionary including section per material (m2).
    stot : float
        Total section (m2).
    scond : float
        Conductive section (m2)

    """
    stot = np.sum(Aw*nw)
    smat = {}
    scond = 0.
    for m in np.unique(mat):
        t = np.argwhere(mat == m)
        smat[m] = np.sum(Aw[t]*nw[t])
        if m in lcond:
            scond += smat[m]
    return smat, stot, scond


def evaluate_volumes(Aw: np.ndarray, nw: np.ndarray, lw: np.ndarray, mat: np.ndarray) -> Tuple[dict, float]:
    """Evaluate the volume per material and in total.

    Parameters
    ----------
    Aw : numpy.ndarray
        Array of wires section (m2).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    lw : numpy.ndarray
        Array of wires length (m).
    mat : numpy.ndarray
        Array of strings defining the wires material.

    Returns
    -------
    vmat : dict
        Dictionary including volume per material (m3).
    vtot : float
        Total volume (m3).

    """
    vtot = np.sum(Aw*nw*lw)
    vmat = {}
    for m in np.unique(mat):
        t = np.argwhere(mat == m)
        vmat[m] = np.sum(Aw[t]*nw[t]*lw[t])
    return vmat, vtot


def get_lay_angles(lay: np.ndarray, rw: np.ndarray) -> np.ndarray:
    """Evaluate the lay angle for each layer.

    Parameters
    ----------
    lay : numpy.ndarray
        Array of lay length for each layer (m).
    rw : numpy.ndarray
        Array of radial positions (m).

    Returns
    -------
    numpy.ndarray
        Array of lay angle (radians).

    """
    return np.arctan(2. * np.pi * rw / lay)


def lay2length(lay: np.ndarray, rw: np.ndarray, lcbl: float) -> np.ndarray:
    """Evaluate the wires length for each layer.

    Parameters
    ----------
    lay : numpy.ndarray
        Array of lay length for each layer (m).
    rw : numpy.ndarray
        Array of radial positions (m).
    lcbl : float
        Cable length (m).

    Returns
    -------
    lw : numpy.ndarray
        Array of wires length (m).

    """
    lw = 2. * np.pi * (lcbl/lay) * np.sqrt(rw**2+lay**2/(2.*np.pi)**2)
    lw[0] = lcbl
    return lw


def length2lay(lw: np.ndarray, rw: np.ndarray) -> np.ndarray:
    """Evaluate the lay length for each layer.

    Parameters
    ----------
    lw : numpy.ndarray
        Array of lay length for each layer (m).
    rw : numpy.ndarray
        Array of radial positions (m).

    Returns
    -------
    numpy.ndarray
        Array of lay length for each layer (m).

    """
    r = (rw[1:]**2) / ((lw[1:] / lw[0]) ** 2 - 1.)
    return np.append([np.nan], 2. * np.pi * np.sqrt(r))

# %%


def get_normalized_lay(bnds: np.ndarray, cmin: float = 0.5, cmax: float = 0.5) -> np.ndarray:
    """Give the lay length for each layer relative to a given rule.

    Parameters
    ----------
    bnds : numpy.ndarray
        Array of min-max lay length for a given rule (m).
    cmin : float, optional
        Multiplicative coefficient for minimal rule (no unit). The default
        is 0.5.
    cmax : float, optional
        Multiplicative coefficient for maximal rule (no unit). The default
        is 0.5.

    Returns
    -------
    numpy.ndarray
        Array of lay length for each layer (m).

    """
    if np.all(cmin+cmax == 1.):
        return cmin * bnds[:, 0] + cmax * bnds[:, 1]
    else:
        raise AssertionError("sum of mult coeff must be 1 (=%s here)"
                             % str(cmin+cmax))

# %%


def get_initial_wire_index(nw: np.ndarray) -> np.ndarray:
    """Give the first wire index of each layer.

    Parameters
    ----------
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).

    Returns
    -------
    numpy.ndarray
        Array of index of the first wire for each layer.

    """
    return np.append([0], np.cumsum(nw, dtype=np.int))
