"""Functions and classes relative to overheadlines conductors."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

# %%

__DMAT__ = dict(ST1A='steel',
                ST6C='steel',
                heart='steel',
                AL1='alu',
                AL4='alloy',
                FIBRE='optical_fiber',
                COAXE='optical_fiber',
                QUARTE='optical_fiber')

__DCRTE__ = dict(ST1A={'sigmay': 0.9, 'K': 0.33},
                 ST6C={'sigmay': 0.9, 'K': 0.33},
                 heart={'sigmay': 0.9, 'K': 0.33},
                 AL1={'sigmay': 0.675, 'K': 1.0E-04},
                 AL4={'sigmay': 0.825, 'K': 1.0E-04},
                 FIBRE={'sigmay': 0.0, 'K': 0.0},
                 COAXE={'sigmay': 0.0, 'K': 0.0},
                 QUARTE={'sigmay': 0.0, 'K': 0.0})

__CREEP__ = dict(ST1A={'CTPS': 0., 'CT': 0., 'CSIGMA': 0., 'K_CREEP': 0.33},
                 ST6C={'CTPS': 0., 'CT': 0., 'CSIGMA': 0., 'K_CREEP': 0.33},
                 heart={'CTPS': 0., 'CT': 0., 'CSIGMA': 0., 'K_CREEP': 0.33},
                 AL1={'CTPS': 0.248, 'CT': 0.0198,
                      'CSIGMA': 1.6825, 'K_CREEP': 1.54E-06},  # Mezni 2018
                 # AL1={'CTPS': 0.205, 'CT': 0.025,
                 #      'CSIGMA': 1.35, 'K_CREEP': 5.95E-06}, Nakouri 2018
                 AL4={'CTPS': 0.200, 'CT': 0.0177,
                      'CSIGMA': 1.247, 'K_CREEP': 2.40E-06},  # Mezni 2018
                 FIBRE={'CTPS': 0., 'CT': 0., 'CSIGMA': 0., 'K_CREEP': 0.},
                 COAXE={'CTPS': 0., 'CT': 0., 'CSIGMA': 0., 'K_CREEP': 0.},
                 QUARTE={'CTPS': 0., 'CT': 0., 'CSIGMA': 0., 'K_CREEP': 0.},
                 )

__HEAT__ = dict(ST1A=480.0,
                ST6C=480.0,
                heart=480.0,
                AL1=900.0,
                AL4=900.0,
                FIBRE=0.0,
                COAXE=0.0,
                QUARTE=0.0)

# %%


class TablesOHLconductors:
    """Specific for TSO dataset."""

    def __init__(self, dfin: pd.DataFrame) -> None:
        """Init with args.

        Parameters
        ----------
        dfin : pandas.DataFrame
            Dataframe containing basic geometrical data on conductors
            ['material', 'dwires', 'nbwires'].

        Returns
        -------
        None.

        """
        self.dfin = dfin
        # dataframe including output data for layers
        self.dflayers = None
        # dataframe including output data for conductors
        self.dfconductors = None
        return

    def create_dataset(self,
                       cndcol: str = 'cable',
                       sortcols: list = ['layer'], drename: dict = {},
                       headers: list = ['material', 'dwires', 'nbwires'],
                       layerscols: list = ['initial_angle',
                                           'resistivity_20deg',
                                           'mass_resist_20deg',
                                           'density',
                                           'thermal_dilatation_coefficient',
                                           'young_modulus', 'poisson_ratio',
                                           'yield_stress',
                                           'hardening_parameter',
                                           'ultimate_stress',
                                           'ultimate_strain',
                                           'lay_lengths', 'wire_lengths'],
                       conductorscols: list = ['diameter', 'section',
                                               'lineic_mass',
                                               'coefficient_dilatation',
                                               'axial_stiffness',
                                               'minimal_bending_stiffness',
                                               'maximal_bending_stiffness',
                                               'beta_bending',
                                               'rated_strength', 'bimaterial',
                                               'rugosity',
                                               'conductive_section',
                                               'electrical_resistance',
                                               'diameter_steel_core',
                                               'olla_thermal_magnetic_effects'],
                       detailed_material: bool = True,
                       rl: float = 0.5,
                       mu: float = 0.6,
                       formulEI: str = 'PAPAILIOU',
                       formulEP: str = 'rte',
                       nfit: int = 7):
        """Create the database in dfout relative to all conductors included in dfin.

        Parameters
        ----------
        cndcol : str
            Name column containing conductors name.
        sortcols : list, optional
            List of strings defining columns used to sort lines relative to one
            conductor.
        drename : dict, optional
            Dictionary used to rename columns from strdcable.cable.strandedcable.
        headers : list, optional
            List of strings defining columns needed to initialize
            strdcable.cable.strandedcable.
        layerscols : list, optional
            List of strings defining the selected variables to be included in
            the output dataframe for layers.
        conductorscols : list, optional
            List of strings defining the selected variables to be included in
            the output dataframe for conductors.
        detailed_material: bool, optional
            Get the area and volume of each material.
        rl: float, optional
            Giving the laylength ratio based on normative values. The default
            is 0.5.
        mu: float, optional
            Friction coefficient (same for all wires contact). The default is
            0.5.
        formulEI : str, optional
            Formulation used to compute the minimal and maximal bending
            stiffness. The value must be in ['FOTI', 'LEHANNEUR', 'COSTELLO',
            'EPRI', 'PAPAILIOU']. The default is 'PAPAILIOU'.
        formulEP : str, optional
            Formulation used to compute the used for elastoplastic behaviour.
            The value must be in ['default', 'rte']. The default is 'rte'.
        nfit : int, optional
            Integer related to the bending behaviour approximation. The default
            is 7.

        Returns
        -------
        None.

        """
        from strdcable.cable import get_cable
        from strdcable.cable import SimplifiedBending

        if cndcol in self.dfin.columns:
            lcnd = np.unique(self.dfin[cndcol])
        else:
            raise AssertionError("unknown col '%s'" % cndcol)
        ldfl = [None] * len(lcnd)
        ldfc = [None] * len(lcnd)
        for i, ncnd in enumerate(lcnd):
            test = self.dfin[cndcol] == ncnd
            cnd, dftmp = get_cable('conductor', self.dfin, test,
                                   headers, drename, sortcols)
            cnd.set_normative_lay_length(cmin=rl, cmax=1.0 - rl)
            cnd.compute_all(compute_physics=True, set_usual_values=True,
                            formulEI=formulEI, formulEP=formulEP)
            dfout = cnd.export('dataframe')
            ldfl[i] = pd.DataFrame.from_dict(pd.merge(dftmp, dfout[layerscols],
                                                      left_index=True,
                                                      right_index=True))
            cols = conductorscols.copy()
            if detailed_material:
                for n in dfout.columns:
                    if n.find('materials_volume_') != -1 or n.find('materials_section_') != -1:
                        cols.append(n)

            sbd = SimplifiedBending(None, cable=cnd)
            sbd.set_friction(mu)
            dfout['beta_bending'] = sbd.get_limit_curvature(nfit)[0]

            dfc = pd.DataFrame.from_dict(pd.merge(dftmp[cndcol],
                                                  dfout[cols],
                                                  left_index=True,
                                                  right_index=True))
            ldfc[i] = dfc.drop_duplicates(keep='first')
        self.dflayers = pd.concat(ldfl, ignore_index=True)
        self.dfconductors = pd.concat(ldfc, ignore_index=True)
        if drename is not None:
            self.dflayers.rename(columns=drename, inplace=True)
            self.dfconductors.rename(columns=drename, inplace=True)
        if len(self.dfin) != len(self.dflayers):
            raise AssertionError("something wrong in dflayers")
        if len(self.dfconductors) != len(lcnd):
            raise AssertionError("something wrong in dfconductors")
        return

# %%


def laybounds_EN50182(il: int, nw: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Evaluate the min-max lay lengths defined by EN 50182 *(sections 5.5.4 & 5.5.5)*.

    Parameters
    ----------
    il : int
        Layer index (no unit).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    mat : numpy.ndarray
        Array of strings defining the wires material.

    Returns
    -------
    numpy.ndarray
        Min-max ratio normalized to layer radial position.

    """
    nl = len(mat)
    case = "case layer %i : '%s'" % (il, mat[il])
    nsteel = 0
    for m in mat:
        if __DMAT__[m] == 'steel':
            nsteel += 1
    talu = __DMAT__[mat[il]] == 'alu'
    if (il == 0) and (nw == 1):
        bornes = [np.nan, np.nan]
    elif (il == nl - 1) and talu:
        bornes = [10, 14]
    elif talu:
        bornes = [10, 16]
    elif __DMAT__[mat[il]] == 'alloy':
        bornes = [10, 16]
    elif __DMAT__[mat[il]] == 'steel':
        if (il == 0) and (nw == 3):
            bornes = [16, 26]
        elif (nw == 6) and (nsteel in [2, 3]):
            bornes = [16, 26]
        elif (nw == 12) and (nsteel == 3):
            bornes = [14, 22]
        elif (nw == 6) and (nsteel == 4):
            bornes = [17, 25]
        elif (nw == 12) and (nsteel == 4):
            bornes = [16, 22]
        elif (nw == 18) and (nsteel == 4):
            bornes = [14, 18]
        elif nsteel > 4:
            if il == nsteel - 1:
                bornes = [14, 18]
            else:
                bornes = [16, 26]
        elif __DMAT__[mat[0]] == 'optical_fiber':
            # TODO : voir avec le CNER
            bornes = [16, 26]
        else:
            raise AssertionError("EN50182 0: error case (%s)" % case)
    else:
        raise AssertionError("EN50182 1: error case (%s)" % case)
    return np.array(bornes)

# %%


def get_charac_rules(nl: int, dw: np.ndarray, mat: np.ndarray) -> pd.DataFrame:
    """Give the values for different mechanical characteristics based on given rules.

    *(steel NF EN 50189 -- alloy NF EN 50183 -- alu NF EN 60889)*.

    Parameters
    ----------
    nl : int
        Number of layers (no unit).
    dw : numpy.ndarray
        Array of wires diameter (m).
    mat : numpy.ndarray
        Array of strings defining the wires material.

    Returns
    -------
    pandas.DataFrame
        Dataframe including all columns in given rules for each layers.

    """
    from strdcable.config import cfg, __PATH_STRDCABLE__
    path = __PATH_STRDCABLE__+'/'
    df_norm = pd.read_csv(path+cfg['norm_OHL_EN_wires']['file'])

    ldrop = ['dwire_min', 'dwire_max', 'material', 'nature']
    ldf = [None] * nl
    for i in range(nl):
        res = df_norm[(dw[i] <= df_norm.dwire_max) &
                      (dw[i] > df_norm.dwire_min) &
                      (mat[i] == df_norm.material)].copy()
        ldf[i] = res.drop(ldrop, axis=1)
    df_add = pd.concat(ldf, ignore_index=True)
    df_add.rename(columns={"sigma1pct": "sigmapct"}, inplace=True)
    df_add = df_add.assign(epsilonpct=np.where(np.isfinite(df_add.sigmapct),
                                               1.0E-02, np.nan))
    return df_add

# %%


def get_param_SafeBorderLine(mat: np.ndarray,
                             N: Union[float, np.ndarray] = None) -> Tuple[float, float, float, float, bool]:
    """Evaluate the safe border line parameters.

    Orange Book -- chap *'fatigue of overhead conductors'*
    "Safe Border Line Method" 3-30 -- page 214/614.

    Parameters
    ----------
    mat : numpy.ndarray
        Array of strings defining the wires material.
    N : float or numpy.ndarray, optional
        Number of cyclic vibration (no unit).

    Returns
    -------
    a_min : float
        First coefficient of the first part defining safe border line.
    p_min : float
        Second coefficient of the first part defining safe border line.
    a_max : float
        First coefficient of the second part defining safe border line.
    p_max : float
        Second coefficient of the second part defining safe border line.
    test : bool, optional
        Test on validity.

    """
    nb_layers = {'alu': 0, 'alloy': 0, 'steel': 0, 'optical_fiber': 0}
    for m in mat:
        nb_layers[__DMAT__[m]] += 1

    if nb_layers['alu'] == 1 or nb_layers['alloy'] == 1:
        a_min, p_min, a_max, p_max = (730E+06, -0.2, 430E+06, -0.17)
        vlim = 2.00E+07
    elif nb_layers['alu'] > 1 or nb_layers['alloy'] > 1:
        a_min, p_min, a_max, p_max = (450E+06, -0.2, 263E+06, -0.17)
        vlim = 1.56E+07
    else:
        raise AssertionError("wrong number of aluminum layers")

    if type(N) not in [int, list, np.ndarray]:
        return a_min, p_min, a_max, p_max
    else:
        test = N > vlim
        return a_min, p_min, a_max, p_max, test


def get_SafeBorderLine(mat: np.ndarray, N: Union[float, np.ndarray]) -> np.ndarray:
    """Evaluate the safe border line for a given number of cyclic vibration.

    Orange Book -- chap *'fatigue of overhead conductors'*
    "Safe Border Line Method" 3-30 -- page 214/614.

    Parameters
    ----------
    mat : numpy.ndarray
        Array of strings defining the wires material.
    N : float or numpy.ndarray, optional
        Number of cyclic vibration (no unit).

    Returns
    -------
    numpy.ndarray
        Array of poffenberger-swart stresses -- sigma_a(Yb) (MPa).

    """
    if type(N) not in [int, list, np.ndarray]:
        raise AssertionError("type %s uncorrect" % type(N))

    a_min, p_min, a_max, p_max, test = \
        get_param_SafeBorderLine(mat, N)

    s_min = a_min * np.power(np.array(N), p_min)
    s_max = a_max * np.power(np.array(N), p_max)
    return np.where(test, s_max, s_min)


def inverse_SafeBorderLine(mat: np.ndarray, sigma_a: np.ndarray,
                           infinite_life: float = 1.e18) -> np.ndarray:
    """Evaluate the safe border line for a given fictive stress.

    Orange Book -- chap *'fatigue of overhead conductors'*
    "Safe Border Line Method" 3-30 -- page 214/614.

    Parameters
    ----------
    mat : numpy.ndarray
        Array of strings defining the wires material.
    sigma_a : numpy.ndarray
        Array of poffenberger-swart stress (MPa).
    infinite_life : float
        Number of cycles corresponding to an assumed infinite lifetime.


    Returns
    -------
    numpy.ndarray
        Array of number of cyclic vibration (no unit).

    """
    if type(sigma_a) not in [float, np.float32, np.float64, list, np.ndarray]:
        raise AssertionError("type %s uncorrect" % type(sigma_a))

    a_min, p_min, a_max, p_max = get_param_SafeBorderLine(mat)

    sig = np.array(sigma_a)
    lim_inf = np.ones(len(sig)) * infinite_life
    nb_min = np.ones(len(sig)) * infinite_life
    nb_max = np.ones(len(sig)) * infinite_life

    mask = np.argwhere(sig > 0.)
    nb_min[mask] = np.power(10., 1 / p_min * np.log10(sig[mask] / a_min))
    nb_max[mask] = np.power(10., 1 / p_max * np.log10(sig[mask] / a_max))

    return np.minimum(np.maximum(nb_min, nb_max), lim_inf)

# %%


def get_electric_resistance(Aw: np.ndarray, nw: np.ndarray, lw: np.ndarray, Rtwenty: np.ndarray, lcbl: float) -> float:
    """Estimate the electric resistance per unit length.

    Parameters
    ----------
    Aw : numpy.ndarray
        Array of wires section (m2).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    lw : numpy.ndarray
        Array of wires length (m).
    Rtwenty : numpy.ndarray
        Array of wires maximal resistivity at 20°C (Ohm.m).
    lcbl : float
        Cable length (m).

    Returns
    -------
    float
        Electric resistance based on wires characteristics (Ohm/m).

    """
    lrtwenty = np.where(np.logical_or(lw == 0, Rtwenty == 0),
                        np.inf, lw * Rtwenty)
    er = nw / (lrtwenty / Aw)
    return 1. / np.sum(er) / lcbl

# %%


def get_specific_heat_capacity(Aw: np.ndarray, nw: np.ndarray, lw: np.ndarray,
                               rhow: np.ndarray, mat: np.ndarray) -> float:
    """Estimate the specific heat capacity.

    Parameters
    ----------
    Aw : numpy.ndarray
        Array of wires section (m2).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    lw : numpy.ndarray
        Array of wires length (m).
    rhow : numpy.ndarray
        Array of wires density (no unit).
    mat : numpy.ndarray
        Array of strings defining the wires material.

    Returns
    -------
    float
        Electric resistance based on wires characteristics (J/kg/K).

    """
    cw = np.array([__HEAT__[m] for m in mat])
    mw = Aw*nw*lw*rhow*1000.
    return np.sum(mw*cw) / np.sum(mw)

# %%


def get_wind_power_cigre(ymax: Union[float, np.ndarray], freq: Union[float, np.ndarray],
                         D: float, Iv: float) -> np.ndarray:
    """Estimate the wind input power.

    Foti & Martinelli (2018) *An enhanced unified model for the self-damping
    of stranded cables under aeolian vibrations* -- eqs (4) & (5).

    Parameters
    ----------
    ymax : float or numpy.ndarray
        Antinode vibration amplitude (m).
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    D : float
        Cable diameter (m).
    Iv : float
        Turbulence intensity of the wind (no unit).

    Returns
    -------
    numpy.array
        Array of wind input power.

    """
    IL = 0.09
    Bw = np.power(1.+(Iv/IL)**2, -0.5)
    p = [-99.73, 101.62, 0.1627, 0.2256]
    yD = ymax/D
    return Bw*D**4*freq**3*(p[0]*yD**3+p[1]*yD**2+p[2]*yD+p[3])


# %% annealing


param_harvey = {'AL1': dict(a=134., b=-0.24, c=0.241, d=-0.00254),
                'AL4': dict(a=176., b=-0.52, c=0.300, d=-0.00305)}
param_morgan_1979 = {'AL1': dict(a=-8.3, b=0.035, c=9., m=0.285, Wa=56.),
                     'AL4': dict(a=-14.5, b=0.060, c=18., m=0.790, Wa=60.)}
param_morgan_1996 = {'AL1': dict(a=8.65, b=85., c=-4750., d=7.5, Wa=56.),
                     'AL4': dict(a=22.4, b=270., c=-11000., d=4., Wa=60.)}


def _harvey(T: Union[float, np.ndarray], t: Union[float, np.ndarray],
            dw: Union[float, np.ndarray],
            a: float = 176.0, b: float = -0.52,
            c: float = 0.118, d: float = 0.0012,
            **kwargs) -> Union[float, np.ndarray]:
    """Estimate the remaining strength.

    J.R. Harvey (1972) *Effect of elevated temperature operation on the
    strength of aluminum conductors*

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature (°C).
    t : float or numpy.ndarray
        Exposition time (hours).
    dw : float or numpy.ndarray
        Strand diameter (inches).
    a : float, optional
        First Harvey coefficient (page 1170). The default is 176.0 (AL4).
    b : float, optional
        Second Harvey coefficient (page 1170). The default is -0.52 (AL4).
    c : float, optional
        Third Harvey coefficient (page 1170). The default is 0.118 (AL4).
    d : float, optional
        Fourth Harvey coefficient (page 1170). The default is 0.0012 (AL4).

    Returns
    -------
    float or numpy.ndarray
        Remaining strength as a percentage of initial strength.

    """
    if type(T) in [list, np.ndarray]:
        coeff = np.where((a+b*T) > 100., 100., a+b*T)
    else:
        coeff = min(a+b*T, 100.)
    return 100. - coeff * t ** ((c+d*T)/dw)


def _inv_harvey(T: Union[float, np.ndarray], W: Union[float, np.ndarray],
                dw: Union[float, np.ndarray],
                a: float = 176.0, b: float = -0.52,
                c: float = 0.118, d: float = 0.0012,
                **kwargs) -> Union[float, np.ndarray]:
    """Estimate the critical time exposition.

    J.R. Harvey (1972) *Effect of elevated temperature operation on the
    strength of aluminum conductors*

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature (°C).
    W : float or numpy.ndarray
        Resistance loss (%).
    dw : float or numpy.ndarray
        Strand diameter (inches).
    a : float, optional
        First Harvey coefficient (page 1170). The default is 176.0 (AL4).
    b : float, optional
        Second Harvey coefficient (page 1170). The default is -0.52 (AL4).
    c : float, optional
        Third Harvey coefficient (page 1170). The default is 0.118 (AL4).
    d : float, optional
        Fourth Harvey coefficient (page 1170). The default is 0.0012 (AL4).

    Returns
    -------
    float or numpy.ndarray
        Exposition time (hours).

    """
    if type(T) in [list, np.ndarray]:
        coeff = np.where((a+b*T) > 100., 100., a+b*T)
    else:
        coeff = min(a+b*T, 100.)
    return ((100. - W) / coeff) ** (dw/(c+d*T))


def _get_logR(R: float):
    """Give the usual log for Morgan.

    Parameters
    ----------
    R : float or numpy.ndarray
        Ratio corresponding to wire drawing (no unit).

    Returns
    -------
    float or numpy.ndarray
        Log divided by 80.

    """
    return np.log(R/80.)


def _morgan_1979(T: Union[float, np.ndarray], t: Union[float, np.ndarray],
                 R: Union[float, np.ndarray],
                 a: float = -14.5, b: float = 0.06, c: float = 18.0,
                 m: float = 0.79, Wa: float = 60.0,
                 **kwargs) -> Union[float, np.ndarray]:
    """Estimate the remaining strength.

    V.T. Morgan (1979) *The loss of tensile strength of hard-drawn conductors
    by annealing in service*

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature (°C).
    t : float or numpy.ndarray
        Exposition time (hours).
    R : float or numpy.ndarray
        Ratio corresponding to wire drawing (no unit).
    a : float, optional
        First Morgan coefficient (page 702). The default is -14.5 (AL4).
    b : float, optional
        Second Morgan coefficient (page 702). The default is 0.06 (AL4).
    c : float, optional
        Third Morgan coefficient (page 702). The default is 18.0 (AL4).
    m : float, optional
        Fourth Morgan coefficient (page 702). The default is 0.79 (AL4).
    Wa: float, optional
        Fifth Morgan coefficient (page 702). The default is 60.0 (AL4).

    Returns
    -------
    float or numpy.ndarray
        Remaining strength as a percentage of initial strength.

    """
    return Wa * (1.-np.exp(-np.exp(a + m*np.log(t) + b*T + c*_get_logR(R))))


def _inv_morgan_1979(T: Union[float, np.ndarray], W: Union[float, np.ndarray],
                     R: Union[float, np.ndarray],
                     a: float = -14.5, b: float = 0.06, c: float = 18.0,
                     m: float = 0.79, Wa: float = 60.0,
                     **kwargs) -> Union[float, np.ndarray]:
    """Estimate the critical time exposition.

    V.T. Morgan (1979) *The loss of tensile strength of hard-drawn conductors
    by annealing in service*

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature (°C).
    W : float or numpy.ndarray
        Resistance loss (%).
    R : float or numpy.ndarray
        Ratio corresponding to wire drawing (no unit).
    a : float, optional
        First Morgan coefficient (page 702). The default is -14.5 (AL4).
    b : float, optional
        Second Morgan coefficient (page 702). The default is 0.06 (AL4).
    c : float, optional
        Third Morgan coefficient (page 702). The default is 18.0 (AL4).
    m : float, optional
        Fourth Morgan coefficient (page 702). The default is 0.79 (AL4).
    Wa: float, optional
        Fifth Morgan coefficient (page 702). The default is 60.0 (AL4).

    Returns
    -------
    float or numpy.ndarray
        Exposition time (hours).

    """
    return np.exp((np.log(-np.log(1.-W/Wa))-(a + b*T + c*_get_logR(R)))/m)


def _morgan_1996(T: Union[float, np.ndarray], t: Union[float, np.ndarray],
                 R: Union[float, np.ndarray],
                 a: float = 22.4, b: float = 270.0, c: float = -11000.0,
                 d: float = 4.0, Wa: float = 60.0,
                 **kwargs) -> Union[float, np.ndarray]:
    """Estimate the remaining strength.

    V.T. Morgan (1996) *Effect of elevated temperature operation on the tensile
    strength of overhead conductors*

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature (°C).
    t : float or numpy.ndarray
        Exposition time (hours).
    R : float or numpy.ndarray
        Ratio corresponding to wire drawing (no unit).
    a : float, optional
        First Morgan coefficient (page 347). The default is 22.4 (AL4).
    b : float, optional
        Second Morgan coefficient (page 347). The default is 270.0 (AL4).
    c : float, optional
        Third Morgan coefficient (page 347). The default is -11000.0 (AL4).
    d : float, optional
        Fourth Morgan coefficient (page 347). The default is 4.0 (AL4).
    Wa: float, optional
        Fifth Morgan coefficient (page 347). The default is 60.0 (AL4).

    Returns
    -------
    float or numpy.ndarray
        Remaining strength as a percentage of initial strength.

    """
    Tk = T + 273.15
    return Wa*(1.-np.exp(-np.exp(a+(b/Tk)*np.log(t)+c/Tk+d*_get_logR(R))))


def _inv_morgan_1996(T: Union[float, np.ndarray], W: Union[float, np.ndarray],
                     R: Union[float, np.ndarray],
                     a: float = 22.4, b: float = 270.0, c: float = -11000.0,
                     d: float = 4.0, Wa: float = 60.0,
                     **kwargs) -> Union[float, np.ndarray]:
    """Estimate the critical time exposition.

    V.T. Morgan (1996) *Effect of elevated temperature operation on the tensile
    strength of overhead conductors*

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature (°C).
    W : float or numpy.ndarray
        Resistance loss (%).
    R : float or numpy.ndarray
        Ratio corresponding to wire drawing (no unit).
    a : float, optional
        First Morgan coefficient (page 347). The default is 22.4 (AL4).
    b : float, optional
        Second Morgan coefficient (page 347). The default is 270.0 (AL4).
    c : float, optional
        Third Morgan coefficient (page 347). The default is -11000.0 (AL4).
    d : float, optional
        Fourth Morgan coefficient (page 347). The default is 4.0 (AL4).
    Wa: float, optional
        Fifth Morgan coefficient (page 347). The default is 60.0 (AL4).

    Returns
    -------
    float or numpy.ndarray
        Exposition time (hours).

    """
    Tk = T + 273.15
    return np.exp((np.log(-np.log(1.-W/Wa))-(a+c/Tk+d*_get_logR(R)))/(b/Tk))


def _get_limit_annealing(method: str, mat: str) -> float:
    """Select correct annealing limit for a material.

    Parameters
    ----------
    method : str
        Method used to obtain limit annealing.
        Must be in ['harvey', 'morgan_1979', 'morgan_1996']
    mat : str
        Name of material. Must be in ['ALU', 'ALMELEC', 'ACIER']

    Returns
    -------
    float
        Limit annealing of chosen material.

    """
    if method == 'harvey':
        limits = {'ALU': 95.,
                  'ALMELEC': 95.,
                  'ACIER': 250.}
    elif method in ['morgan_1979', 'morgan_1996']:
        limits = {'ALU': 80.,
                  'ALMELEC': 50.,
                  'ACIER': 250.}
    return limits[mat]


def _get_annealing(temp: Union[float, np.ndarray],
                   time: Union[float, np.ndarray],
                   cwire: float, mat: str,
                   method: str) -> Union[float, np.ndarray]:
    """Estimate the remaining strength.

    Parameters
    ----------
    temp : float or numpy.ndarray
        Temperature (°C).
    time : float or numpy.ndarray
        Exposition time (hours).
    cwire : float or numpy.ndarray
        Third input for annealing method.
    mat : str
        Name of material.
    method : str
        Method used to obtain annealing values.
        Must be in ['harvey', 'morgan_1979', 'morgan_1996']

    Returns
    -------
    W: float or numpy.ndarray
        Remaining strength as a percentage of initial strength.

    """
    if method == 'harvey':
        fct = _harvey
        inv_fct = _inv_harvey
        p = param_harvey
    elif method == 'morgan_1979':
        fct = _morgan_1979
        inv_fct = _inv_morgan_1979
        p = param_morgan_1979
    elif method == 'morgan_1996':
        fct = _morgan_1996
        inv_fct = _inv_morgan_1996
        p = param_morgan_1996
    else:
        raise NotImplementedError("unknown method for annealing")

    W = np.zeros(len(temp))
    if mat in p.keys():
        for i, T in enumerate(temp):
            if i > 0:
                teq = inv_fct(T, W[i-1], cwire, **p[mat])
            else:
                teq = 0.
            t = teq + time[i]
            W[i] = fct(T, t, cwire, **p[mat])
    return W

# %% creep


def _get_stress_creep(sigma: Union[float, np.ndarray],
                      eps: Union[float, np.ndarray],
                      T: Union[float, np.ndarray],
                      deltaT: Union[float, np.ndarray],
                      tps: Union[float, np.ndarray],
                      young: Union[float, np.ndarray],
                      alpha: Union[float, np.ndarray],
                      k: Union[float, np.ndarray],
                      ctps: Union[float, np.ndarray],
                      cT: Union[float, np.ndarray],
                      csigma: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Estimate the creep stress.

    Kopsidas et al (2016) *Overhead line design considerations for conductor creep*

    Parameters
    ----------
    sigma : float or numpy.ndarray
        Tensile stress (Pa).
    eps : float or numpy.ndarray
        Tensile strain (no unit).
    T : float or numpy.ndarray
        Temperature (°C).
    deltaT : float or numpy.ndarray
        Temperature difference (°C).
    tps : float or numpy.ndarray
        Exposition time (hours).
    young : float or numpy.ndarray
        Wires young modulus (Pa).
    alpha : float or numpy.ndarray
        Coefficient dilatation (1/°C).
    k : float or numpy.ndarray
        First Kopsidas coefficient (page 2426).
    ctps : float or numpy.ndarray
        Second Kopsidas coefficient (page 2426).
    cT : float or numpy.ndarray
        Third Kopsidas coefficient (page 2426).
    csigma : float or numpy.ndarray
        Fourth Kopsidas coefficient (page 2426).

    Returns
    -------
    float or numpy.ndarray
        Total stress including creep (Pa).

    """
    sig_fl = young*k*np.exp(cT*T)*tps**ctps*(sigma/9.81*1.0E-06)**csigma
    return sig_fl + sigma - young*(eps-alpha*deltaT)


def _get_force_creep_ASCR(H: Union[float, np.ndarray],
                          eps: Union[float, np.ndarray],
                          T: Union[float, np.ndarray],
                          deltaT: Union[float, np.ndarray],
                          tps: Union[float, np.ndarray],
                          EA: Union[float, np.ndarray],
                          alpha: Union[float, np.ndarray],
                          RTS: Union[float, np.ndarray],
                          Thigh: float = 35.0,
                          isstrain: bool = False):
    """Estimate the creep strain or force.

    Kopsidas et al (2016) *Overhead line design considerations for conductor creep*

    Parameters
    ----------
    sigma : float or numpy.ndarray
        Tensile stress (Pa).
    eps : float or numpy.ndarray
        Tensile strain (no unit).
    T : float or numpy.ndarray
        Temperature (°C).
    deltaT : float or numpy.ndarray
        Temperature difference (°C).
    tps : float or numpy.ndarray
        Exposition time (hours).
    EA : float or numpy.ndarray
        Axial stiffness (N).
    alpha : float or numpy.ndarray
        Coefficient dilatation (1/°C).
    RTS : float or numpy.ndarray
        Rated strength (N).
    Thigh : float or numpy.ndarray
        Temperature (°C). The default is 35.0.
    isstrain : bool, optional
        Select the output. The default is False.

    Returns
    -------
    float or numpy.ndarray
        Total force or strain depending on .

    """
    if type(T) in [list, np.ndarray]:
        eps_cr = np.where(T >= Thigh,
                          0.24E-06 * (H/RTS*100.) * (T-15.) * tps**0.16,
                          2.4E-06 * (H/RTS*100.)**1.3 * tps**0.16)
    else:
        if T >= Thigh:
            eps_cr = 0.24E-06 * (H/RTS*100.) * (T-15.) * tps**0.16
        else:
            eps_cr = 2.4E-06 * (H/RTS*100.)**1.3 * tps**0.16
    if isstrain:
        return eps_cr
    else:
        return EA*eps_cr + H - EA*(eps-alpha*deltaT)
