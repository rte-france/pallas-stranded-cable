"""Functions relative to mechanical properties of cables."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple
from typing import Union

import numpy as np
from scipy.linalg import solve

# %%
__DCDEFAULT__ = dict(ST1A={'sigmay': 1.0, 'K': 1.0},
                     ST6C={'sigmay': 1.0, 'K': 1.0},
                     heart={'sigmay': 1.0, 'K': 1.0},
                     AL1={'sigmay': 1.0, 'K': 1.0E-04},
                     AL4={'sigmay': 1.0, 'K': 1.0E-04},
                     FIBRE={'sigmay': 0.0, 'K': 0.0},
                     COAXE={'sigmay': 0.0, 'K': 0.0},
                     QUARTE={'sigmay': 0.0, 'K': 0.0})

# %%


def get_layer_axial_stiffness(young: np.ndarray, Aw: np.ndarray, nw: np.ndarray, layangles: np.ndarray) -> np.ndarray:
    """Evaluate the axial stiffness of each layer.

    A. Cloutier (2013) *Stick-slip mechanical models for overhead
    electrical conductors in bending*.

    Parameters
    ----------
    young : numpy.ndarray
        Array of wires young modulus (Pa).
    Aw : numpy.ndarray
        Array of wires section (m2).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    layangles : numpy.ndarray
        Array of lay angle (radians).

    Returns
    -------
    numpy.ndarray
        Array of axial stiffness of each layer (Pa.m2).

    """
    return young * Aw * nw * np.cos(layangles) ** 3


def get_rated_strength(sigmau: np.ndarray, Aw: np.ndarray, nw: np.ndarray) -> float:
    """Evaluate the cable rated strength.

    A. Cloutier (2013) *Stick-slip mechanical models for overhead
    electrical conductors in bending*.

    Parameters
    ----------
    sigmau : numpy.ndarray
        Array of wires ultimate stress (Pa).
    Aw : numpy.ndarray
        Array of wires section (m2).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).

    Returns
    -------
    float
        Cable rated strength (N).

    """
    Flayers = sigmau * Aw * nw
    return np.sum(Flayers)

# %%


def get_EI_bounds_layers(young: np.ndarray, Aw: np.ndarray, nw: np.ndarray, layangles: np.ndarray,
                         rw: np.ndarray, Iw: np.ndarray, cpoiss: np.ndarray = None,
                         formul: str = 'PAPAILIOU') -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the bending stiffness of each layer using its radial position.

    Foti & Martinelli (2016) *Mechanical modeling of metallic strands subjected
    to tension, torsion and bending*.

    A. Cloutier (2013) *Stick-slip mechanical models for overhead electrical
    conductors in bending* -- section 2.3 "FRICTIONLESS BENDING BEHAVIOR".

    Parameters
    ----------
    young : numpy.ndarray
        Array of wires young modulus (Pa).
    Aw : numpy.ndarray
        Array of wires section (m2).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    layangles : numpy.ndarray
        Array of lay angle (radians).
    rw : numpy.ndarray
        Array of radial positions (m).
    Iw : numpy.ndarray
        Array of wires inertia (m4).
    cpoiss : numpy.ndarray, optional
        Array of wires Poisson ratio (no unit).
    formul : str, optional
            Formulation used to compute the minimal bending stiffness. The
            value must be in ['FOTI', 'LEHANNEUR', 'COSTELLO', 'EPRI', 'PAPAILIOU'].
            The default is 'PAPAILIOU'.

    Returns
    -------
    Bmin : numpy.ndarray
        Array of minimal bending stiffness for each layer (N.m2).
    Bcompl : numpy.ndarray
        Array of complementary bending stiffness for each layer (N.m2).

    """
    cang = np.cos(layangles)
    sang = np.sin(layangles)
    c2ang = cang**2
    s2ang = sang**2

    Bcompl = 0.5 * young * Aw * nw * (rw**2) * (cang**3)

    EIlayers = young * Iw * nw
    EIcos = EIlayers * cang
    if formul in ['FOTI', 'LEHANNEUR', 'COSTELLO']:
        if cpoiss is None:
            raise AssertionError("'cpoiss' needed in formula %s" % formul)
        elif formul == 'FOTI':
            Bmin = 0.5 * EIcos * (1. + c2ang + sang / (1. + cpoiss))
        elif formul == 'LEHANNEUR':
            Bmin = EIcos * (1. - cpoiss * s2ang / (2. * (1. + cpoiss)))
        elif formul == 'COSTELLO':
            Bmin = EIcos * 2. / (2. + cpoiss * s2ang)
    elif formul == 'EPRI':
        Bmin = EIlayers
    elif formul == 'PAPAILIOU':
        Bmin = EIcos
    else:
        raise AssertionError("unknown formula %s" % formul)
    return Bmin, Bcompl


def RebuffelLehanneur_limits(nl: int, rw: np.ndarray, Fw: np.ndarray, lambd: np.ndarray, theta: np.ndarray,
                             rF: np.ndarray, rcang: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give bending curvature and moment limits according to Rebuffel and Lehanneur assumptions.

    A. Cloutier (2013) *Stick-slip mechanical models for overhead electrical
    conductors in bending* -- section 3.4.5 "Slip conditions (SC1) from
    Rebuffel-Lehanneur (1949)".

    Parameters
    ----------
    nl : int
        Number of layers (no unit).
    rw : numpy.ndarray
        Array of radial positions (m).
    Fw : numpy.ndarray
        Array of one wire force for each layers (N).
    lambd : numpy.ndarray
        See strdcable.physical_tools.get_bending_limits.
    theta : numpy.ndarray
        See strdcable.physical_tools.get_bending_limits.
    rF : numpy.ndarray
        See strdcable.physical_tools.get_bending_limits.
    rcang : numpy.ndarray
        See strdcable.physical_tools.get_bending_limits.

    Returns
    -------
    kappa_b : numpy.ndarray
        Array of incipient slip curvature for each layer (1/m).
    kappa_t : numpy.ndarray
        Array of total slip curvature for each layer (1/m).
    Mrf : numpy.ndarray
        Array of residual moment for each layer (N.m).

    """
    def sumexp(i):
        return np.sum(np.sin(theta[i])*(np.exp(lambd[i]*theta[i])-1.))

    c1 = (Fw/rF)[1:] / rw[1:]
    c2 = np.array([sumexp(i) for i in range(nl-1)])

    # eq (3.8) in Cloutier
    kappa_b = c1 * lambd
    # eq (3.9) in Cloutier
    kappa_t = c1 * (np.exp(0.5*lambd) - 1.)
    # eq (3.10) in Cloutier
    Mrf = (2.*Fw*rcang)[1:] * c2
    return kappa_b, kappa_t, Mrf


def nonsympressure_limits(nl: int, layangles: np.ndarray, rw: np.ndarray, dw: np.ndarray, nw: np.ndarray,
                          mu: Union[float, np.ndarray], Fw: np.ndarray, lambd: np.ndarray, theta: np.ndarray,
                          rF: np.ndarray, formul: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give bending curavture and moment limits without assumption of symmetry of pressure between two layers.

    A. Cloutier (2013) *Stick-slip mechanical models for overhead electrical
    conductors in bending* -- section 3.4.7 "Slip conditions (SC3)".

    Parameters
    ----------
    nl : int
        Number of layers (no unit).
    layangles : numpy.ndarray
        Array of lay angle (radians).
    rw : numpy.ndarray
        Array of radial positions (m).
    dw : numpy.ndarray
        Array of wires diameter (m).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    mu : float or numpy.array
        Friction coefficient for each interlayer (no unit).
    Fw : numpy.ndarray
        Array of one wire force for each layers (N).
    lambd : numpy.ndarray
        See strdcable.physical_tools.get_bending_limits.
    theta : numpy.ndarray
        See strdcable.physical_tools.get_bending_limits.
    rF : numpy.ndarray
        See strdcable.physical_tools.get_bending_limits.
    formul : str
        Formulation used for the bending behaviour. The value must be in
        ['SC1', 'SC2', 'SC3', 'SC5', 'FOTI'].

    Returns
    -------
    kappa_b : numpy.ndarray
        Array of incipient slip curvature for each layer (1/m).
    kappa_t : numpy.ndarray
        Array of total slip curvature for each layer (1/m).
    Mrf : numpy.ndarray
        Array of residual moment for each layer (N.m).

    """
    cang = np.cos(layangles)
    s2ang = np.sin(layangles)**2

    r = 0.5*dw[1:]
    # eq (3.18) in Cloutier
    b = (s2ang[1:] / r)[::-1]
    # eq (3.19) in Cloutier
    c = get_muci(layangles, r, mu, formul)
    rFrw = rF * rw
    ilambd = lambd[::-1]

    a = [None] * (nl-1)
    sb = np.ones(nl-1) * np.nan
    st = np.ones(nl-1) * np.nan
    sm = np.ones(nl-1) * np.nan
    for m in range(nl-1):
        i = nl-1-m-1
        dlbd = (lambd[::-1]-lambd[i])[:m]

        # eq (3.29)
        a[i] = np.zeros(m+1)
        for j in range(m):
            ak = np.array([akj[j] for akj in (a[::-1])[j:m]])
            a[i][j] = c[i]/dlbd[j]*np.sum(b[j:m] * ak)
        a[i][m] = Fw[i+1]-np.sum(a[i][:m])

        ilbd = ilambd[:m+1]
        # sum eq (3.34)
        sb[i] = np.sum(a[i] * ilbd)
        # sum eq (3.36)
        st[i] = np.sum(a[i] * np.exp(ilbd*np.pi/2.))

        def stmp(j):
            t = theta[i][j]
            return (np.sum(a[i]*np.exp(ilbd*t))-Fw[i+1])*rw[i+1]*np.sin(t)
        # sum eq (3.37)
        sm[i] = np.sum([stmp(j) for j in range(nw[i+1]//2)])

    # eq (3.34)
    kappa_b = sb/rFrw[1:]
    # eq (3.36)
    kappa_t = (st-Fw[1:])/rFrw[1:]
    # eq (3.37)
    Mrf = 2.*cang[i+1]*sm
    return kappa_b, kappa_t, Mrf


def get_muci(layangles: np.ndarray, r: np.ndarray, mu: np.ndarray, formul: str) -> np.ndarray:
    """Get frictional coefficient.

    A. Cloutier (2013) *Stick-slip mechanical models for overhead electrical
    conductors in bending* -- eq (3.19).

    Parameters
    ----------
    layangles : numpy.ndarray
        Array of lay angle (radians).
    r : numpy.ndarray
        Array of wires radius (m).
    mu : float or numpy.array
        Friction coefficient for each interlayer (no unit).
    formul : str
        Formulation used for the bending behaviour. The value must be in
        ['SC1', 'SC2', 'SC3', 'SC5', 'FOTI'].

    Returns
    -------
    numpy.ndarray
        Array of ci coefficient in eq (3.19).

    """
    if formul == 'SC2':  # eq (3.47) in Cloutier
        cmult = mu[:-1]+mu[1:]
    elif formul == 'SC3':  # eq (3.19) in Cloutier
        cmult = (mu[:-1]-mu[1:]*np.cos(layangles[1:-1]+layangles[2:]))
    elif formul == 'SC5':  # eq (3.46) in Cloutier
        cmult = mu[:-1]
    elif formul == 'FOTI':
        cmult = mu[:-1]*(1.+mu[1:]/(mu[:-1]+mu[1:]))
    else:
        raise AssertionError("unknown input '%s'" % formul)
    return r[:-1] / np.sin(layangles)[1:-1] * cmult


def get_bending_limits(Flayers: np.ndarray, young: np.ndarray,
                       layangles: np.ndarray, rw: np.ndarray,
                       Aw: np.ndarray, dw: np.ndarray, nw: np.ndarray,
                       mu: Union[float, np.ndarray],
                       formul: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give bending curavture and moment limits.

    A. Cloutier (2013) *Stick-slip mechanical models for overhead electrical
    conductors in bending*

    Parameters
    ----------
    Flayers : numpy.ndarray
        Array of layers force (N).
    young : numpy.ndarray
        Array of wires young modulus (Pa).
    layangles : numpy.ndarray
        Array of lay angle (radians).
    rw : numpy.ndarray
        Array of radial positions (m).
    Aw : numpy.ndarray
        Array of wires section (m2).
    dw : numpy.ndarray
        Array of wires diameter (m).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    mu : float or numpy.array
        Friction coefficient for each interlayer (no unit).
    formul : str
        Formulation used for the bending behaviour. The value must be in
        ['SC1', 'SC2', 'SC3', 'SC5', 'FOTI'].

    Returns
    -------
    kappa_b : numpy.ndarray
        Array of incipient slip curvature for each layer (1/m).
    kappa_t : numpy.ndarray
        Array of total slip curvature for each layer (1/m).
    Mrf : numpy.ndarray
        Array of residual moment for each layer (N.m).

    """
    nl = len(young)
    Fw = Flayers / nw

    cang = np.cos(layangles)

    rF = Aw*young*cang**2
    # eq (3.17) in Cloutier
    lambd = np.sin(layangles)[1:] * mu
    theta = [np.array([np.pi*((2*j-1)/nb-0.5) for j in range(nb//2)])
             for nb in nw[1:]]

    if formul == 'SC1':
        return RebuffelLehanneur_limits(nl, rw,
                                        Fw, lambd, theta, rF, rw*cang)
    elif formul in ['SC2', 'SC3', 'SC5', 'FOTI']:
        return nonsympressure_limits(nl, layangles, rw, dw, nw, mu,
                                     Fw, lambd, theta, rF, formul)
    else:
        raise AssertionError("unknown input '%s'" % formul)


def linearized_bending_law(Bmin: np.ndarray, Bcompl: np.ndarray, Mrf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Give the evolution of the bending stiffness due to successive slips of the cable interlayers.

    A. Cloutier (2013) *Stick-slip mechanical models for overhead electrical
    conductors in bending* -- section "2.5.2 Bending moment"

    Parameters
    ----------
    Bmin : numpy.ndarray
        Array of minimal bending stiffness for each layer (N.m2).
    Bcompl : numpy.ndarray
        Array of complementary bending stiffness for each layer (N.m2).
    Mrf : numpy.ndarray
        Array of residual moment for each layer (N.m).

    Returns
    -------
    B : numpy.ndarray
        Array of tangential bending stiffness of the cable (N.m2).
    kappa : numpy.ndarray
        Array of limit bending curvature defining B (1/m).

    """
    B = (np.sum(Bmin)+np.cumsum(Bcompl))[::-1]
    kappa = np.sort(Mrf / Bcompl[1:])
    return B, kappa


def linearized_macro_bending(B: np.ndarray, kappa: np.ndarray) -> Tuple[float, float]:
    """Give the limit bending moment and curvature based on successive slips.

    Foti & Martinelli (2018) *An enhanced unified model for the self-damping
    of stranded cables under aeolian vibrations*.

    Parameters
    ----------
    B : numpy.ndarray
        Array of tangential bending stiffness of the cable (N.m2).
    kappa : numpy.ndarray
        Array of limit bending curvature defining B (1/m).

    Returns
    -------
    M0 : float
        Limit bending moment (N.m).
    kappa0 : float
        Limit curvature (1/m).

    """
    kappa_num = np.append([0.], kappa)
    Mnum = np.zeros(kappa_num.shape)
    for i in range(1, len(kappa_num)):
        Mnum[i] = Mnum[i-1] + (kappa_num[i]-kappa_num[i-1])*B[i-1]
    kappa0 = (Mnum[-1]-kappa_num[-1]*B[-1])/(B[0]-B[-1])
    M0 = B[0]*kappa0
    return M0, kappa0

# %%


def get_lineic_mass(Aw: np.ndarray, nw: np.ndarray, lw: np.ndarray, rhow: np.ndarray, lcbl: float) -> float:
    """Evaluate the volume per material and in total.

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
    lcbl : float
        Cable length (m).

    Returns
    -------
    float
        Cable lineic mass (kg/m).

    """
    return np.sum(Aw*nw*lw*rhow*1000.)/lcbl

# %%


def get_coeff_dilatation(alpha: np.ndarray, young: np.ndarray, Aw: np.ndarray, nw: np.ndarray,
                         layangles: np.ndarray) -> float:
    """Evaluate the coefficient dilatation [norm CEI61597-1995].

    Parameters
    ----------
    alpha : numpy.ndarray
        Array of coefficient dilatation (1/°C).
    young : numpy.ndarray
        Array of wires young modulus (Pa).
    Aw : numpy.ndarray
        Array of wires section (m2).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    layangles : numpy.ndarray
        Array of lay angle (radians).

    Returns
    -------
    float
        Cable coefficient dilatation (1/°C).

    """
    alpha_all = alpha * young * Aw * nw * np.cos(layangles) ** 3
    return np.sum(alpha_all) / np.sum(get_layer_axial_stiffness(young, Aw, nw,
                                                                layangles))

# %%


def get_elastocoeff(young: np.ndarray, epsilonpct: np.ndarray, sigmapct: np.ndarray,
                    epsilonu: np.ndarray, sigmau: np.ndarray, mat: np.ndarray,
                    dictmult: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Give the relevant elastoplastic characteristics depending on limit strains and stresses.

    Parameters
    ----------
    young : numpy.ndarray
        Array of wires young modulus (Pa).
    epsilonpct : numpy.ndarray
        Array of limit elastic strain (no unit).
    sigmapct : numpy.ndarray
        Array of limit elastic stress (Pa).
    epsilonu : numpy.ndarray
        Array of ultimate strain (no unit).
    sigmau : numpy.ndarray
        Array of ultimate stress (Pa).
    mat : numpy.ndarray
        Array of strings defining the wires material.
    dictmult : dict
        Dictionary of multiplicative coefficients for relevant characteristics
        (no unit).

    Returns
    -------
    sigmay : numpy.ndarray
        Array of yield stress (Pa).
    hardening : numpy.ndarray
        Array of hardening parameter (Pa).

    """
    mSY = np.zeros(len(young))
    mK = np.zeros(len(young))
    for i, m in enumerate(mat):
        try:
            mSY[i] = dictmult[m]['sigmay']
            mK[i] = dictmult[m]['K']
        except KeyError:
            mSY[i] = 1.0
            mK[i] = 1.0
    sigmay = sigmapct * mSY
    hardening = (sigmau - sigmapct) / (epsilonu - epsilonpct) * mK
    msk = np.where(np.isnan(sigmay))
    sigmay[msk] = sigmau[msk] * mSY[msk]
    hardening[msk] = young[msk] * mK[msk]
    return sigmay, hardening

# %%


def get_wires_axialstress(epsilon: np.ndarray, eP: np.ndarray, xP: np.ndarray,
                          young: np.ndarray, sigmay: np.ndarray, hardening: np.ndarray,
                          deltaT: np.ndarray = 0., alpha: np.ndarray = 0.,
                          tol: float = 1.0E-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give the axial stress in each layers for given axial strains.

    Evaluation using a classical elastoplastic algorithm for kinematic hardening.

    Parameters
    ----------
    epsilon : numpy.ndarray
        Array of axial strains for each layer (no unit).
    eP : numpy.ndarray
        Array of internal variable 1 (no unit).
    xP : numpy.ndarray
        Array of internal variable 2 (Pa).
    young : numpy.ndarray
        Array of wires young modulus (Pa).
    sigmay : numpy.ndarray
        Array of yield stress (Pa).
    hardening : numpy.ndarray
        Array of hardening parameter (Pa).
    deltaT : numpy.ndarray, optional
        Array of delta temperature (°C). The default is 0.
    alpha : numpy.ndarray, optional
        Array of wires coefficient dilatation (1/°C). The default is 0.
    tol : float, optional
        Internal tolerance for the algorithm. The default is 1.0E-10.

    Returns
    -------
    sigma : numpy.ndarray
        Array of axial stress for each layer (Pa).
    eP : numpy.ndarray
        Array of update of internal variable 1 (no unit).
    xP : numpy.ndarray
        Array of update of internal variable 2 (Pa).
    """
    sig_t = young*(epsilon-eP-alpha*deltaT)

    # kinematic hardening for traction
    eta_t = sig_t-xP
    signe_t = np.sign(eta_t)
    phiP = np.abs(eta_t)-sigmay
    sgn_gP = phiP/(young+hardening) * signe_t
    t = phiP > tol
    xP = np.where(t, xP+sgn_gP*hardening, xP)
    eP = np.where(t, eP+sgn_gP, eP)
    sigma = np.where(t, sig_t-sgn_gP*young, sig_t)

    # no compression
    signe_t2 = np.sign(sigma)
    phiP = np.abs(sigma)
    sgn_gP = np.abs(sigma)/young * signe_t2
    t = signe_t2 < 0.
    xP = np.where(t, xP+sgn_gP*hardening, xP)
    eP = np.where(t, eP+sgn_gP, eP)
    sigma = np.where(t, sigma-sgn_gP*young, sigma)

    return sigma, eP, xP


def get_internal_forces(Aw: np.ndarray, nw: np.ndarray, layangles: np.ndarray,
                        epscable: float,
                        young: np.ndarray,
                        sigmay: np.ndarray, hardening: np.ndarray,
                        eP: np.ndarray = 0., xP: np.ndarray = 0.,
                        deltaT: np.ndarray = 0., alpha: np.ndarray = 0.,
                        tol: float = 1.e-10) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the internal behaviour of the cable for a given axial strain.

    Parameters
    ----------
    Aw : numpy.ndarray
        Array of wires section (m2).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    layangles : numpy.ndarray
        Array of lay angle (radians).
    epscable : float
        Axial strain of the cable (no unit).
    young : numpy.ndarray
        Array of wires young modulus (Pa).
    sigmay : numpy.ndarray
        Array of yield stress (Pa).
    hardening : numpy.ndarray
        Array of hardening parameter (Pa).
    eP : numpy.ndarray, optional
        Array of internal variable 1 (no unit). The default is 0.
    xP : numpy.ndarray, optional
        Array of internal variable 2 (Pa). The default is 0.
    deltaT : numpy.ndarray, optional
        Array of delta temperature (°C). The default is 0.
    alpha : numpy.ndarray, optional
        Array of wires thermal dilatation coefficient (1/°C). The default is 0.
    tol : float, optional
        Internal tolerance for the algorithm. The default is 1.0E-10.

    Returns
    -------
    float
        Total force (N).
    numpy.ndarray
        Array of layers force (N).
    numpy.ndarray
        Array of wires stress (Pa).

    """
    cang = np.cos(layangles)
    eps = epscable * cang**2
    sigma, _, _ = get_wires_axialstress(eps, eP, xP,
                                        young, sigmay, hardening,
                                        deltaT=deltaT, alpha=alpha,
                                        tol=tol)

    Flayers = sigma * cang * Aw * nw
    return float(np.sum(Flayers)), Flayers, sigma


def approxim_axial_behavior(Aw: np.ndarray, nw: np.ndarray, layangles: np.ndarray,
                            young: np.ndarray, sigmay: np.ndarray, hardening: np.ndarray,
                            epsround: int = 3, epsmax: float = 10.):
    """Evaluate an approximate internal behaviour of the cable knowing the layers behaviours.

    Parameters
    ----------
    Aw : numpy.ndarray
        Array of wires section (m2).
    nw : numpy.ndarray
        Array of integers defining the number of wires per layer (no unit).
    layangles : numpy.ndarray
        Array of lay angle (radians).
    young : numpy.ndarray
        Array of wires young modulus (Pa).
    sigmay : numpy.ndarray
        Array of yield stress (Pa).
    hardening : numpy.ndarray
        Array of hardening parameter (Pa).
    epsround : int, optional
        Precision for the approximation (no unit). The default is 3.
    epsmax : float, optional
        Maximal axial strain (no unit). The default is 10.

    Returns
    -------
    EA : numpy.ndarray
        Array of interval stiffness (N).
    FY : numpy.ndarray
        Array of interval yield force (N).
    EP : numpy.ndarray
        Array of deformations (no unit)

    """
    # internal strains inducing slopes change
    all_eps = np.zeros(len(young)+1)
    all_eps[:-1] = (sigmay / young) / (np.cos(layangles)**2)
    all_eps[-1] = epsmax
    all_eps = np.sort(np.unique(np.around(all_eps, epsround)))

    teps = np.zeros(len(young) + 1)
    teps[:-1] = (sigmay / young) / (np.cos(layangles)**2)
    teps[-1] = epsmax
    unq, idx = np.unique(np.around(teps, epsround), return_inverse=True)
    eps_c = np.sort([np.mean(teps[idx == i]) for i in np.unique(idx)])
    eps_c = eps_c[eps_c != 0.0]
    Fapprox = np.zeros(len(eps_c))
    for i, epscable in enumerate(eps_c):
        Fapprox[i], _, _ = get_internal_forces(Aw, nw, layangles, epscable, young, sigmay, hardening)
    ln = len(Fapprox)
    A = np.zeros((ln, ln))
    A[:, 0] = eps_c[0]
    for i in range(1, ln):
        A[i:, i] = eps_c[i] - eps_c[i - 1]
    EA = solve(A, Fapprox)
    FY = np.cumsum(EA[:-1] * A[-1, :-1])
    return EA, FY, eps_c[:-1]


def power_dissipated_foti(ymax: Union[float, np.ndarray], freq: Union[float, np.ndarray],
                          tension: Union[float, np.ndarray], ml: float, M0: float, kappa0: float,
                          formul: str) -> np.ndarray:
    """Evaluate the power dissipated by self-damping.

    Foti & Martinelli (2018) *An enhanced unified model for the self-damping
    of stranded cables under aeolian vibrations*.

    Parameters
    ----------
    ymax : float or numpy.ndarray
        Antinode vibration amplitude (m).
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    ml : float
        Lineic mass of the cable (kg/m).
    M0 : float
        Limit bending moment (N.m).
    kappa0 : float
        Limit curvature (1/m).
    formul : str
        Formulation used for the dissipation. The value must be in
        ['foti_full', 'foti_gross', 'foti_micro'].

    Returns
    -------
    numpy.array
        Array of power dissipated by damping.

    """
    def G1(kn):
        # eq (29)
        return 1./3.-3./8.*np.cos(2.*np.pi*kn)+1./24.*np.cos(6.*np.pi*kn)

    def G2(kn):
        # eq (30)
        return np.pi-4.*np.pi*kn+np.sin(4.*np.pi*kn)

    EImax_ef = M0/kappa0
    # eq (27)
    Pms = 128.*np.pi**5*ml**3*EImax_ef/kappa0*ymax**3*freq**7/tension**3
    # eq (28)
    Pgs = 4.*np.pi**3*ml**2*EImax_ef*ymax**2*freq**5/tension**2
    if formul == 'foti_full':
        # eq (23)
        lbd = 1./freq*np.sqrt(tension/ml)
        # eq (25)
        rsin = lbd**2*kappa0/(4*np.pi**2*ymax)
        if rsin > 1.:
            kn = 0.25
        elif rsin < 0.:
            kn = 0.
        else:
            kn = np.arcsin(rsin)/(2.*np.pi)
        return G1(kn)*Pms+G2(kn)*Pgs
    elif formul == 'foti_gross':
        return G2(0.)*Pgs
    elif formul == 'foti_micro':
        return G1(0.25)*Pms
    else:
        raise AssertionError("'%s' unknown" % formul)


def power_dissipated_cieren(ymax: Union[float, np.ndarray], freq: Union[float, np.ndarray],
                            tension: Union[float, np.ndarray], ml: float, EImin: float, EImax: float, kappa0: float,
                            formul: str, N: int = 1001) -> np.ndarray:
    """Evaluate the power dissipated by self-damping.

    Cieren (2020) *AAA*.

    Parameters
    ----------
    ymax : float or numpy.ndarray
        Antinode vibration amplitude (m).
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    ml : float
        Lineic mass of the cable (kg/m).
    EImin: float
        Minimal tangential bending stiffness of the cable (N.m2).
    EImax: float
        Maximal tangential bending stiffness of the cable (N.m2).
    kappa0 : float
        Limit curvature (1/m).
    formul : str
        Formulation used for the dissipation. The value must be in
        ['cieren_exact', 'cieren_approx'].
    N: int
        Number of discretized points used in the calculation
        of the dissipation.

    Returns
    -------
    numpy.array
        Array of power dissipated by damping.

    """
    def eds(x0, cm, cM, x):
        ep = cm / cM
        xb = (1. - ep) * x0
        y = x / xb
        return cM*xb**2*((8*(1+ep)+4*(1+2*ep)*y+4*ep*y**2)*np.exp(-y)-8*(1+ep)+4*y)

    lm = np.sqrt(tension/ml) / freq
    if formul == 'cieren_exact':
        x = np.linspace(0, 0.5, N) * lm
        c = 4. * np.pi**2 * ymax / lm**2 * np.sin(2. * np.pi * x / lm)
        e = eds(kappa0, EImin, EImax, c)
        return freq * 2. * np.sum(0.5 * (e[1:] + e[:-1]) * np.diff(x)) / lm
    elif formul == 'cieren_approx':
        # valid if ymax < (lm/(2*np.pi))**2 * kappa0/2
        return freq * 8*EImax/(9*np.pi*kappa0) * (4*np.pi**2*ymax/lm**2)**3
    else:
        raise AssertionError("'%s' unknown" % formul)
