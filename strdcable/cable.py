"""Cable and overheadline conductor object."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.optimize import minimize_scalar

import strdcable.geometrical_tools as gtls
import strdcable.overheadlines as fohl
import strdcable.physical_tools as mtls
from strdcable.build_tools import convertvars
from strdcable.build_tools import get_input_from_data
from strdcable.build_tools import testvar

# %%


class StrandedCable(object):
    """A stranded cable."""

    def __init__(self,
                 dwires: np.ndarray = None, nbwires: np.ndarray = None,
                 material: np.ndarray = None,
                 density: np.ndarray = None,
                 young: np.ndarray = None, poisson: np.ndarray = None,
                 alpha: np.ndarray = None,
                 epsilonpct: np.ndarray = None, sigmapct: np.ndarray = None,
                 sigmay: np.ndarray = None, hardening: np.ndarray = None,
                 epsilonu: np.ndarray = None, sigmau: np.ndarray = None,
                 laylengths: np.ndarray = None, wirelengths: np.ndarray = None,
                 initang: np.ndarray = None,
                 length: float = 1.,
                 compute_physics: bool = True) -> None:
        """Init with args.

        All input data must have the same shape and define each
        layer (except length)

        Parameters
        ----------
        dwires : numpy.ndarray
            Diameter (m)
        nbwires : numpy.ndarray
            Number of wires per layer (no unit).
        material : numpy.ndarray
            Material (no unit).
        density : numpy.ndarray, optional
            Density (no unit).
        young : numpy.ndarray, optional
            Young modulus (Pa).
        poisson : numpy.ndarray, optional
            Poisson ratio (no unit).
        alpha : numpy.ndarray, optional
            Thermal dilatation coefficient (1/°C).
        epsilonpct : numpy.ndarray, optional
            Limit elastic strain (no unit).
        sigmapct : numpy.ndarray, optional
            Limit elastic stress (Pa).
        sigmay : numpy.ndarray, optional
            Yield stress (Pa).
        hardening : numpy.ndarray, optional
            Hardening parameter (Pa).
        epsilonu : numpy.ndarray, optional
            Ultimate strain (no unit).
        sigmau : numpy.ndarray, optional
            Ultimate stress (Pa).
        laylengths : numpy.ndarray, optional
            Lay length (m).
        wirelengths : numpy.ndarray, optional
            Wire length (m).
        initang : numpy.ndarray, optional
            Initial angular position (radians).
        length : float, optional
            Cable length (m). The default is 1.0.
        compute_physics : bool, optional
            Whether or not compute the homogenized physical characteristics.
            Some optional inputs must be defined if True. The default is True.

        Returns
        -------
        None.

        """
        for v in [dwires, nbwires, material, density, young,
                  poisson, alpha, epsilonpct, sigmapct, sigmay,
                  epsilonu, sigmau, laylengths, wirelengths, initang]:
            testvar(v, np.ndarray)
        self.dwires = dwires
        self.nbwires = nbwires
        self.material = material
        self.density = density
        self.young = young
        self.poisson = poisson
        self.alpha = alpha
        self.epsilonpct = epsilonpct
        self.sigmapct = sigmapct
        self.sigmay = sigmay
        self.hardening = hardening
        self.epsilonu = epsilonu
        self.sigmau = sigmau
        self.laylengths = laylengths
        self.wirelengths = wirelengths
        self.initang = initang
        #
        self.length = length
        self.compute_all(compute_physics)
        return

    def compute_geometrical(self) -> None:
        """Compute characteristics relative to the geometry of the cable.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        lcond = ['AL1', 'AL4']
        self.nlayers = len(self.dwires)
        #
        if self.initang is None:
            self.initang = np.zeros(self.nlayers)
        # each layer
        # radial positions (m)
        self.positions = gtls.get_radial_positions(self.dwires)
        # area (m2)
        self.Awires = (self.dwires**2) * np.pi / 4.
        # inertia (m4)
        self.Iwires = (self.dwires**4) * np.pi / 64.
        laylengths = self.laylengths
        wirelengths = self.wirelengths
        if wirelengths is None and laylengths is None:
            laylengths = np.empty(self.nlayers)
            laylengths.fill(np.nan)
        elif wirelengths is None or np.all(np.isnan(wirelengths)):
            wirelengths = gtls.lay2length(laylengths, self.positions,
                                          self.length)
        else:
            laylengths = gtls.length2lay(wirelengths, self.positions)
            self.length = wirelengths[0]
        # lay length per layer (m)
        self.laylengths = laylengths
        # wire length per layer (m)
        self.wirelengths = wirelengths
        self.layangles = np.append([0.],
                                   gtls.get_lay_angles(self.laylengths[1:],
                                                       self.positions[1:]))
        # macro
        # external diameter(m)
        self.D = 2*self.positions[-1]+self.dwires[-1]
        # total area (m2)
        self.Amat, self.A, self.Acond = gtls.evaluate_sections(self.Awires,
                                                               self.nbwires,
                                                               self.material,
                                                               lcond=lcond)
        self.bimaterial = len(np.unique(self.material)) > 1
        if self.bimaterial:
            tmp = np.where(self.material[:-1] != self.material[1:])[0]
            if len(np.unique(self.material)) == 2:
                n = tmp[0]
            else:
                n = tmp[1]
            self.Dheart = 2*self.positions[n]+self.dwires[n]
            nlayercond = 0
            for k in lcond:
                nlayercond += len(np.where(self.material == k)[0])
            if nlayercond < 3:
                self.thermal_magn_effect = False
            else:
                self.thermal_magn_effect = True
        else:
            self.Dheart = 0.
            self.thermal_magn_effect = False
        # external rugosity (no unit)
        self.rugosity = gtls.evaluate_rugosity(self.dwires,
                                               self.positions, 'RTE')
        return

    def compute_physical_macro(self, formulEI: str = 'PAPAILIOU') -> None:
        """Compute characteristics relative to the macroscopic physics.

        Parameters
        ----------
        formulEI : str, optional
            Formulation used to compute the minimal and maximal bending
            stiffness. The value must be in ['FOTI', 'LEHANNEUR', 'COSTELLO',
            'EPRI', 'PAPAILIOU']. The default is 'PAPAILIOU'.

        Returns
        -------
        None.

        """
        # macro
        self.Vmat, self.V = gtls.evaluate_volumes(self.Awires, self.nbwires,
                                                  self.wirelengths,
                                                  self.material)
        self.m = mtls.get_lineic_mass(self.Awires, self.nbwires,
                                      self.wirelengths, self.density,
                                      self.length)
        self.EA = np.sum(mtls.get_layer_axial_stiffness(self.young,
                                                        self.Awires,
                                                        self.nbwires,
                                                        self.layangles))
        self.RTS = mtls.get_rated_strength(self.sigmau,
                                           self.Awires, self.nbwires)
        self.c_dilat = mtls.get_coeff_dilatation(self.alpha, self.young,
                                                 self.Awires, self.nbwires,
                                                 self.layangles)
        self.Bmin, self.Bcompl = mtls.get_EI_bounds_layers(self.young,
                                                           self.Awires,
                                                           self.nbwires,
                                                           self.layangles,
                                                           self.positions,
                                                           self.Iwires,
                                                           cpoiss=self.poisson,
                                                           formul=formulEI)
        self.EImin = np.sum(self.Bmin)
        self.EImax = np.sum(self.Bmin + self.Bcompl)
        return

    def compute_physical_layers(self, dmult: dict = mtls.__DCDEFAULT__) -> None:
        """Compute characteristics relative to the physics of each layer.

        Parameters
        ----------
        dmult : dict, optional
            Dictionary defining the ponderation applied to elastoplastic
            characteristics of each material. The default is defined in
            strdcable.physical_tools.__DCDEFAULT__.

        Returns
        -------
        None.

        """
        try:
            sy, K = mtls.get_elastocoeff(self.young,
                                         self.epsilonpct, self.sigmapct,
                                         self.epsilonu, self.sigmau,
                                         self.material,
                                         dmult)
        except TypeError:
            sy = None
            K = None
        sigmay = self.sigmay
        hardening = self.hardening
        if sigmay is None or np.all(np.isnan(sigmay)):
            self.sigmay = sy
        if hardening is None or np.all(np.isnan(hardening)):
            self.hardening = K
        return

    def compute_all(self, compute_physics: bool = True,
                    formulEI: str = 'PAPAILIOU') -> None:
        """Compute characteristics relative to the cable.

        Parameters
        ----------
        compute_physics : bool, optional
            Whether or not compute the homogenized physical characteristics.
            Some optional inputs must be defined if True. The default is True.
        formulEI : str, optional
            Formulation used to compute the minimal and maximal bending
            stiffness. The value must be in ['FOTI', 'LEHANNEUR', 'COSTELLO',
            'EPRI', 'PAPAILIOU']. The default is 'PAPAILIOU'.

        Returns
        -------
        None.

        """
        self.compute_geometrical()
        if compute_physics:
            self.compute_physical_macro(formulEI=formulEI)
            self.compute_physical_layers(dmult=mtls.__DCDEFAULT__)
        return

    def get_lay_bounds(self) -> np.ndarray:
        """Give normative bounds for lay lengths.

        Parameters
        ----------
        None.

        Returns
        -------
        numpy.ndarray
            (2, nlayers) array of min_max lay lengths for each layer (m). The
            default value is numpy.nan.

        """
        return np.full((2, self.nlayers), np.nan)

    def set_normative_lay_length(self,
                                 cmin: float = 0.5, cmax: float = 0.5) -> None:
        """Set the lay length for each layer according given ratios from specific rule.

        Parameters
        ----------
        cmin : float, optional
            Multiplicative coefficient for minimal rule (no unit). The default
            is 0.5.
        cmax : float, optional
            Multiplicative coefficient for maximal rule (no unit). The default
            is 0.5.

        Returns
        -------
        None.

        """
        self.wirelengths = None
        self.layangles = None
        self.laylengths = gtls.get_normalized_lay(self.get_lay_bounds(),
                                                  cmin=cmin, cmax=cmax)
        return

    def calculate_tension_deformation_curve(self, epsmax: float = 0.1, n_points: int = 20):
        """Calculate total axial tension values for a list of strain values

        Parameters
        ----------
        epsmax : float
            Maximum strain value for which the tension is computed (no unit).
        n_points: int
            Number of equally spaced-points for which the tension is computed.

        Returns
        -------
        values_force : ndarray
            Array of total tension in the cable for all calculated strain values (N).
        values_epsilon : ndarray
            Array of strain values for which the tension is calculated.

        """

        values_epsilon = np.linspace(0.0, epsmax, n_points)
        values_force = np.zeros((n_points, ))

        for i in range(n_points):
            values_force[i], _, _ = mtls.get_internal_forces(self.Awires, self.nbwires, self.layangles,
                                                             values_epsilon[i],
                                                             self.young, self.sigmay, self.hardening)

        return values_force, values_epsilon

    def approximate_axial_behaviour(self, epsround: int = 3,
                                    epsmax: float = 10.
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate an approximate internal axial behaviour of the cable.

        Parameters
        ----------
        epsround : int, optional
            Precision for the approximation (no unit). The default is 3.
        epsmax : float, optional
            Maximal axial strain (no unit). The default is 10.0.

        Returns
        -------
        numpy.ndarray
            Interval stiffness (N).
        numpy.ndarray
            Interval yield force (N).
        numpy.ndarray
            Interval deformation (no unit)

        """
        return mtls.approxim_axial_behavior(self.Awires, self.nbwires,
                                            self.layangles,
                                            self.young, self.sigmay,
                                            self.hardening,
                                            epsround=epsround, epsmax=epsmax)

    def approximate_bending_behaviour(self, tension: float, mu: np.ndarray,
                                      formul: str = 'FOTI'
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate an approximate internal bending behaviour of the cable.

        Parameters
        ----------
        tension : float
            Mechanical tension (N).
        mu : numpy.array
            (nlayers-1) array of friction coefficient for each interlayer
            (no unit).
        formul : str, optional
            Formulation used for the bending behaviour. The value must be in
            ['SC1', 'SC2', 'SC3', 'SC5', 'FOTI']. The default is 'FOTI'.

        Returns
        -------
        numpy.ndarray
            (nlayers-1) array of tangential bending stiffness (Pa.m4).
        numpy.ndarray
            (nlayers-1) array of limit curvature (1/m).

        """
        if len(mu) != self.nlayers-1:
            raise AssertionError("wrong friction coeff (%s)" % str(mu))
        epscable = tension/self.EA
        sy = 1.0E+18    # purely elastic behaviour of each wire
        hardg = np.nan  # purely elastic behaviour of each wire
        H, Flayers, _ = mtls.get_internal_forces(self.Awires, self.nbwires,
                                                 self.layangles, epscable,
                                                 self.young, sy, hardg)

        if np.abs(H-tension)/tension > 1.0E-01:
            raise AssertionError("wrong evaluation of epscable")

        _, _, Mrf = mtls.get_bending_limits(Flayers, self.young,
                                            self.layangles, self.positions,
                                            self.Awires, self.dwires,
                                            self.nbwires, mu, formul)
        return mtls.linearized_bending_law(self.Bmin, self.Bcompl, Mrf)

    def copy(self):
        """Return a copy of the instance (same as copy.deepcopy)."""
        return copy.deepcopy(self)

    def get_dict_vars(self) -> Tuple[dict, dict, dict]:
        """Return the main variables as dict depending on type.

        Parameters
        ----------
        None.

        Returns
        -------
        dict
            Three dictionaries with computed values.

        """
        dscal = dict(nb_layers=self.nlayers,
                     diameter=self.D,
                     section=self.A,
                     volume=self.V,
                     lineic_mass=self.m,
                     length=self.length,
                     axial_stiffness=self.EA,
                     minimal_bending_stiffness=self.EImin,
                     maximal_bending_stiffness=self.EImax,
                     rated_strength=self.RTS,
                     coefficient_dilatation=self.c_dilat,
                     bimaterial=self.bimaterial,
                     rugosity=self.rugosity)
        ddict = dict(materials_section=self.Amat,
                     materials_volume=self.Vmat)
        darray = dict(wire_diameter=self.dwires,
                      nb_wires=self.nbwires,
                      material=self.material,
                      initial_angle=self.initang,
                      density=self.density,
                      young_modulus=self.young,
                      poisson_ratio=self.poisson,
                      thermal_dilatation_coefficient=self.alpha,
                      yield_stress=self.sigmay,
                      hardening_parameter=self.hardening,
                      ultimate_strain=self.epsilonu,
                      ultimate_stress=self.sigmau,
                      lay_lengths=self.laylengths,
                      wire_lengths=self.wirelengths)
        return dscal, ddict, darray

    def export(self, format: str = 'dict') -> Union[dict, pd.DataFrame]:
        """Give a summary of the instance attributes.

        Parameters
        ----------
        format : str, optional
            Extracted format of output data. The value must be in
            ['dict', 'dataframe']. The default is 'dict'.

        Returns
        -------
        dict or pandas.DataFrame
            Output data. The type depends on the the given format.

        """
        dscal, ddict, darray = self.get_dict_vars()
        return convertvars(dscal, ddict, darray, format)

# %%


class Conductor(StrandedCable):
    """A overheadline conductor."""

    def __init__(self,
                 dwires=None, nbwires=None,
                 material=None,
                 density=None,
                 young=None, poisson=None,
                 alpha=None,
                 epsilonpct=None, sigmapct=None,
                 sigmay=None, hardening=None,
                 epsilonu=None, sigmau=None,
                 laylengths=None, wirelengths=None,
                 initang=None,
                 resistivity_20deg=None,
                 mass_resist_20deg=None,
                 compute_physics=True
                 ):
        """Init with args.

        Same as strdcable.cable.stranded_cable, plus additional parameters.

        Parameters
        ----------
        resistivity_20deg : numpy.ndarray, optional
            Defined in EN 60889.
        mass_resist_20deg : numpy.ndarray, optional
            Defined in EN 60889.

        Returns
        -------
        None.

        """
        for v in [resistivity_20deg, mass_resist_20deg]:
            testvar(v, np.ndarray)
        self.resistivity_20deg = resistivity_20deg  # max resist 20°C (Ohm.m)
        self.mass_resist_20deg = mass_resist_20deg  # Cresist mass 20°C (1/°C)
        super().__init__(dwires=dwires, nbwires=nbwires,
                         material=material,
                         density=density,
                         young=young, poisson=poisson,
                         alpha=alpha,
                         epsilonpct=epsilonpct, sigmapct=sigmapct,
                         sigmay=sigmay, hardening=hardening,
                         epsilonu=epsilonu, sigmau=sigmau,
                         laylengths=laylengths, wirelengths=wirelengths,
                         initang=initang,
                         compute_physics=compute_physics)
        return

    # replace existing parent functions
    def compute_all(self, compute_physics: bool = True,
                    set_usual_values: bool = True,
                    formulEI: str = 'PAPAILIOU', formulEP: str = 'rte') -> None:
        """Compute characteristics relative to the cable.

        Parameters
        ----------
        compute_physics : bool, optional
            Whether or not compute the homogenized physical characteristics.
            Some optional inputs must be defined if True. The default is True.
        set_usual_values : bool, optional
            Whether or not authorize to replace missing physical values. The
            default is True.
        formulEI : str, optional
            Formulation used to compute the minimal and maximal bending
            stiffness. The value must be in ['FOTI', 'LEHANNEUR', 'COSTELLO',
            'EPRI', 'PAPAILIOU']. The default is 'PAPAILIOU'.
        formulEP : str, optional
            Formulation used to compute the used for elastoplastic behaviour.
            The value must be in ['default', 'rte']. The default is 'rte'.

        Returns
        -------
        None.

        """
        self.compute_geometrical()
        if compute_physics:
            if set_usual_values:
                self.check_norm()
            self.compute_physical_macro(formulEI=formulEI)
            self.RDC20 = fohl.get_electric_resistance(self.Awires,
                                                      self.nbwires,
                                                      self.wirelengths,
                                                      self.resistivity_20deg,
                                                      self.length)
            self.heat_capacity = fohl.get_specific_heat_capacity(self.Awires,
                                                                 self.nbwires,
                                                                 self.wirelengths,
                                                                 self.density,
                                                                 self.material)
            if formulEP == 'rte':
                self.compute_physical_layers(dmult=fohl.__DCRTE__)
            elif formulEP == 'default':
                self.compute_physical_layers(dmult=mtls.__DCDEFAULT__)
            else:
                raise RuntimeError("wrong input for 'formulEP'")
        return

    def get_lay_bounds(self) -> np.ndarray:
        """Give normative bounds for lay lengths defined in EN 50182.

        Parameters
        ----------
        None.

        Returns
        -------
        numpy.ndarray
            (2, nlayers) array of min_max lay lengths for each layer (m). The
            default value is numpy.nan.

        """
        rule = fohl.laybounds_EN50182
        dl = (2*self.positions+self.dwires)
        return np.array([rule(i, self.nbwires[i], self.material) * dl[i]
                         for i in range(self.nlayers)])

    def get_dict_vars(self) -> Tuple[dict, dict, dict]:
        """Return the main variables as dict depending on type.

        Parameters
        ----------
        None.

        Returns
        -------
        dict
            Three dictionaries with computed values.

        """
        dscal = dict(nb_layers=self.nlayers,
                     diameter=self.D,
                     section=self.A,
                     volume=self.V,
                     lineic_mass=self.m,
                     length=self.length,
                     axial_stiffness=self.EA,
                     minimal_bending_stiffness=self.EImin,
                     maximal_bending_stiffness=self.EImax,
                     rated_strength=self.RTS,
                     coefficient_dilatation=self.c_dilat,
                     bimaterial=self.bimaterial,
                     rugosity=self.rugosity,
                     diameter_steel_core=self.Dheart,
                     olla_thermal_magnetic_effects=self.thermal_magn_effect,
                     conductive_section=self.Acond,
                     electrical_resistance=self.RDC20)
        ddict = dict(materials_section=self.Amat,
                     materials_volume=self.Vmat)
        darray = dict(wire_diameter=self.dwires,
                      nb_wires=self.nbwires,
                      material=self.material,
                      initial_angle=self.initang,
                      density=self.density,
                      young_modulus=self.young,
                      poisson_ratio=self.poisson,
                      thermal_dilatation_coefficient=self.alpha,
                      yield_stress=self.sigmay,
                      hardening_parameter=self.hardening,
                      ultimate_strain=self.epsilonu,
                      ultimate_stress=self.sigmau,
                      lay_lengths=self.laylengths,
                      wire_lengths=self.wirelengths)
        return dscal, ddict, darray

    def export(self, format: str = 'dict'):
        """Give a summary of the instance attributes.

        Parameters
        ----------
        format : str, optional
            Extracted format of output data. The value must be in
            ['dict', 'dataframe']. The default is 'dict'.

        Returns
        -------
        dict or pandas.DataFrame
            Output data. The type depends on the the given format.

        """
        dscal, ddict, darray = self.get_dict_vars()
        dscaladd = dict(resistivity_20deg=self.resistivity_20deg,
                        mass_resist_20deg=self.mass_resist_20deg)
        return convertvars({**dscal, **dscaladd},
                           ddict, darray, format)

    # additional functions
    def check_norm(self) -> None:
        """Apply missing physical characteristics from given rules.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        df_add = fohl.get_charac_rules(self.nlayers, self.dwires,
                                       self.material)
        for k in df_add.columns:
            if hasattr(self, k):
                vold = getattr(self, k)
                if vold is None or np.all(np.isnan(vold)):
                    setattr(self, k, df_add[k].values)
            else:
                raise AttributeError("unknown characteristic '%s'" % k)
        return

    def get_lifespan(self, sigma_a: np.ndarray,
                     method: str = 'safe_border_line',
                     infinite_life: float = 1.e18) -> np.ndarray:
        """Evaluate the lifespan of the conductor for given stress amplitude.

        Parameters
        ----------
        sigma_a : numpy.ndarray
            Array of stress amplitude (Pa).
        method : str, optional
            Option to compute the lifespan. The value must be in
            ['safe_border_line']. The default is 'safe_border_line'.

        Returns
        -------
        numpy.ndarray
            Array of supported number of cycles (no unit).
        """
        if method == 'safe_border_line':
            return fohl.inverse_SafeBorderLine(self.material, sigma_a,
                                               infinite_life=1.e18)
        else:
            raise RuntimeError("unknown method '%s'" % method)

    def get_fatigue_stress(self, N: np.ndarray,
                           method: str ='safe_border_line') ->np.ndarray:
        """Evaluate the stress amplitude of a given number of cyclic vibration.

        Parameters
        ----------
        N : numpy.ndarray
            number of cycles (no unit)
        method : str, optional
            Option to compute the lifespan. The value must be in
            ['safe_border_line']. The default is 'safe_border_line'.

        Returns
        -------
        numpy.ndarray
            Array of stress amplitude (Pa).
        """
        if method == 'safe_border_line':
            return fohl.get_SafeBorderLine(self.material, N)
        else:
            raise NotImplementedError("unknown method '%s'" % method)

    def get_ratio_fstress_fymax(self, EI: float) -> float:
        """Give the fictive stress ratio which must be multiplied by fymax.

        (Orange Book -- chap 'fatigue of overhead conductors'
        "eq 3.2-14" 3-12 -- page 196/614).

        Parameters
        ----------
        EI : float
            Bending stiffness (Pa.m4).

        Returns
        -------
        float
            Fictive stress ratio (Pa.s/m).
        """
        return np.pi*self.dwires[-1]*self.young[-1]*np.sqrt(self.m/EI)

    def get_ratio_fstress_Y(self, tension: float, x: float, EI: float) -> float:
        """Give the fictive stress ratio which must be multiplied by Y.

        (Orange Book -- chap 'fatigue of overhead conductors'
        "eq 3.2-15" 3-12 -- page 196/614).

        Parameters
        ----------
        tension : float
            Mechanical tension (N).
        x : float
            Abscissa relative where Y is given (m).
        EI : float
            Bending stiffness (Pa.m4).

        Returns
        -------
        float
            Fictive stress ratio (Pa/m)
        """
        p = np.sqrt(tension/EI)
        denom = 4.*(np.exp(-p*x)-1.+p*x)
        return (self.dwires[-1]*self.young[-1]*p**2) / denom

# %%


def get_cable(dtype: str, df: pd.DataFrame, test: Union[np.ndarray, pd.Series, bool],
              headers: list, drename: dict,
              sortcols: list) -> Tuple[Union[StrandedCable, Conductor], pd.DataFrame]:
    """Give the instance corresponding to the chosen type and input data.

    Parameters
    ----------
    dtype : str
        Chosen cable type. The value must be in ['strandedcable', 'conductor'].
    df : pandas.DataFrame
        Dataframe containing basic geometrical data on the given cable
        ['material', 'dwires', 'nbwires'].
    test : numpy.ndarray
        Array of imposed test to get relevant index (same size as df).
    sortcols : list, optional
        List of strings defining columns used to sort lines relative to one
        conductor.
    drename : dict, optional
        Dictionary used to rename columns from strdcable.cable.strandedcable
    headers : list, optional
        data needed to initialize strdcable.cable.strandedcable

    Raises
    ------
    AssertionError
        Wrong input type.

    Returns
    -------
    strdcable.cable.StrandedCable or strdcable.cable.Conductor
        Output instance (type depending on dtype)
    dtest : pandas.DataFrame
        A dataframe corresponding to an extraction of the initial dataframe
        according to test

    """
    dtest, dinput = get_input_from_data(df, test,
                                        headers, drename, sortcols)
    if dtype == 'strandedcable':
        cl = StrandedCable
    elif dtype == 'conductor':
        cl = Conductor
    else:
        raise AssertionError("unknown type '%s'" % dtype)
    return cl(**dinput, compute_physics=False), dtest

# %%


class SimplifiedBending:
    """An analytical bending behaviour."""

    def __init__(self, dfin: pd.DataFrame,
                 cable: StrandedCable = None,
                 sortcols: list = None, drename: dict = None,
                 headers: list = ['material', 'dwires', 'nbwires'],
                 dtype: str = 'conductor') -> None:
        """Init with args.

        Same as strdcable.cable.stranded_cable, plus additional parameters.

        Parameters
        ----------
        dfin : pandas.DataFrame
            Dataframe containing basic geometrical data on conductors
            ['material', 'dwires', 'nbwires'].
        cable : StrandedCable, optional
            Potential input strdcable object. The default is None.
        sortcols : list, optional
            List of strings defining columns used to sort lines relative to one
            conductor.
        drename : dict, optional
            Dictionary used to rename columns from strdcable.cable.strandedcable.
        headers : list, optional
            List of strings defining columns needed to initialize
            strdcable.cable.strandedcable.
        dtype : str, optional
            Chosen cable type. The value must be in ['strandedcable', 'conductor'].

        Returns
        -------
        None.

        """
        if cable is None:
            self.cable, _ = get_cable(dtype, dfin, None,
                                      headers, drename, sortcols)
        else:
            self.cable = cable
        # Array of friction coeff for each interlayer (no unit).
        self.mu = None
        # Array of successive bending stiffness (Pa.m4).
        self.EI = None
        # Array of limit curvatures (1/m).
        self.kappa = None
        # Linearized limit moment and curvature.
        self.M0 = None
        self.kappa0 = None
        return

    def set_laylengths(self, rmn: Union[float, np.ndarray]) -> None:
        """Change the lay length of each layer according to the normative bounds.

        Parameters
        ----------
        rmn : float or numpy.ndarray
            Multiplicative ratio relative to the minimal bound. The value
            must be in range [0., 1.]. If not float, (nlayers) array.

        Returns
        -------
        None.

        """
        if np.any(rmn > 1.) or np.any(rmn < 0.):
            raise AssertionError('ratio should be in [0., 1.]')
        if isinstance(rmn, np.ndarray) and len(rmn) != self.cable.nlayers:
            raise AssertionError('all layers should have defined ratio')
        self.cable.set_normative_lay_length(rmn, 1.-rmn)
        return

    def set_friction(self, mu: Union[float, np.ndarray]) -> None:
        """Set the friction coefficient in the cable.

        Parameters
        ----------
        mu : float or numpy.array
            Friction coefficient for each interlayer (no unit). If not float,
            (nlayers-1) array.

        Returns
        -------
        None.

        """
        if np.any(mu <= 0.):
            raise AssertionError('friction should be strictly positive')
        if isinstance(mu, np.ndarray) and len(mu) != self.cable.nlayers-1:
            raise AssertionError('interlayers should have defined friction')
        elif isinstance(mu, np.ndarray):
            self.mu = mu
        elif type(mu) in [int, float, np.float32, np.float64]:
            self.mu = np.ones(self.cable.nlayers-1) * mu
        else:
            raise AssertionError("unknown type '%s'" % type(mu))
        return

    def evaluate(self, tension: float, formul: str = 'FOTI'):
        """Evaluate an approximate internal bending behaviour of the conductor.

        Parameters
        ----------
        tension : float
            Mechanical tension (N).
        formul : str, optional
            Formulation used for the bending behaviour. The value must be in
            ['SC1', 'SC2', 'SC3', 'SC5', 'FOTI']. The default is 'FOTI'.

        Returns
        -------
        None.

        """
        self.cable.compute_all()
        feval = self.cable.approximate_bending_behaviour
        self.EI, self.kappa = feval(tension, self.mu, formul=formul)
        self.M0, self.kappa0 = mtls.linearized_macro_bending(self.EI,
                                                             self.kappa)
        kp = np.concatenate(([0.], self.kappa))
        ei = np.array(self.EI)
        mi = np.concatenate(([0.], np.cumsum(ei[:-1] * np.diff(kp))))
        self.x0 = (mi[-1] - ei[-1] * kp[-1]) / (ei[0] - ei[-1])
        return

    def get_limit_curvature(self, nfit: int) ->np.ndarray:
        """
        .

        Parameters
        ----------
        nfit : int
            Integer related to the bending behaviour approximation.

        Returns
        -------
        np.ndarray
            DESCRIPTION.

        """
        tensions = np.linspace(0.05, 0.35, nfit) * self.cable.RTS
        x0 = np.zeros_like(tensions)
        for j, h in enumerate(tensions):
            self.evaluate(h)
            x0[j] = self.x0
        return np.polyfit(tensions, x0 / tensions, 0)

    def get_self_damping(self, ymax: float, freq: float, tension:float, formul: str) -> np.ndarray:
        """Evaluate the power dissipated by self-damping.

        Parameters
        ----------
        ymax : float or numpy.ndarray
            Antinode vibration amplitude (m).
        freq : float or numpy.ndarray
            Frequencies relative to vibration (Hz).
        tension : float or numpy.ndarray
            Mechanical tension (N).
        formul : str
            Formulation used for the dissipation. The value must be in
            ['foti_full', 'foti_gross', 'foti_micro', 'cieren_exact', 'cieren_approx'].

        Returns
        -------
        numpy.array
            Array of power dissipated by damping.

        """
        if formul in ['foti_full', 'foti_gross', 'foti_micro']:
            return mtls.power_dissipated_foti(ymax, freq, tension,
                                              self.cable.m,
                                              self.M0, self.kappa0, formul)
        elif formul in ['cieren_exact', 'cieren_approx']:
            return mtls.power_dissipated_cieren(ymax, freq, tension,
                                                self.cable.m,
                                                self.cable.EImin, self.cable.EImax,
                                                self.kappa0, formul)
        else:
            raise

    def apply_EBP(self, freq: Union[float, np.ndarray], tension: Union[float, np.ndarray], Iv: float,
                  wind_power: str = 'cigre',
                  dissipation: str = 'foti_full') -> np.ndarray:
        """Evaluate the potential maximal ibration amplitude usin the Energy Balance Principle.

        Parameters
        ----------
        freq : float or numpy.ndarray
            Frequencies relative to vibration (Hz).
        tension : float or numpy.ndarray
            Mechanical tension (N).
        Iv : float
            Turbulence intensity of the wind (no unit).
        wind_power : str, optional
            Formulation used for wind input power. The value must be in ['cigre'].
            The default is 'cigre'.
        dissipation : str, optional
            Formulation used for used for the dissipation. The value must be in
            ['foti_full', 'foti_gross', 'foti_micro', 'cieren_exact', 'cieren_approx'].
            The default is 'foti_full'.

        Returns
        -------
        numpy.array
            Array of antinode vibration amplitude (m).

        """
        def solve(f: float) -> scipy.optimize.OptimizeResult:
            """Solve the Energy Balance Principle by minimizing the absolute difference.

            Parameters
            ----------
            f : float
                Frequencies relative to vibration (Hz).

            Returns
            -------
            scipy.optimize.minimize_scalar
                Optimization result.

            """

            def mybalance(x: np.ndarray) -> np.ndarray:
                """Set my dummy function.

                Parameters
                ----------
                x : numpy.ndarray
                    Antinode vibration amplitude (m).

                Returns
                -------
                numpy.array
                    Array of absolute diff between wind power and self-damping dissipation.

                """
                if wind_power == 'cigre':
                    Pw = fohl.get_wind_power_cigre(x, f, self.cable.D, Iv)
                else:
                    raise AssertionError("wrong input wind power %s" % wind_power)
                return np.abs(Pw-self.get_self_damping(x, f, tension, dissipation))
            dargs = dict(bounds=(0., 100.*self.cable.D),
                         method='bounded')
            return minimize_scalar(mybalance, **dargs)

        return np.array([solve(fr).x for fr in freq])
