# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import pytest
import strdcable.cable as cable
import numpy as np
import copy


def test_StrandedCable_validinstantiation():
    """
    test for StrandedCable class in cable.py
    - tests class instantiation with non-default inputs

    """
    try:
        _ = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                            nbwires=np.array([1, 2]),
                                            material=np.array(['ST6C', 'AL1']),
                                            density=np.array([7.8, 2.7]),
                                            young=np.array([1e9, 1e8]),
                                            poisson=np.array([0.3, 0.33]),
                                            alpha=np.array([1e-5, 1e-5]),
                                            epsilonpct=np.array([1.0, 1.0]),
                                            sigmapct=np.array([1.0, 1.0]),
                                            sigmay=np.array([1e6, 1e6]),
                                            hardening=np.array([1e8, 1e8]),
                                            epsilonu=np.array([2.0, 2.0]),
                                            sigmau=np.array([1e10, 1e10]),
                                            laylengths=np.array([1.5, 2.0]),
                                            wirelengths=np.array([1.5, 2.0]),
                                            initang=np.array([np.nan, 15.0]),
                                            length=1.0,
                                            compute_physics=True)
        assert True
    except Exception:
        assert False


def test_StrandedCable_compute_geometrical():
    """
    test for compute_geometrical method in StrandedCable class of cable.py
    - tests if geometric characteristics are computed

    """
    try:
        strandedcable = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                            nbwires=np.array([1, 2]),
                                            material=np.array(['ST6C', 'AL1']),
                                            density=np.array([7.8, 2.7]),
                                            young=np.array([1e9, 1e8]),
                                            poisson=np.array([0.3, 0.33]),
                                            alpha=np.array([1e-5, 1e-5]),
                                            epsilonpct=np.array([1.0, 1.0]),
                                            sigmapct=np.array([1.0, 1.0]),
                                            sigmay=np.array([1e6, 1e6]),
                                            hardening=np.array([1e8, 1e8]),
                                            epsilonu=np.array([2.0, 2.0]),
                                            sigmau=np.array([1e10, 1e10]),
                                            laylengths=np.array([1.5, 2.0]),
                                            wirelengths=np.array([1.5, 2.0]),
                                            initang=np.array([np.nan, 15.0]),
                                            length=1.0,
                                            compute_physics=True)

        assert strandedcable.nlayers == 2
        assert strandedcable.length == 1.5
        assert strandedcable.layangles[0] == 0.0
        assert strandedcable.D == 3.0
        assert strandedcable.bimaterial is True
    except Exception:
        assert False


def test_StrandedCable_compute_physical_macro():
    """
    test for compute_physical_macro method in StrandedCable class of cable.py
    - tests if macroscopic quantities are computed for all formulations

    """
    strandedcable = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                        nbwires=np.array([1, 2]),
                                        material=np.array(['ST6C', 'AL1']),
                                        density=np.array([7.8, 2.7]),
                                        young=np.array([1e9, 1e8]),
                                        poisson=np.array([0.3, 0.33]),
                                        alpha=np.array([1e-5, 1e-5]),
                                        epsilonpct=np.array([1.0, 1.0]),
                                        sigmapct=np.array([1.0, 1.0]),
                                        sigmay=np.array([1e6, 1e6]),
                                        hardening=np.array([1e8, 1e8]),
                                        epsilonu=np.array([2.0, 2.0]),
                                        sigmau=np.array([1e10, 1e10]),
                                        laylengths=np.array([1.5, 2.0]),
                                        wirelengths=np.array([1.5, 2.0]),
                                        initang=np.array([np.nan, 15.0]),
                                        length=1.0,
                                        compute_physics=True)

    formulEI_list = ['FOTI', 'LEHANNEUR', 'COSTELLO', 'EPRI', 'PAPAILIOU']

    for formul in formulEI_list:
        try:
            strandedcable.compute_physical_macro(formulEI=formul)
            assert strandedcable.Vmat is not None
            assert strandedcable.V is not None
            assert strandedcable.m is not None
            assert strandedcable.EA is not None
            assert strandedcable.RTS is not None
            assert strandedcable.c_dilat is not None
            assert strandedcable.Bmin is not None
            assert strandedcable.Bcompl is not None
            assert strandedcable.EImin is not None
            assert strandedcable.EImax is not None
        except Exception:
            assert False


def test_StrandedCable_compute_physical_layers():
    """
    test for compute_physical_layers method in StrandedCable class of cable.py
    - tests if layer quantities are computed with and without specifying yield strength and hardening

    """
    cable1 = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                 nbwires=np.array([1, 2]),
                                 material=np.array(['ST6C', 'AL1']),
                                 density=np.array([7.8, 2.7]),
                                 young=np.array([1e9, 1e8]),
                                 poisson=np.array([0.3, 0.33]),
                                 alpha=np.array([1e-5, 1e-5]),
                                 epsilonpct=np.array([1.0, 1.0]),
                                 sigmapct=np.array([1.0, 1.0]),
                                 sigmay=np.array([1e6, 1e6]),
                                 hardening=np.array([1e8, 1e8]),
                                 epsilonu=np.array([2.0, 2.0]),
                                 sigmau=np.array([1e10, 1e10]),
                                 laylengths=np.array([1.5, 2.0]),
                                 wirelengths=np.array([1.5, 2.0]),
                                 initang=np.array([np.nan, 15.0]),
                                 length=1.0,
                                 compute_physics=True)

    cable2 = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                 nbwires=np.array([1, 2]),
                                 material=np.array(['ST6C', 'AL1']),
                                 density=np.array([7.8, 2.7]),
                                 young=np.array([1e9, 1e8]),
                                 poisson=np.array([0.3, 0.33]),
                                 alpha=np.array([1e-5, 1e-5]),
                                 epsilonpct=np.array([1.0, 1.0]),
                                 sigmapct=np.array([1.0, 1.0]),
                                 epsilonu=np.array([2.0, 2.0]),
                                 sigmau=np.array([1e10, 1e10]),
                                 laylengths=np.array([1.5, 2.0]),
                                 wirelengths=np.array([1.5, 2.0]),
                                 initang=np.array([np.nan, 15.0]),
                                 length=1.0,
                                 compute_physics=True)

    for val in cable1.sigmay:
        assert val == 1e6
    for val in cable1.hardening:
        assert val == 1e8
    assert cable2.sigmay is not None
    assert cable2.hardening is not None


def test_StrandedCable_compute_all():
    """
    test for compute_all method in StrandedCable class of cable.py
    - tests if geometrical quantities are always computed and if macroscopic and layer physical quantities
      are computed when instructed to

    """
    strandedcable = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                        nbwires=np.array([1, 2]),
                                        material=np.array(['ST6C', 'AL1']),
                                        density=np.array([7.8, 2.7]),
                                        young=np.array([1e9, 1e8]),
                                        poisson=np.array([0.3, 0.33]),
                                        alpha=np.array([1e-5, 1e-5]),
                                        epsilonpct=np.array([1.0, 1.0]),
                                        sigmapct=np.array([1.0, 1.0]),
                                        sigmay=np.array([1e6, 1e6]),
                                        hardening=np.array([1e8, 1e8]),
                                        epsilonu=np.array([2.0, 2.0]),
                                        sigmau=np.array([1e10, 1e10]),
                                        laylengths=np.array([1.5, 2.0]),
                                        wirelengths=np.array([1.5, 2.0]),
                                        initang=np.array([np.nan, 15.0]),
                                        length=1.0,
                                        compute_physics=False)

    try:
        strandedcable.compute_all(compute_physics=False)
        assert strandedcable.nlayers == 2
        assert strandedcable.Awires[0] == np.pi / 4
        assert strandedcable.Awires[1] == np.pi / 4
        assert strandedcable.Iwires[0] == np.pi / 64
        assert strandedcable.Iwires[1] == np.pi / 64
        assert strandedcable.D == 3.0
        assert strandedcable.bimaterial is True
    except Exception:
        assert False

    formulEI_list = ['FOTI', 'LEHANNEUR', 'COSTELLO', 'EPRI', 'PAPAILIOU']
    for formul in formulEI_list:
        try:
            strandedcable.compute_all(compute_physics=True, formulEI=formul)
            test_StrandedCable_compute_physical_macro()
            test_StrandedCable_compute_physical_layers()
        except Exception:
            assert False


def test_StrandedCable_get_lay_bounds():
    """
    test for get_lay_bounds method in StrandedCable class of cable.py
    - tests if normative bounds for lay lengths

    """
    strandedcable = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                        nbwires=np.array([1, 2]),
                                        material=np.array(['ST6C', 'AL1']),
                                        density=np.array([7.8, 2.7]),
                                        young=np.array([1e9, 1e8]),
                                        poisson=np.array([0.3, 0.33]),
                                        alpha=np.array([1e-5, 1e-5]),
                                        epsilonpct=np.array([1.0, 1.0]),
                                        sigmapct=np.array([1.0, 1.0]),
                                        sigmay=np.array([1e6, 1e6]),
                                        hardening=np.array([1e8, 1e8]),
                                        epsilonu=np.array([2.0, 2.0]),
                                        sigmau=np.array([1e10, 1e10]),
                                        laylengths=np.array([1.5, 2.0]),
                                        wirelengths=np.array([1.5, 2.0]),
                                        initang=np.array([np.nan, 15.0]),
                                        length=1.0,
                                        compute_physics=False)

    try:
        lay_bounds = strandedcable.get_lay_bounds()
        assert np.isnan(lay_bounds).all()
    except Exception:
        assert False


def test_StrandedCable_set_normative_lay_length():
    """
    test for set_normative_lay_length method in StrandedCable class of cable.py
    - tests if lay lengths are computed with and without multiplicative coefficients

    """
    strandedcable = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                        nbwires=np.array([1, 2]),
                                        material=np.array(['ST6C', 'AL1']),
                                        density=np.array([7.8, 2.7]),
                                        young=np.array([1e9, 1e8]),
                                        poisson=np.array([0.3, 0.33]),
                                        alpha=np.array([1e-5, 1e-5]),
                                        epsilonpct=np.array([1.0, 1.0]),
                                        sigmapct=np.array([1.0, 1.0]),
                                        sigmay=np.array([1e6, 1e6]),
                                        hardening=np.array([1e8, 1e8]),
                                        epsilonu=np.array([2.0, 2.0]),
                                        sigmau=np.array([1e10, 1e10]),
                                        laylengths=np.array([1.5, 2.0]),
                                        wirelengths=np.array([1.5, 2.0]),
                                        initang=np.array([np.nan, 15.0]),
                                        length=1.0,
                                        compute_physics=False)

    try:
        strandedcable.set_normative_lay_length()
        assert strandedcable.wirelengths is None
        assert strandedcable.layangles is None
        laylenghts = copy.deepcopy(strandedcable.laylengths)
        strandedcable.set_normative_lay_length(cmin=0.5, cmax=0.5)
        assert np.not_equal(strandedcable.laylengths, laylenghts).all()
    except Exception:
        assert False


def test_StrandedCable_approximate_axial_behaviour():
    """
    test for approximate_axial_behaviour method in StrandedCable class of cable.py
    - tests if axial behaviour of the cable is computed

    """
    strandedcable = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                        nbwires=np.array([1, 2]),
                                        material=np.array(['ST6C', 'AL1']),
                                        density=np.array([7.8, 2.7]),
                                        young=np.array([1e9, 1e8]),
                                        poisson=np.array([0.3, 0.33]),
                                        alpha=np.array([1e-5, 1e-5]),
                                        epsilonpct=np.array([1.0, 1.0]),
                                        sigmapct=np.array([1.0, 1.0]),
                                        sigmay=np.array([1e6, 1e6]),
                                        hardening=np.array([1e8, 1e8]),
                                        epsilonu=np.array([2.0, 2.0]),
                                        sigmau=np.array([1e10, 1e10]),
                                        laylengths=np.array([1.5, 2.0]),
                                        wirelengths=np.array([1.5, 2.0]),
                                        initang=np.array([np.nan, 15.0]),
                                        length=1.0,
                                        compute_physics=True)

    try:
        _, _, _ = strandedcable.approximate_axial_behaviour(epsround=4)
        assert True
    except Exception:
        assert False


def test_StrandedCable_approximate_bending_behaviour():
    """
    test for approximate_bending_behaviour method in StrandedCable class of cable.py
    - tests if bending behaviour of the cable is computed for all formulations

    """
    strandedcable = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                        nbwires=np.array([1, 2]),
                                        material=np.array(['ST6C', 'AL1']),
                                        density=np.array([7.8, 2.7]),
                                        young=np.array([1e9, 1e8]),
                                        poisson=np.array([0.3, 0.33]),
                                        alpha=np.array([1e-5, 1e-5]),
                                        epsilonpct=np.array([1.0, 1.0]),
                                        sigmapct=np.array([1.0, 1.0]),
                                        sigmay=np.array([1e6, 1e6]),
                                        hardening=np.array([1e8, 1e8]),
                                        epsilonu=np.array([2.0, 2.0]),
                                        sigmau=np.array([1e10, 1e10]),
                                        laylengths=np.array([1.5, 2.0]),
                                        wirelengths=np.array([1.5, 2.0]),
                                        initang=np.array([np.nan, 15.0]),
                                        length=1.0,
                                        compute_physics=True)

    formul_list = ['SC1', 'SC2', 'SC3', 'SC5', 'FOTI']

    tension = 1e4
    mu = np.array([1.0])

    for formul in formul_list:
        try:
            strandedcable.approximate_bending_behaviour(tension, mu, formul)
            assert True
        except Exception:
            assert False


def test_StrandedCable_copy():
    """
    test for copy method in StrandedCable class of cable.py
    - tests if object is copied

    """
    strandedcable = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                        nbwires=np.array([1, 2]),
                                        material=np.array(['ST6C', 'AL1']),
                                        density=np.array([7.8, 2.7]),
                                        young=np.array([1e9, 1e8]),
                                        poisson=np.array([0.3, 0.33]),
                                        alpha=np.array([1e-5, 1e-5]),
                                        epsilonpct=np.array([1.0, 1.0]),
                                        sigmapct=np.array([1.0, 1.0]),
                                        sigmay=np.array([1e6, 1e6]),
                                        hardening=np.array([1e8, 1e8]),
                                        epsilonu=np.array([2.0, 2.0]),
                                        sigmau=np.array([1e10, 1e10]),
                                        laylengths=np.array([1.5, 2.0]),
                                        wirelengths=np.array([1.5, 2.0]),
                                        initang=np.array([np.nan, 15.0]),
                                        length=1.0,
                                        compute_physics=True)

    try:
        copied_cable = strandedcable.copy()
        assert type(copied_cable) is cable.StrandedCable
    except Exception:
        assert False


def test_StrandedCable_get_dict_vars():
    """
    test for get_dict_vars in StrandedCable class of cable.py
    - tests if output dictionaries have proper values (scalar, dict and array)

    """
    strandedcable = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                        nbwires=np.array([1, 2]),
                                        material=np.array(['ST6C', 'AL1']),
                                        density=np.array([7.8, 2.7]),
                                        young=np.array([1e9, 1e8]),
                                        poisson=np.array([0.3, 0.33]),
                                        alpha=np.array([1e-5, 1e-5]),
                                        epsilonpct=np.array([1.0, 1.0]),
                                        sigmapct=np.array([1.0, 1.0]),
                                        sigmay=np.array([1e6, 1e6]),
                                        hardening=np.array([1e8, 1e8]),
                                        epsilonu=np.array([2.0, 2.0]),
                                        sigmau=np.array([1e10, 1e10]),
                                        laylengths=np.array([1.5, 2.0]),
                                        wirelengths=np.array([1.5, 2.0]),
                                        initang=np.array([np.nan, 15.0]),
                                        length=1.0,
                                        compute_physics=True)

    try:
        dscal, ddict, darray = strandedcable.get_dict_vars()

        for v in dscal.values():
            if isinstance(v, list):
                for elem in v:
                    if isinstance(elem, dict) or isinstance(elem, np.ndarray):
                        assert False
                    else:
                        assert True
            else:
                if isinstance(v, dict) or isinstance(v, np.ndarray):
                    assert False
                else:
                    assert True

        for v in ddict.values():
            if isinstance(v, list):
                for elem in v:
                    if isinstance(elem, float) or isinstance(elem, int) or isinstance(elem, bool) or isinstance(elem, np.ndarray):
                        assert False
                    else:
                        assert True
            else:
                if isinstance(v, float) or isinstance(v, int) or isinstance(v, bool) or isinstance(v, np.ndarray):
                    assert False
                else:
                    assert True

        for v in darray.values():
            if isinstance(v, list):
                for elem in v:
                    if isinstance(elem, float) or isinstance(elem, int) or isinstance(elem, bool) or isinstance(elem, dict):
                        assert False
                    else:
                        assert True
            else:
                if isinstance(v, float) or isinstance(v, int) or isinstance(v, bool) or isinstance(v, dict):
                    assert False
                else:
                    assert True

    except Exception:
        assert False


def test_StrandedCable_export():
    """
    test for export in StrandedCable class of cable.py
    - tests if export works for available formats

    """
    strandedcable = cable.StrandedCable(dwires=np.array([1.0, 1.0]),
                                        nbwires=np.array([1, 2]),
                                        material=np.array(['ST6C', 'AL1']),
                                        density=np.array([7.8, 2.7]),
                                        young=np.array([1e9, 1e8]),
                                        poisson=np.array([0.3, 0.33]),
                                        alpha=np.array([1e-5, 1e-5]),
                                        epsilonpct=np.array([1.0, 1.0]),
                                        sigmapct=np.array([1.0, 1.0]),
                                        sigmay=np.array([1e6, 1e6]),
                                        hardening=np.array([1e8, 1e8]),
                                        epsilonu=np.array([2.0, 2.0]),
                                        sigmau=np.array([1e10, 1e10]),
                                        laylengths=np.array([1.5, 2.0]),
                                        wirelengths=np.array([1.5, 2.0]),
                                        initang=np.array([np.nan, 15.0]),
                                        length=1.0,
                                        compute_physics=True)

    try:
        out = strandedcable.export('dict')
        assert isinstance(out, dict)
        out = strandedcable.export('dataframe')
        assert isinstance(out, pd.DataFrame)
    except Exception:
        assert False
