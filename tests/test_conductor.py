import pytest
import numpy as np

def test_conductor():
    from strdcable.cable import Conductor
    cable = Conductor(
            dwires=np.array([3e-3, 2e-3]),
            nbwires=np.array([1, 6]),
            material=np.array(['ST6C', 'AL1']),  # 'ST6C' is a grade of steel, 'AL1' is a grade of aluminium
            laylengths=np.array([np.nan, 0.2])
            )

    assert cable.A == pytest.approx(2.5918139392115792e-05)
    assert cable.m == pytest.approx(0.10604410834231327)
    assert cable.EA == pytest.approx(2733197.4739603605)

