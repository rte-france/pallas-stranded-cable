Install
=======

The latest version of the package can be installed using pip with the following
command::

    python -m pip install https://github.com/rte-france/pallas-stranded-cable/archive/refs/heads/main.zip

If you have the sources of the packages locally (through ``git clone`` or by
other means), you can install it in the current Python environment by running
the following command at the root of the repository::

    pip install -e .


Simple usage
============

Conductor cable using known materials
-------------------------------------

This example defines a two-layer conductor cable directly from grades of steel and aluminium, speeding up the definition.
When using grades of materials implemented in this package, only the diameter of wires, their number per layer,
the grade of materials and the lay length for each layer need to be defined. Mechanical and thermal properties are
already tabulated in the package.

.. code-block:: python

    import numpy as np
    from strdcable.cable import Conductor

    cable_definition = dict(
        dwires=np.array([3e-3, 2e-3]),
        nbwires=np.array([1, 6]),
        material=np.array(['ST6C', 'AL1']),  # 'ST6C' is a grade of steel, 'AL1' is a grade of aluminium
        laylengths=np.array([np.nan, 0.2])
    )

    cable = Conductor(**cable_definition)

The export method of the StrandedCable class allows exporting all of the cable's properties in a dataframe:

.. code-block:: text

    df = cable.export(format='dataframe')

    >>> df[['nb_wires', 'wire_diameter', 'young_modulus', 'ultimate_stress']]
       nb_wires  wire_diameter  young_modulus  ultimate_stress
    0         1          0.003   207000000000       1650000000
    1         6          0.002    68000000000        185000000

Homogenized cable properties can be accessed directly from the StrandedCable object:

.. code-block:: text

    >>> cable.A
    2.5918139392115792e-05

    >>> cable.m
    0.10604410834231327

    >>> cable.EA
    2733197.4739603605

    >>> cable.c_dilat
    1.6843561209162503e-05

Generic stranded cable
----------------------

This example defines a two-layer stranded cable with standard values for material properties (steel and aluminium)
and calculates its homogenized physical properties.
A StrandedCable object is created from a dictionary gathering the definition of layers and their properties:

.. code-block:: python

    import numpy as np
    from strdcable.cable import StrandedCable

    cable_definition = dict(
        dwires=np.array([3e-3, 2e-3]),      # Diameter of wires (m) in each layer starting from the central one
        nbwires=np.array([1, 6]),           # Number of wires
        density=np.array([7.8, 2.7]),       # Mass densities (g/cm^3)
        young=np.array([210e9, 70e9]),      # Young's moduli (Pa)
        sigmay=np.array([350e6, 276e6]),    # Yield strengths (Pa)
        sigmau=np.array([420e6, 310e6]),    # Ultimate strengths (Pa)
        alpha=np.array([1.15e-5, 1.7e-5]),  # Thermal expansion coefficients (K^-1)
        laylengths=np.array([np.nan, 0.2])  # We define lay lengths (m) for each layer. The central layer is straight with a single wire, so a NaN is inputted
    )

    cable = StrandedCable(**cable_definition)

The conductor's properties can be accessed similarly to the previous example:

.. code-block:: text

    df = cable.export(format='dataframe')

    >>> df[['nb_wires', 'wire_diameter', 'young_modulus', 'ultimate_stress']]
       nb_wires  wire_diameter  young_modulus  ultimate_stress
    0         1          0.003   2.100000e+11      420000000.0
    1         6          0.002   7.000000e+10      310000000.0

    >>> cable.A  # total area of the cross-section
    2.5918139392115792e-05

    >>> cable.m  # lineic mass
    0.10618548001172481

    >>> cable.EA  # axial rigidity
    2791756.1860059425

    >>> cable.c_dilat  # thermal expansion coefficient
    1.4075599241280196e-05
