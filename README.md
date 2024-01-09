# Stranded cable

_**stranded-cable**_ is a python package proposed to evaluate stranded cables' physical characteristics.

## Installation

### Using pip
To install the package using pip, execute the following command:
```shell script
python -m pip install git+https://gitlab.eurobios.com/rte/stranded-cable
```

### Using conda
(not available yet)

The package is available on conda-forge. To install, execute the following command: 
```shell script
python -m conda install stranded-cable -c conda-forge
```

## Building the documentation

First, make sure you have sphinx and the Readthedocs theme installed.

If you use pip, open a terminal and enter the following commands:
```shell script
pip install sphinx
pip install sphinx_rtd_theme
```

If you use conda, open an Anaconda Powershell Prompt and enter the following commands:
```shell script
conda install sphinx
conda install sphinx_rtd_theme
```

Then, in the same terminal or anaconda prompt, build the doc with:
```shell script
cd doc
make html
```

The documentation can then be accessed from `doc/_build/html/index.html`.

## Simple usage

This example defines a two-layer conductor cable directly from grades of steel and aluminium, speeding up the definition.
When using grades of materials implemented in this package, only the diameter of wires, their number per layer, 
the grade of materials and the lay length for each layer need to be defined. Mechanical and thermal properties are
already tabulated in the package.
```python
import numpy as np
from strdcable.cable import Conductor

cable_definition = dict(
    dwires=np.array([3e-3, 2e-3]),
    nbwires=np.array([1, 6]),
    material=np.array(['ST6C', 'AL1']),  # 'ST6C' is a grade of steel, 'AL1' is a grade of aluminium
    laylengths=np.array([np.nan, 0.2])
)

cable = Conductor(**cable_definition)
```

The export method of the StrandedCable class allows exporting all of the cable's properties in a dataframe:
```python
df = cable.export(format='dataframe')

>>> df[['nb_wires', 'wire_diameter', 'young_modulus', 'ultimate_stress']]
   nb_wires  wire_diameter  young_modulus  ultimate_stress
0         1          0.003   207000000000       1650000000
1         6          0.002    68000000000        185000000
```

Homogenized cable properties can be accessed directly from the StrandedCable object:
```python
>>> cable.A
2.5918139392115792e-05

>>> cable.m
0.10604410834231327

>>> cable.EA
2733197.4739603605

>>> cable.c_dilat
1.6843561209162503e-05
```

## Acknowledgements

_**stranded-cable**_ is developed by [Eurobios](http://www.eurobios.com/) and supported by [Rte-R&D](https://www.rte-france.com/) _via_ the OLLA project (see [ResearchGate](https://www.researchgate.net/project/OLLA-overhead-lines-lifespan-assessment)).
