**stranded-cable** has been thought as a generic tool based on analytical or simplified modelling. However a focus is given on overhead lines (OHL). Specific functions have been implemented for the concerned cables (so-called conductors).

.. code-block:: python

   from strdcable.cable import Conductor
   cnd = Conductor(**dinput)

Main evaluated characteristics
==============================

Geometrical properties
----------------------

* D : diameter (m)
* A : section (m^2)
* V : volume (m^3)
* m : lineic_mass (kg/m)

Generic physical properties
---------------------------

* EA : axial stiffness (Pa.m^2)
* EImin : minimal bending stiffness (Pa.m^4)
* EImax : maximal bending stiffness (Pa.m^4)
* RTS : rated tensile strength (N)
* c_dilat : global coefficient of dilatation (no unit)

Overhead lines conductors options
---------------------------------

* fictive stress (Poffenberger-Swart & f.y_max)
* lifespan using the Safe Border Line

Data included
=============

Normative data on OHL wires
---------------------------

Main physical characteristics of OHL conductors' wires are defined in normative documents. This file gives the required minimal values for given range of diameters (dwire_min <= d < dwire_max) and material. The characteritics might be :

* nature : nature relative to the given material (no unit)
* density : density of the given material (no unit)
* alpha : thermal dilatation coefficient (1/°C)
* young : young modulus (Pa)
* poisson : poisson ratio (no unit)
* sigma1pct : limit elastic stress (Pa)
* epsilonu : ultimate strain (no unit)
* sigmau : ultimate stress (Pa)
* resistivity_20deg : maximal electrical resistance at 20°C (Ohm.m)
* mass_resist_20deg : Cresist mass 20°C (1/°C)
