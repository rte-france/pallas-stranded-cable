"""."""
import matplotlib.pyplot as plt
import numpy as np

from strdcable.cable import StrandedCable

if __name__ == "__main__":
    # data from Judge et al : http://dx.doi.org/10.1016/j.conbuildmat.2011.12.073
    dexp = dict(
        strain=np.array([0, 0.00208, 0.006, 0.00852, 0.01216,
                         0.01636, 0.02184, 0.03088, 0.03744, 0.04692,
                         0.052, 0.0568, 0.066]),
        force=np.array([0, 0.36, 0.964, 1.522, 2.216,
                        2.648, 2.954, 3.212, 3.314, 3.422,
                        3.498, 3.534, 3.606])*1.0E06
    )
    dvol = dict(
        strain=np.array([0, 0.00864, 0.01088, 0.0134, 0.02112,
                         0.02868, 0.03552, 0.04548, 0.052, 0.0664]),
        force=np.array([0.0, 2.5, 2.862, 2.988, 3.084,
                        3.11, 3.176, 3.3, 3.334, 3.456])*1.0E06)

    dinput = dict(
        density=np.array([7.78]*7),
        alpha=np.array([1.92]*7)*1.0E-07,
        young=np.array([188.0]*7)*1.0E+09,
        poisson=np.array([3.3]*7),
        sigmay=np.array([1.540]*7)*1.0E+09,
        hardening=np.array([5.5]*7)*1.0E+09,
        sigmau=np.array([2.0]*7)*1.0E+09,
        dwires=np.array([5.8, 4.3, 3.2, 5.3, 5.0, 5.0, 5.0]) * 1.0E-03,
        nbwires=np.array([1, 7, 17, 14, 21, 27, 33]),
        laylengths=np.array([np.nan, 0.15, 0.21, 0.32, 0.42, 0.52, 0.62])
    )

    # initiate cable
    cable = StrandedCable(**dinput, compute_physics=True)

    # maximum tested strain
    epsmax = 0.065
    # number of tested strain values
    npts = 500

    # calculate the global response
    dres = dict()
    dres['force'], dres['strain'] = cable.calculate_tension_deformation_curve(epsmax, npts)

    # plot of force-strain curve
    fig, ax = plt.subplots()
    for d, lbl, stl in [(dexp, 'experimental results (Bridon 2007)', '--x'),
                        (dvol, '3D model results (Judge et al 2012)', '-'),
                        (dres, 'strdcable results', '-')]:
        ax.plot(d['strain']*1.0E+02, d['force']*1.0E-03, stl, label=lbl)
    ax.set_xlabel("Axial strain [%]")
    ax.set_ylabel("Mechanical tension [kN]")
    ax.set_xlim(0.0, 6.5)
    ax.set_ylim(0.0, 4.0E+03)
    ax.grid(True)
    ax.legend(loc='lower right')
    plt.show()
