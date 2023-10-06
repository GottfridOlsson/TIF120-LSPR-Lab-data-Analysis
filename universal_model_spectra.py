import numpy as np
import matplotlib.pyplot as plt
from dielectric_functions import *


# Calculate disk extinction crossection using model from 
# https://www.researchgate.net/publication/319937693_Universal_analytical_modeling_of_plasmonic_nanoparticles
def get_disk_crossection(wl, eps1_metal, eps2_metal, eps1_medium, diameter, height):
    R = diameter / height
    L = diameter

    eps_metal = eps1_metal + 1j * eps2_metal

    #https://www.rsc.org/suppdata/c6/cs/c6cs00919k/c6cs00919k1.pdf page 7
    eps_1 = - 0.479 - 1.36 * R**0.872
    V_1_per_V = 0.944
    a_12 = 7.05/(1-eps_1)
    a_14 = -10.9/R**0.98
    V_per_L3 = np.pi * (4 + 3 * (R - 1) * (2 * R + np.pi - 2)) / (24 * R**3)
    V_1 = V_1_per_V * V_per_L3 * L**3

    s = np.sqrt(eps1_medium) * L / wl

    A_1 = a_12 * s**2 + \
          (4 * np.pi**2 * 1j * V_1 / (3 * L**3)) * s**3 + \
          a_14 * s**4
    
    alpha = (eps1_medium / (4 * np.pi)) * V_1  / \
            (1 / (eps_metal/eps1_medium - 1) - 1/(eps_1 - 1) - A_1)

    sigma_ext = (4 * np.pi / (wl * eps1_medium)) * np.imag(alpha)

    return sigma_ext

if __name__ == "__main__":

    wl_min = 1239.8 / 3.2
    wl_max = 1239.8 / 0.3
    wl = np.linspace(wl_min, wl_max, 1000)
    energy = 1239.8 / wl

    eps1_metal, eps2_metal = eps_metals(wl, "Au")
    eps1_medium, eps2_medium = eps_constant(wl, 1.26)

    height = 20
    # https://pubs.acs.org/doi/epdf/10.1021/nn102166t fig 2
    for diameter in [52, 68, 92, 124, 157, 177, 197, 217, 245, 265, 317, 397, 497, 552]:
        V = 4 * np.pi * (diameter/2)**2 * height / 3
        sigma_ext = get_disk_crossection(wl, eps1_metal, eps2_metal, eps1_medium, diameter, height)

        plt.plot(energy, sigma_ext/V, label=f"Au, diameter {diameter}nm")

    plt.xlabel("Photon energy / eV")
    plt.ylabel("Extinction crossection ($\sigma_{ext}/V$) / nm$^{-1}$")
    plt.show()