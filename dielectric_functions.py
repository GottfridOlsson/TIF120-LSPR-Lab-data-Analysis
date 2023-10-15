import numpy as np

def eps_glycol(wavelength, concentration):
    # Dielectric constant of ethylene glycol solution
    # Source: https://www.researchgate.net/figure/Relation-between-solution-concentration-and-refractive-index_fig5_338015692
    n = (1.33293 + 0.09872 * concentration) * np.ones_like(wavelength)
    k = np.zeros_like(wavelength)
    eps1 = n**2 - k**2
    eps2 = 2 * n * k
    return eps1, eps2

def eps_metals(wavelength, metal):
    # Experimental dielectric constant of metals 
    # Source: https://refractiveindex.info/
    # Wavelength given in nanometers
    data       = np.genfromtxt("refractive_indices/" + metal + ".csv", skip_header=1)
    n = np.interp(wavelength, data[:,0]*1000, data[:,1])
    k = np.interp(wavelength, data[:,0]*1000, data[:,2])
    eps1 = (n**2 - k**2)
    eps2 = 2 * n * k
    return eps1, eps2

def eps_SiO2(wavelength):
    # Experimental dielectric constant of silica 
    # Source: https://refractiveindex.info/
    # Wavelength given in nanometers
    data = np.genfromtxt("refractive_indices/SiO2.csv", skip_header=1)
    n_SiO2 = np.interp(wavelength, data[:,0]*1000, data[:,1])
    eps1   = n_SiO2**2
    eps2   = 0 * np.ones_like(wavelength)

    return eps1, eps2

def eps_constant(wavelength, value):
    eps1 = value * np.ones_like(wavelength)
    eps2 = np.zeros_like(wavelength)
    return eps1, eps2

def eps_H2O(wavelength): return eps_constant(wavelength, 1.77)

def eps_air(wavelength): return eps_constant(wavelength, 1.00)

