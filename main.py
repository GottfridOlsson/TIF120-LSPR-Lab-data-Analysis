import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mie_gans_spectra import get_mie_gans_crossection
from universal_model_spectra import get_disk_crossection
from dielectric_functions import *
from scipy.optimize import minimize

# Parameters
nominal_density = 7.125 # particles per um2 from lab pm picture
assumed_eps_medium = 1.26

# --- MEASURED SPECTRA --- #
data = np.genfromtxt("Data from lab/Part1_Group1_20230927_no_footer.csv", delimiter=",", skip_header=2)
spectra = {}
for i, sample in enumerate(["ref", "A", "B", "C", "D", "E", "F", "G"]):
    spectra[sample] = data[:,2*i:2*(i+1)]

# --- NOMINAL SPECTRA --- #
nominal_data = {
    "A": {"metal": "Au", "diameter": 140, "height": 30, "density": nominal_density},
    "B": {"metal": "Au", "diameter": 120, "height": 30, "density": nominal_density},
    "C": {"metal": "Ag", "diameter": 140, "height": 30, "density": nominal_density},
    "D": {"metal": "Au", "diameter":  99, "height": 30, "density": nominal_density},
    "E": {"metal": "Cu", "diameter": 140, "height": 30, "density": nominal_density},
}

for sample, particle in nominal_data.items():

    wl = np.linspace(300, 1100, 1000)

    eps1_metal, eps2_metal   = eps_metals(wl, particle["metal"])
    eps1_medium, eps2_medium = eps_constant(wl, assumed_eps_medium) 
    
    ext = 1e-6 * particle["density"] * get_disk_crossection(
        wl, eps1_metal, eps2_metal, eps1_medium,particle["diameter"], particle["height"])

    particle["wl"] = wl
    particle["ext"] = ext


# --- FITTED SPECTRA --- #

# Particles to fit to data with intitial values for fit
fitted_data = {
    "A": {"metal": "Au", "diameter": 140, "height": 30, "density": nominal_density},
    "B": {"metal": "Au", "diameter": 120, "height": 30, "density": nominal_density},
    "C": {"metal": "Ag", "diameter": 140, "height": 15, "density": nominal_density},
    "D": {"metal": "Au", "diameter":  99, "height": 30, "density": nominal_density},
    "E": {"metal": "Cu", "diameter": 140, "height": 10, "density": nominal_density},
}

def chi2_error(params, *args):
    diameter, height, density = params
    metal, wl, extinction = args
    eps1_metal, eps2_metal   = eps_metals(wl, metal)
    eps1_medium, eps2_medium = eps_constant(wl, assumed_eps_medium) 
    theoretical_extinction = 1e-6 * density * get_disk_crossection(wl, eps1_metal, eps2_metal, eps1_medium, diameter, height)
    return np.sqrt(np.sum((extinction - theoretical_extinction)**2))

for sample, particle in fitted_data.items():

    wl_measured  = spectra[sample][:,0]
    ext_measured = spectra[sample][:,1]

    x0 = np.array([particle["diameter"], particle["height"], nominal_density])
    args = (particle["metal"], wl_measured, ext_measured)
    res = minimize(chi2_error, x0, args=args, tol=1e-6)
    print(res.message)
    diameter, height, density = res.x
    particle["diameter"] = diameter
    particle["height"] = height
    particle["density"] = density

    wl = np.linspace(300, 1100, 1000)

    eps1_metal, eps2_metal   = eps_metals(wl, particle["metal"])
    eps1_medium, eps2_medium = eps_constant(wl, assumed_eps_medium)

    ext = 1e-6 * density * get_disk_crossection(
        wl, eps1_metal, eps2_metal, eps1_medium, diameter, height)
    
    particle["wl"] = wl
    particle["ext"] = ext


# --- Plotting --- #
fig = plt.figure()
ax = fig.add_subplot(3,1,1)
for i, sample in enumerate(["A", "B", "C", "D", "E"]):
    wl_measured  = spectra[sample][:,0]
    ext_measured = spectra[sample][:,1]
    ax.plot(wl_measured, ext_measured/np.max(ext_measured), label=f"Sample {sample}")
    
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

ax = fig.add_subplot(3,1,2)
for i, sample in enumerate(["A", "B", "C", "D", "E"]):
    wl_nominal = nominal_data[sample]["wl"]
    ext_nominal = nominal_data[sample]["ext"]
    ax.plot(wl_nominal, ext_nominal/np.max(ext_nominal), label=f'{nominal_data[sample]["metal"]}, $D = {nominal_data[sample]["diameter"]:.0f}$ nm, $h = {nominal_data[sample]["height"]:.0f}$ nm')
    
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.set_ylabel("Extinction (normalized)")
    

ax = fig.add_subplot(3,1,3)
for i, sample in enumerate(["A", "B", "C", "D", "E"]):
    wl_fitted = fitted_data[sample]["wl"]
    ext_fitted = fitted_data[sample]["ext"]
    ax.plot(wl_fitted, ext_fitted/np.max(ext_fitted), label=f'{fitted_data[sample]["metal"]}, $D = {fitted_data[sample]["diameter"]:.0f}$ nm, $h = {fitted_data[sample]["height"]:.0f}$ nm')

    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Wavelength / nm")





fig.tight_layout()
plt.show()