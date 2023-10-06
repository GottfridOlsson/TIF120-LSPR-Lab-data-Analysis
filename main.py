import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mie_gans_spectra import get_mie_gans_crossection
from universal_model_spectra import get_disk_crossection
from dielectric_functions import *
from scipy.optimize import minimize
import plot_functions as f
import os

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
f.set_LaTeX_and_CMU(True)
f.set_font_size(axis=11, tick=9, legend=7)
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16/2.54, 14/2.54), sharex=True, sharey=False)

for i, sample in enumerate(["A", "B", "C", "D", "E"]):
    wl_measured  = spectra[sample][:,0]
    ext_measured = spectra[sample][:,1]
    axs[0].plot(wl_measured, ext_measured/np.max(ext_measured), label=f"Sample {sample}")
    
    wl_nominal = nominal_data[sample]["wl"]
    ext_nominal = nominal_data[sample]["ext"]
    axs[1].plot(wl_nominal, ext_nominal/np.max(ext_nominal), label=f'{nominal_data[sample]["metal"]}, $D={nominal_data[sample]["diameter"]:3.0f}$\,nm, $h={nominal_data[sample]["height"]:2.0f}$\,nm')
    
    wl_fitted = fitted_data[sample]["wl"]
    ext_fitted = fitted_data[sample]["ext"]
    axs[2].plot(wl_fitted, ext_fitted/np.max(ext_fitted), label=f'{fitted_data[sample]["metal"]}, $D={fitted_data[sample]["diameter"]:3.0f}$\,nm, $h={fitted_data[sample]["height"]:2.0f}$\,nm')


x_labels = ["", "", "Wavelength / nm"]
y_labels = ["", "Normalized extinction", ""]
x_lims = [(None, None), (None, None), (280,1120)]
y_lims = [(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)]


for i, ax in enumerate(axs):
    #ax.set_yticks([])
    #f.set_axis_scale(   axs, xScale_string='linear', yScale_string='linear')
    f.set_axis_labels(  ax, x_label=x_labels[i], y_label=y_labels[i])
    #f.set_axis_invert(  ax, x_invert=False, y_invert=False)
    f.set_axis_limits(  ax, x_lims[i][0], x_lims[i][1], y_lims[i][0], y_lims[i][1])
    f.set_grid(         ax, grid_major_on=True, grid_major_linewidth=0.7, grid_minor_on=False, grid_minor_linewidth=0.3) # set_grid must be after set_axis_scale for some reason (at least with 'log')
    f.set_legend(       ax, legend_on=True, alpha=1.0, location='best')
    ax.yaxis.set_ticklabels([]) # remove labels but keep tick marks


f.align_labels(fig)
f.set_layout_tight(fig)
f.export_figure_as_pdf(os.path.abspath(os.path.dirname(__file__)) + "\\Figure\\TIF120_LSPR_measured_theoretical_spectra.pdf")
plt.show()