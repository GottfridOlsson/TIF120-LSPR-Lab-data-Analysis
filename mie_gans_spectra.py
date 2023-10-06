import numpy as np
import matplotlib.pyplot as plt
from dielectric_functions import *



# Returns mie gans spectrum for a ellipsoidal nanoparticle with its symmetry axis
# parallell to the incident light given type of metal ("Au", "Ag" or "Cu"),
# its shape and the surrounding medium ("air", "SiO2", "glycol" or "eps=4").
# If medium is "glycol", glycon concentration is set by glycol_conc.
# If random_orientation is true, the mode which oscillated parallell to the symmetry axis
# will be included.
def get_mie_gans_crossection(wl, eps1_metal, eps2_metal, eps1_medium, diameter, height, lengthwise_mode=False):

    R = height/diameter
    P = np.zeros(3)
    
    # from https://core.ac.uk/download/pdf/287744118.pdf
    type = ""
    if (R == 1.0):
        type = "sphere"
        P[0] = 1/3
    elif (R > 1.0):
        type = "rod"
        e = np.sqrt(1 - (1/R)**2)
        P[0] = ((1-e**2)/e**2) * ((1/(2*e)) * np.log((1+e)/(1-e)) - 1)
    else: 
        type = "disk"
        e = np.sqrt(1 - R**2)
        P[0] = (1/e**2) * (1 - np.sqrt(1-e**2)/e * np.arcsin(e))

    P[1] = (1 - P[0])/2
    P[2] = (1 - P[0])/2
    V = 4 * np.pi * height * (0.5 * diameter)**2 / 3


    sigma_abs = np.zeros_like(wl)
    modes = []
    
    if lengthwise_mode: modes = [0, 1, 2]
    else: modes = [1, 2]

    for j in modes:
        sigma_abs += (1 / P[j]**2) * eps2_metal / (
            (eps1_metal + ((1-P[j])/P[j]) * eps1_medium)**2 + eps2_metal**2
        )

    sigma_abs *= ((2 * np.pi) / (3 * wl)) * eps1_medium**1.5 * V

    return sigma_abs



# Test by reproducing result of https://pubs.acs.org/doi/epdf/10.1021/jp990183f
# (Their result is wrong: https://pubs.acs.org/doi/epdf/10.1021/jp035241i)
# Corrected: https://pubs.acs.org/doi/epdf/10.1021/jp058091f
if __name__ == "__main__":

    particles = [
        {"metal": "Au", "diameter": 100, "height": 260},
        {"metal": "Au", "diameter": 100, "height": 290},
        {"metal": "Au", "diameter": 100, "height": 310},
        {"metal": "Au", "diameter": 100, "height": 330},
        {"metal": "Au", "diameter": 100, "height": 360}
    ]

    wl = np.linspace(300, 900, 500)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for particle in particles:

        eps1_metal, eps2_metal   = eps_metals(wl, particle["metal"])
        eps1_medium, eps2_medium = eps_constant(wl, 2.05) 

        sigma_abs = get_mie_gans_crossection(
            wl, 
            eps1_metal,
            eps2_metal,
            eps1_medium,
            particle["diameter"], 
            particle["height"], 
            lengthwise_mode=True)
        
        ax.plot(wl, sigma_abs, label=f'{particle["metal"]}, R = {particle["height"]/particle["diameter"]}')

    ax.legend()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorption crossection (nm^2)")
    plt.show()