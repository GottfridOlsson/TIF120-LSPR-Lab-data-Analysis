import numpy as np
import CSV_handler as CSV
import os
import matplotlib.pyplot as plt


def variance_y_of_linear_fit(var_k, var_m, cov_mk, x):
    # y = k*x + m have variance Var[y] = Var[k]*x^2 + 2*Cov[m,k]*x + Var[m]
    return var_k*x**2 + 2*cov_mk*x + var_m


# Read data #
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
datafile_path = CURRENT_PATH + '\\Data from lab\\LSPR_Exp2_Group1.txt'
data = np.genfromtxt(datafile_path, skip_header=8)

time = data[:,0]        # time / seconds
LSPR_peak = data[:,1]   # wavelength / nm


# Look at data #
if False:
    plt.plot(time, LSPR_peak)
    plt.xlabel("Time / s")
    plt.ylabel("LSPR peak / nm")
    plt.grid()
    plt.show()


# Pick out LSPR peaks for different concentrations of glycol #
concentrations = np.array([0, 5, 10, 20, 30, 50]) # percent glycol in water (zero means only water, values from lab)
LSPR_peak_time_ranges = [(60, 120), (240,300), (400,450), (550,600), (700,760), (810,870), (1060,1120)] # corresponding times for concentrations (values from plot)

LSPR_peak_avg, time_avg = [], []
for start, end in LSPR_peak_time_ranges:
    LSPR_peak_avg.append(np.average(np.select(np.logical_and(start <= time, time <= end), LSPR_peak)))
    time_avg.append((start+end)/2)

if False:
    plt.vlines(LSPR_peak_time_ranges, ymax=766.6, ymin=761.5, color='k',linestyles='--')
    plt.plot(time_avg, LSPR_peak_avg, color='r', marker='*', linestyle='')
    plt.show()


# Linear fit and plot #
# measure lambda (x) and want to know concentration (y)
LSPR_peaks = np.array(LSPR_peak_avg[0:-1])
LSPR_peak_unknown_concentration = LSPR_peak_avg[-1]
fit_coefficients, covariance = np.polyfit(LSPR_peaks, concentrations, deg=1, cov=True)
k_fit = fit_coefficients[0]
m_fit = fit_coefficients[1]

# Prediction interval from variance #
var_k = covariance[0,0]
var_m = covariance[1,1]
cov_mk = covariance[1,0]

x_plot = np.linspace(LSPR_peak_avg[0]-0.2, LSPR_peak_avg[-2]+0.2, 1000)
y_plot = x_plot*k_fit + m_fit

var_y_plot = variance_y_of_linear_fit(var_k, var_m, cov_mk, x_plot)
sigma_y_plot = np.sqrt(var_y_plot)
sigmas = 2

calculated_concentration = LSPR_peak_unknown_concentration*k_fit + m_fit
calculated_concentration_uncertainty = 2*np.sqrt(variance_y_of_linear_fit(var_k, var_m, cov_mk, LSPR_peak_unknown_concentration))

if True:
    plt.plot(LSPR_peaks, concentrations, color='k', linestyle='', marker='.', label='measured data')
    plt.plot(x_plot, y_plot, color='k', linestyle='-', marker='', label='fitted line')
    plt.fill_between(x_plot, y_plot + sigmas*sigma_y_plot, y_plot - sigmas*sigma_y_plot, alpha=0.85, label=f'Prediction interval {sigmas} sigma')
    #plt.plot(LSPR_peak_unknown_concentration, calculated_concentration, color='r', linestyle='', marker='s', label='unknown concentration')
    #plt.vlines(LSPR_peak_unknown_concentration, ymin=0, ymax=50,        color='k', linestyles='--', label='unknown concentration')
    plt.xlabel('LSPR peak / nm')
    plt.ylabel('Concentration of glycol in water %')
    plt.legend()
    plt.show()


# Sensitivity = d(lambda_LSPR) / d(refractive index) #
C_ethyleneGlycol = [0, 10, 20, 30, 40, 50]
n_ethyleneGlycol = [1.33300, 1.34242, 1.35238, 1.36253, 1.37275, 1.38313] # source: https://pubs.acs.org/doi/pdf/10.1021/ac60106a033

fit_coeffs_ethyleneGlycol = np.polyfit(C_ethyleneGlycol, n_ethyleneGlycol, deg=1)
concentration_linspace = np.linspace(0, 50, 1000)
lin_fit_ethyleneGlycol = fit_coeffs_ethyleneGlycol[0]*concentration_linspace + fit_coeffs_ethyleneGlycol[1]
refractive_indexes_ethyleneGlycol = fit_coeffs_ethyleneGlycol[0]*concentrations + fit_coeffs_ethyleneGlycol[1]
print(fit_coeffs_ethyleneGlycol)
if False:
    plt.plot(concentration_linspace, lin_fit_ethyleneGlycol, linestyle='--',label='fit')
    plt.plot(C_ethyleneGlycol, n_ethyleneGlycol, linestyle='', marker='.', label='old experiment (source)')
    plt.plot(concentrations, refractive_indexes_ethyleneGlycol, linestyle='', marker='*', label='n from measured C')
    plt.plot(n_linspace, lambda_fit, label='fitted line')
    plt.legend()
    plt.show()

fit_coeffs_n_vs_lambda, covariance_n_vs_lambda = np.polyfit(refractive_indexes_ethyleneGlycol, LSPR_peaks, deg=1, cov=True)
sensitivity = fit_coeffs_n_vs_lambda[0] # this is lambda per refractive index
twoSigma_sensitivity = 2*np.sqrt(covariance_n_vs_lambda[0,0])
print(fit_coeffs_n_vs_lambda)


n_linspace = np.linspace(1.3,1.4,1000)
lambda_fit = sensitivity*n_linspace + fit_coeffs_n_vs_lambda[1]

if False:
    plt.plot(n_linspace, lambda_fit, label='fitted line')
    plt.plot(refractive_indexes_ethyleneGlycol, LSPR_peaks, linestyle='', marker='*', label='measured data')
    plt.legend()
    plt.show()


# Print #
print(f"\nLinear fit: (concentration in percent) = k*(LSPR_peak_wavelength) + m, where k = {k_fit:.3f}/nm and m = {m_fit:.3f}.")
print(f"From measured LSPR_peak = {LSPR_peak_unknown_concentration:.3f} nm --> concentration = ({calculated_concentration:.3f} \pm {calculated_concentration_uncertainty:.3f}) % from linear fit (95 % confidence interval).")
print(f"Sensitivity of sensor is: ({sensitivity:.3f} \pm {twoSigma_sensitivity:.3f}) nm/RIU (95% confidence interval; RIU = refractive index unit)\n")

CSV.print_arrays_to_CSV(CURRENT_PATH+'\\Formatted CSV\\Measured_LSPR_peak_vs_concentration.csv', 
                        'Time / s', time,
                        'LSPR peak wavelength / nm', LSPR_peak)

CSV.print_arrays_to_CSV(CURRENT_PATH+'\\Formatted CSV\\LSPR-peak_vs_ethylene-glycol-concentration_fit.csv', 
                        'Measured (averaged) LSPR peak wavelength \ nm', LSPR_peaks,
                        'Given concentration ethylene glycol in water (percent)', concentrations, 
                        'Fitted LSPR peak wavelength \ nm', x_plot, 
                        'Fitted concentration ethylene glycol in water (percent)', y_plot, 
                        'One sigma uncertainty for fitted concentation ethylene glycol in water (percent)', sigma_y_plot)