import numpy as np
import CSV_handler as CSV
import os
import matplotlib.pyplot as plt



# Read data #
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
datafile_path = CURRENT_PATH + '\\Data from lab\\LSPR_Exp2_Group1.txt'
data = np.genfromtxt(datafile_path, skip_header=8)

time = data[:,0]        # time / seconds
LSPR_peak = data[:,1]   # wavelength / nm




# Look at data #
plt.plot(time, LSPR_peak)
plt.xlabel("Time / s")
plt.ylabel("LSPR peak / nm")
plt.grid()
#plt.show()


# Pick out LSPR peaks for different concentrations of glycol #
concentrations = np.array([5, 10, 20, 30, 50]) # percent glycol in water (zero means only water, values from lab)
LSPR_peak_time_ranges = [(240,300), (400,450), (550,600), (700,760), (810,870), (1060,1120)] # corresponding times for concentrations (values from plot)

LSPR_peak_avg, time_avg = [], []
for start, end in LSPR_peak_time_ranges:
    LSPR_peak_avg.append(np.average(np.select(np.logical_and(start <= time, time <= end), LSPR_peak)))
    time_avg.append((start+end)/2)

plt.vlines(LSPR_peak_time_ranges, ymax=766.6, ymin=761.5, color='k',linestyles='--')
plt.plot(time_avg, LSPR_peak_avg, color='r', marker='*', linestyle='')
#plt.show()


# Linear fit and plot #
# measure lambda (x) and want to know concentration (y)
LSPR_peaks = np.array(LSPR_peak_avg[0:-1])
LSPR_peak_unknown_concentration = LSPR_peak_avg[-1]

fit_coefficients = np.polyfit(LSPR_peaks, concentrations, deg=1)
linear_fit_concentration = LSPR_peaks*fit_coefficients[0] + fit_coefficients[1]
calculated_concentration = LSPR_peak_unknown_concentration*fit_coefficients[0] + fit_coefficients[1]

plt.clf()
plt.plot(LSPR_peaks, concentrations,                                color='k', linestyle='', marker='.', label='data points')
plt.plot(LSPR_peaks, linear_fit_concentration,                      color='k', linestyle='-', marker='', label='fitted line')
plt.plot(LSPR_peak_unknown_concentration, calculated_concentration, color='r', linestyle='', marker='s', label='unknown concentration')
plt.vlines(LSPR_peak_unknown_concentration, ymin=0, ymax=50,        color='k', linestyles='--', label='unknown concentration')
plt.xlabel('LSPR peak / nm')
plt.ylabel('Concentration of glycol in water %')
plt.grid()
plt.legend()
plt.show()
