from matplotlib.pylab import norm
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats as stats
import os
import DelaySpread_und_RMS
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import weibull_min, norm, gamma, laplace
from sklearn.metrics import mean_squared_error
import pickle


#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Data 
data_flat = DelaySpread_und_RMS.data.flatten()
num_milliseconds = DelaySpread_und_RMS.num_milliseconds
num_delays = DelaySpread_und_RMS.num_delays
number = DelaySpread_und_RMS.number
with open('all_peaks_for_every_ms.pkl', 'rb') as f:
    peaks = pickle.load(f)
#print(f"All peaks: {peaks}")
num_peaks_ms = []
'''for ms_peaks in peaks:
    
    count = len(ms_peaks)
    num_peaks_ms.append(count)
#print( f"peaks number :{num_peaks_ms} ")
num_MPC = []

# Collect all peaks for the milliseconds in this second

for sec in range(0,25):
    num_MPC_1 = []
    start_ms = sec * 1000
    end_ms = min((sec + 1) * 1000, num_milliseconds)
    for ms in range(start_ms, end_ms):
        num_MPC_1.append(num_peaks_ms[ms])
    #print (f" peaks :{num_MPC_1}")
    
    num_MPC.append(num_MPC_1)
num_MPC = np.array (num_MPC)'''
#print(num_MPC)


delays =  DelaySpread_und_RMS.delays

RMS_DS_2 =  DelaySpread_und_RMS.rms_delay_spread_2_array
RMS_DS_2_per_second = []
Bc_2 =  np.load('co_bandwidth_of_every_24_ms.npy')
Bc_2_per_second = []

#Functions


def Weibull (val, num_bin):
    val = np.ravel(val)
    shape, loc, scale = weibull_min.fit(val, floc=0)
    bin_no = num_bin
    data_entries, bins = np.histogram(val, bin_no)
    bincenters = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
    pdf = weibull_min.pdf(bincenters, shape, loc=loc, scale=scale)
    #cdf = weibull_min.cdf(bincenters, shape, loc=loc, scale=scale)

    return pdf,  bincenters

def Normal (val, num_bin):
    val = np.ravel(val)
    mu, std = stats.norm.fit(val)
    num_bins = int(num_bin)
    x = np.linspace(min(val), max(val), num_bins)
    pdf = norm.pdf(x, mu, std)

    return pdf, x

def gamma_function (val, num_bin):
    val = np.ravel(val)
    shape, loc, scale = stats.gamma.fit(val)
    x = np.linspace(min(val), max(val), num_bin)
    pdf = gamma.pdf(x, shape, loc=loc, scale=scale)

    return pdf, x

def log (val, num_bin):
    val = np.ravel(val)
    abs_data = np.abs(val)
    abs_data = abs_data[abs_data > 0]
    shape, loc, scale = stats.lognorm.fit(abs_data, floc=0)
    x = np.linspace(min(abs_data), max(abs_data), num_bin)
    pdf = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)

    return pdf, x

def laplace_function (val, num_bin):
    val = np.ravel(val)
    loc, scale = stats.laplace.fit(val)
    x = np.linspace(min(val), max(val), num_bin)
    pdf = laplace.pdf(x, loc, scale)

    return pdf, x

def calculate_rmse(hist, pdf):
    rmse = np.sqrt(mean_squared_error(hist, pdf))
    return rmse

def CDF (val):
    bin_no = np.linspace(np.amin(val), np.amax(val), len(val))
    data_entries, bins = np.histogram(val, bin_no)
    data_entries = data_entries / sum(data_entries)
    cdf_meas_data = np.cumsum(data_entries)
    bincenters = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
    

    return bincenters, cdf_meas_data

'''#cdf fitting
bincenters, cdf_meas_data, popt_cumsum_gauss, gauss_cdf = CDF (data_flat)

plt.figure(figsize=(10,6))
plt.plot(bincenters, cdf_meas_data, label="Measured Data")
plt.plot(bincenters, gauss_cdf(bincenters, *popt_cumsum_gauss), color="green", label="Norm. Distr. CDF Fit")   #CDF fitting
plt.xlabel('Data Value')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)'''



################ peak APDP   ###############################################################################

num_bin = 15
num_MPC = [len(peaks) for peaks in peaks] 

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)
hist_vals, bins, patches= ax.hist(num_MPC, bins=num_bin, density=True,label='Histogram')
# Fit the updated data
weibull_pdf, x_vals_wb = Weibull(num_MPC,num_bin)
normal_pdf, x_vals_n = Normal(num_MPC,num_bin)
gamma_pdf, x_vals_g = gamma_function(num_MPC,num_bin)
lognorm_pdf, x_vals_ln = log(num_MPC,num_bin)
laplace_pdf, x_vals_lp = laplace_function(num_MPC,num_bin)


    # Plot the fitting curves
weibull_line, = ax.plot(x_vals_wb, weibull_pdf, 'r-', lw=2, alpha=0.6, label='Weibull PDF')
normal_line, = ax.plot(x_vals_n, normal_pdf, 'g-', lw=2, alpha=0.6, label='Normal PDF')
gamma_line, = ax.plot(x_vals_g, gamma_pdf, 'b-', lw=2, alpha=0.6, label='Gamma PDF')
lognorm_line, = ax.plot(x_vals_ln, lognorm_pdf, 'm-', lw=2, alpha=0.6, label='Log-normal PDF')
laplace_line, = ax.plot(x_vals_lp, laplace_pdf, 'c-', lw=2, alpha=0.6, label='Laplace PDF')

ax.set_title(f"MPC Histogram and Fitting")
ax.set_xlabel('MPC_APDP')
ax.set_ylabel('PDF')
ax.legend()
ax.grid(True)



'''#Error calculation
bin_centers, cdf_data = CDF(num_MPC)
#RMSE
cdf_weibull = np.cumsum (weibull_pdf)
cdf_normal = np.cumsum (normal_pdf)
cdf_gamma = np.cumsum (gamma_pdf)
cdf_lognormal = np.cumsum (lognorm_pdf)
cdf_laplace = np.cumsum (laplace_pdf)

rmse_weibull = calculate_rmse(cdf_data, cdf_weibull)
rmse_normal = calculate_rmse(cdf_data, cdf_normal)
rmse_gamma = calculate_rmse(cdf_data, cdf_gamma)
rmse_lognormal = calculate_rmse(cdf_data, cdf_lognormal)
rmse_laplace = calculate_rmse(cdf_data, cdf_laplace)

plt.figure(figsize=(10, 6))
plt.plot(x_vals_wb, rmse_weibull, 'r-', lw=2, alpha=0.6, label='Weibull')
plt.plot(x_vals_wb, rmse_normal, 'r-', lw=2, alpha=0.6, label='Normal')
plt.plot(x_vals_g, rmse_gamma, 'b-', lw=2, alpha=0.6, label='Gamma PDF')
plt.plot(x_vals_ln, rmse_lognormal, 'm-', lw=2, alpha=0.6, label='Log-normal ')
plt.plot(x_vals_lp, rmse_laplace, 'c-', lw=2, alpha=0.6, label='Laplace ')

plt.title('CDFs of Fitted Distributions with RMSE')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)'''

################ rms DS Frank   ###############################################################################
num_bin_1 = 40
RMS_DS_2 = RMS_DS_2*1e6
RMS_DS_2 = np.array([RMS_DS_2]).flatten()

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)
hist_5, bins_5, patches_5= ax.hist(RMS_DS_2, bins=num_bin_1, density=True,label='Histogram')
# Fit the updated data
weibull_5, x_vals_wb_5 = Weibull(RMS_DS_2,num_bin_1)
normal_5, x_vals_n_5 = Normal(RMS_DS_2,num_bin_1)
gamma_5, x_vals_g_5 = gamma_function(RMS_DS_2,num_bin_1)
lognorm_5, x_vals_ln_5 = log(RMS_DS_2,num_bin_1)
laplace_5, x_vals_lp_5 = laplace_function(RMS_DS_2,num_bin_1)


    # Plot the fitting curves
weibull_line, = ax.plot(x_vals_wb_5, weibull_5, 'r-', lw=2, alpha=0.6, label='Weibull PDF')
normal_line, = ax.plot(x_vals_n_5, normal_5, 'g-', lw=2, alpha=0.6, label='Normal PDF')
gamma_line, = ax.plot(x_vals_g_5, gamma_5, 'b-', lw=2, alpha=0.6, label='Gamma PDF')
lognorm_line, = ax.plot(x_vals_ln_5, lognorm_5, 'm-', lw=2, alpha=0.6, label='Log-normal PDF')
laplace_line, = ax.plot(x_vals_lp_5, laplace_5, 'c-', lw=2, alpha=0.6, label='Laplace PDF')


#RMSE
rmse_weibull5 = calculate_rmse(hist_5, weibull_5)
rmse_normal5 = calculate_rmse(hist_5, normal_5)
rmse_gamma5 = calculate_rmse(hist_5, gamma_5)
rmse_log5 = calculate_rmse(hist_5, lognorm_5)
rmse_laplace5 = calculate_rmse(hist_5, laplace_5)


plt.title("Probability of RMS Delay Spread")
plt.xlabel('RMS DS [us]')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)




################ coherence bandwidth Frank   ############################################################################### 
num_bin = update_num_bins (Bc_2)
Bc_2 = np.array(Bc_2).flatten()
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)
hist_7, bins_7, patches_7= ax.hist(Bc_2, bins=num_bin, density=True,label='Histogram')
# Fit the updated data
weibull_7, x_vals_wb_7 = Weibull(Bc_2,num_bin)
normal_7, x_vals_n_7 = Normal(Bc_2,num_bin)
gamma_7, x_vals_g_7 = gamma_function(Bc_2,num_bin)
lognorm_7, x_vals_ln_7 = log(Bc_2,num_bin)
laplace_7, x_vals_lp_7= laplace_function(Bc_2,num_bin)


    # Plot the fitting curves
weibull_line, = ax.plot(x_vals_wb_7, weibull_7, 'r-', lw=2, alpha=0.6, label='Weibull PDF')
normal_line, = ax.plot(x_vals_n_7, normal_7, 'g-', lw=2, alpha=0.6, label='Normal PDF')
gamma_line, = ax.plot(x_vals_g_7, gamma_7, 'b-', lw=2, alpha=0.6, label='Gamma PDF')
lognorm_line, = ax.plot(x_vals_ln_7, lognorm_7, 'm-', lw=2, alpha=0.6, label='Log-normal PDF')
laplace_line, = ax.plot(x_vals_lp_7, laplace_7, 'c-', lw=2, alpha=0.6, label='Laplace PDF')
#RMSE
rmse_weibull7 = calculate_rmse(hist_7, weibull_7)
rmse_normal7 = calculate_rmse(hist_7, normal_7)
rmse_gamma7 = calculate_rmse(hist_7, gamma_7)
rmse_log7 = calculate_rmse(hist_7, lognorm_7)
rmse_laplace7 = calculate_rmse(hist_7, laplace_7)



plt.title("Probability of Coherence Bandwidth")
plt.xlabel('Coherence Bandwidth 2 [Hz]')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)


plt.show()

