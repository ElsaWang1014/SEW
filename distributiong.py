from matplotlib.pylab import norm
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats as stats
import os
import DelaySpread_und_RMS
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import weibull_min, norm, gamma, laplace,kstest
from sklearn.metrics import mean_squared_error
from matplotlib.widgets import Slider
import pandas as pd
import tensorflow as tf

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Data 
data_flat = DelaySpread_und_RMS.data.flatten()
num_milliseconds = DelaySpread_und_RMS.num_milliseconds
num_delays = DelaySpread_und_RMS.num_delays

peaks = DelaySpread_und_RMS.all_peaks_all
print(f"All peaks: {peaks}")
num_peaks_ms = []
for ms_peaks in peaks:
    
    count = len(ms_peaks)
    num_peaks_ms.append(count)
print( f"peaks number :{num_peaks_ms} ")
num_MPC = []

# Collect all peaks for the milliseconds in this second

for sec in range(0,25):
    num_MPC_1 = []
    start_ms = sec * 1000
    end_ms = min((sec + 1) * 1000, num_milliseconds)
    for ms in range(start_ms, end_ms):
        num_MPC_1.append(num_peaks_ms[ms])
    print (f" peaks :{num_MPC_1}")
    
    num_MPC.append(num_MPC_1)
num_MPC = np.array (num_MPC)
print(num_MPC)


delays =  DelaySpread_und_RMS.delays

RMS_DS_2 =  DelaySpread_und_RMS.rms_delay_spread_2_array
RMS_DS_2_per_second = []
'''for sec in range(0,25):
    RMS_DS_2_for_second = np.zeros(num_milliseconds)
    start_ms = sec * 1000
    end_ms = min((sec + 1) * 1000, num_milliseconds)
    for ms in range(start_ms, end_ms):
        RMS_DS_2_for_second[ms - start_ms] = RMS_DS_2[ ms]
    print (f" rms ds :{RMS_DS_2_for_second}")
    # Collect all RMS_DS_2 for the milliseconds in this second
    if RMS_DS_2_for_second:
        RMS_DS_2_for_second = np.concatenate(RMS_DS_2_for_second)
    

    RMS_DS_2_per_second.append(RMS_DS_2_for_second)'''

Bc_2 =  DelaySpread_und_RMS.co_bandwidth_2
Bc_2_per_second = []
'''for sec in range(0,25):
    Bc_2_for_second = np.zeros(num_milliseconds)
    start_ms = sec * 1000
    end_ms = min((sec + 1) * 1000, num_milliseconds)
    for ms in range(start_ms, end_ms):
        Bc_2_for_second[ms - start_ms] = Bc_2[ ms]
    print (f" bc :{Bc_2_for_second}")
    # Collect all Bc_2 for the milliseconds in this second
    if Bc_2_for_second:
        Bc_2_for_second = np.concatenate(RMS_DS_2_for_second)
    

    Bc_2_per_second.append(Bc_2_for_second)'''

def data_per_second(val):
    val_per_second = []
    for sec in range(0,25):
        val_for_second = np.zeros(num_milliseconds)
        start_ms = sec * 1000
        end_ms = min((sec + 1) * 1000, num_milliseconds)
        for ms in range(start_ms, end_ms):
            val_for_second[ms - start_ms] = val[ms]

    val_per_second.append(val_for_second)
        
    return val_per_second



#Functions

def update_num_bins(val, resolution=3):
    #val = np.concatenate(val)
    
    edges = np.histogram_bin_edges(val, bins='auto')
    #bin_width = edges[1] - edges[0]  
    num_bins = len(edges) - 1  
    
    adjusted_num_bins = int(num_bins * resolution)
    
    return adjusted_num_bins


def CDF (val):
    bin_no = np.linspace(np.amin(val), np.amax(val), len(val))
    data_entries, bins = np.histogram(val, bin_no)
    data_entries = data_entries / sum(data_entries)
    cdf_meas_data = np.cumsum(data_entries)
    bincenters = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
    
    
    def gauss_cdf(data, loc, scale):
            return norm.cdf(data, loc=loc, scale=scale)
    
    popt_cumsum_gauss, pcov_cumsum_gauss = curve_fit(gauss_cdf, bincenters, cdf_meas_data, p0=[np.average(val), 1], maxfev=1000000)
  
    return bincenters, cdf_meas_data, popt_cumsum_gauss, gauss_cdf


def Weibull (val):
    num_MPC = np.ravel(num_MPC)
    shape, loc, scale = weibull_min.fit(val, floc=0)
    bin_no = num_bin
    data_entries, bins = np.histogram(val, bin_no)
    bincenters = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
    pdf = weibull_min.pdf(bincenters, shape, loc=loc, scale=scale)
    cdf = weibull_min.cdf(bincenters, shape, loc=loc, scale=scale)

    return pdf, cdf, bincenters

def Normal (val):
   
    mu, std = stats.norm.fit(val)
    num_bins = int(num_bin)
    x = np.linspace(min(val), max(val), num_bins)
    pdf = norm.pdf(x, mu, std)

    return pdf, x

def gamma_function (val):
    shape, loc, scale = stats.gamma.fit(val)
    x = np.linspace(min(val), max(val), num_bin)
    pdf = gamma.pdf(x, shape, loc=loc, scale=scale)

    return pdf, x

def log (val):
    abs_data = np.abs(val)
    abs_data = abs_data[abs_data > 0]
    shape, loc, scale = stats.lognorm.fit(abs_data, floc=0)
    x = np.linspace(min(abs_data), max(abs_data), num_bin)
    pdf = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)

    return pdf, x

def laplace_function (val):
    loc, scale = stats.laplace.fit(val)
    x = np.linspace(min(val), max(val), num_bin)
    pdf = laplace.pdf(x, loc, scale)

    return pdf, x

def calculate_rmse(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return rmse

def calculate_aic(log_likelihood, num_params):
    return 2 * num_params - 2 * log_likelihood

def x_vals (val,num_bin):
    x_vals = np.linspace(min(val), max(val), num_bin)

    return x_vals


#cdf fitting
bincenters, cdf_meas_data, popt_cumsum_gauss, gauss_cdf = CDF (data_flat)

plt.figure(figsize=(10,6))
plt.plot(bincenters, cdf_meas_data, label="Measured Data")
plt.plot(bincenters, gauss_cdf(bincenters, *popt_cumsum_gauss), color="green", label="Norm. Distr. CDF Fit")   #CDF fitting
plt.xlabel('Data Value')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)



################ peak APDP   ###############################################################################
'''#parameters

num_bin = update_num_bins (peaks)
pdf_p2,cdf_p2,weibull_p2 = Weibull (peaks)
shape_p2, loc_p2, scale_p2 = weibull_min.fit(peaks, floc=0)
mu_p2,std_p2,normal_p2 = Normal (peaks)
shape_gamma_p2, loc_gamma_p2, scale_gamma_p2,gamma_p2 = gamma (peaks)
shape_log_p2, loc_log_p2, scale_log_p2,log_p2 = log (peaks)
loc_laplace_p2, scale_laplace_p2,laplace_p2 = laplace (peaks)
hist_2, bin_edges_2 = np.histogram(peaks, bins=num_bin)

#figure
plt.figure(figsize=(10,6))
#Histogram
plt.hist(peaks,bins=num_bin,density=True,label='MPC_APDP')  
#Weibull                                                 
plt.plot(weibull_p2, pdf_p2, 'r-', lw=2, alpha=0.6, label='Weibull PDF')                          
#plt.plot(weibull_p1, cdf_p1, 'b-', lw=2, alpha=0.6, label='Weibull CDF')       
#Normal             
plt.plot(normal_p2, stats.norm.pdf(normal_p2, mu_p2, std_p2), lw=2, label='Normal Distribution')  
#Gamma
plt.plot(gamma_p2, stats.gamma.pdf(gamma_p2, shape_gamma_p2, loc_gamma_p2, scale_gamma_p2), lw=2, label='Gamma Distribution')
#Log
plt.plot(log_p2, stats.lognorm.pdf(log_p2, shape_log_p2, loc_log_p2, scale_log_p2), lw=2, label='Log-Normal Distribution')
#Laplace
plt.plot(laplace_p2, stats.laplace.pdf(laplace_p2, loc_laplace_p2,scale_laplace_p2), lw=2, label='Laplace Distribution')
'''
#num_bin = update_num_bins (num_MPC)



# update 
def update(val):
    
    sec = int(val)-1
    num_MPC_sec = num_MPC[sec]
    
    num_bin = update_num_bins([num_MPC_sec])
    ax.clear()
    hist_vals, bins,  patches = ax.hist(num_MPC_sec, bins=num_bin, density=True, label='Histogram')

    
    '''# Fit the updated data
    weibull_pdf, x_vals_wb = Weibull(num_MPC)
    normal_pdf, x_vals_n = Normal(num_MPC)
    gamma_pdf, x_vals_g = gamma_function(num_MPC)
    lognorm_pdf, x_vals_ln = log(num_MPC)
    laplace_pdf, x_vals_lp = laplace_function(num_MPC)


    # Plot the fitting curves
    weibull_line, = ax.plot(x_vals_wb, weibull_pdf, 'r-', lw=2, alpha=0.6, label='Weibull PDF')
    normal_line, = ax.plot(x_vals_n, normal_pdf, 'g-', lw=2, alpha=0.6, label='Normal PDF')
    gamma_line, = ax.plot(x_vals_g, gamma_pdf, 'b-', lw=2, alpha=0.6, label='Gamma PDF')
    lognorm_line, = ax.plot(x_vals_ln, lognorm_pdf, 'm-', lw=2, alpha=0.6, label='Log-normal PDF')
    laplace_line, = ax.plot(x_vals_lp, laplace_pdf, 'c-', lw=2, alpha=0.6, label='Laplace PDF')


    # Update the fitting curves
    weibull_line.set_ydata(weibull_pdf)
    normal_line.set_ydata(normal_pdf)
    gamma_line.set_ydata(gamma_pdf)
    lognorm_line.set_ydata(lognorm_pdf)
    laplace_line.set_ydata(laplace_pdf)'''


    ax.set_title(f"MPC Histogram and Fitting for Second {sec}")
    ax.set_xlabel('MPC_APDP')
    ax.set_ylabel('PDF')
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.canvas.draw_idle()


# figure
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)
# Initial plot: Get the first set of peaks
first_sec = 0
num_MPC_initial = num_MPC[first_sec]  # Get the peaks for the first millisecond
num_bin = update_num_bins([num_MPC_initial])
# Initial
hist_vals, bins, patches= ax.hist(num_MPC_initial, bins=num_bin, density=True,label='Histogram')
weibull_plot, = ax.plot([], [], 'r-', lw=2, alpha=0.6, label='Weibull PDF')
normal_plot, = ax.plot([], [], 'g-', lw=2, alpha=0.6, label='Normal PDF')
gamma_plot, = ax.plot([], [], 'b-', lw=2, alpha=0.6, label='Gamma PDF')
log_plot, = ax.plot([], [], 'm-', lw=2, alpha=0.6, label='Log-Normal PDF')
laplace_plot, = ax.plot([], [], 'c-', lw=2, alpha=0.6, label='Laplace PDF')

ax.set_title(f"MPC Histogram and Fitting for Second {sec}")
ax.set_xlabel('MPC_APDP')
ax.set_ylabel('PDF')
ax.legend(loc="upper right")
ax.grid(True)

'''weibull_p2 = Weibull (num_MPC)
#shape_p2, loc_p2, scale_p2 = weibull_min.fit(peaks_per_second, floc=0)
normal_p2 = Normal (num_MPC)
gamma_p2 = gamma_function (num_MPC)
log_p2 = log (num_MPC)
laplace_p2 = laplace_function (num_MPC)'''
# hist
#hist_2, bin_edges_2 = np.histogram(num_MPC, bins=num_bin)
#x_vals = np.linspace(min(delays_peaks), max(delays_peaks), 100)


# slider
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Seconds', valmin=1, valmax=25, valinit=1, valstep=1)
slider.on_changed(update)


plt.show()


'''#Error calculation
#RMSE
rmse_weibull2 = calculate_rmse(hist_vals, weibull_p2)
rmse_normal2 = calculate_rmse(hist_vals, normal_p2)
rmse_gamma2 = calculate_rmse(hist_vals, gamma_p2)
rmse_log2 = calculate_rmse(hist_vals, log_p2)
rmse_laplace2 = calculate_rmse(hist_vals, laplace_p2)
print("Weibull RMSE for APDP: ", rmse_weibull2)
print("Normal RMSE for APDP: ", rmse_normal2)
print("Gamma RMSE for APDP: ", rmse_gamma2)
print("Log RMSE for APDP: ", rmse_log2)
print("Laplace RMSE for APDP: ", rmse_laplace2)
#AIC 
aic_weibull2 = calculate_aic(weibull_p2, 3)  # Weibull has 3 parameters: shape, loc, scale
aic_normal2 = calculate_aic(normal_p2, 2)    # Normal has 2 parameters: mu, std
aic_gamma2 = calculate_aic(gamma_p2, 3)      # Gamma has 3 parameters: shape, loc, scale
aic_log2 = calculate_aic(log_p2, 3)          # Log-Normal has 3 parameters: shape, loc, scale
aic_laplace2= calculate_aic(laplace_p2, 2)  # Laplace has 2 parameters: loc, scale
print("Weibull AIC for APDP: ", aic_weibull2)
print("Normal AIC for APDP: ", aic_normal2)
print("Gamma AIC for APDP: ", aic_gamma2)
print("Log AIC for APDP: ", aic_log2)
print("Laplace AIC for APDP: ", aic_laplace2)
#KS Test
ks_weibull2, p_weibull2 = kstest(peaks, 'weibull_min', args=(shape_p2, loc_p2, scale_p2))
ks_normal2, p_normal2 = kstest(peaks, 'norm', args=(mu_p2,std_p2))
ks_gamma2, p_gamma2 = kstest(peaks, 'gamma', args=(shape_gamma_p2, loc_gamma_p2, scale_p2))
ks_log2, p_log2 = kstest(peaks, 'lognorm', args=(shape_log_p2, loc_log_p2, scale_log_p2))
ks_laplace2, p_laplace2 = kstest(peaks, 'laplace', args=(loc_laplace_p2, scale_laplace_p2))
print(f"KS-weibull for APDP: {ks_weibull2}, p-Wert for APDP: {p_weibull2}")
print(f"KS-Normal for APDP: {ks_normal2}, p-Wert for APDP: {p_normal2}")
print(f"KS-Gamma for APDP: {ks_gamma2}, p-Wert for APDP: {p_gamma2}")
print(f"KS-Log for APDP: {ks_log2}, p-Wert for APDP: {p_log2}")
print(f"KS-Laplace for APDP: {ks_laplace2}, p-Wert for APDP: {p_laplace2}")'''






'''


################ rms DS Frank   ###############################################################################
num_bin = update_num_bins (RMS_DS_2)
RMS_DS_2 = RMS_DS_2*1e6
RMS_DS_2 = np.array([RMS_DS_2]).flatten()
pdf_5,cdf_5,weibull_5 = Weibull (RMS_DS_2)
shape_5, loc_5, scale_5 = weibull_min.fit(RMS_DS_2, floc=0)
mu_5,std_5,normal_5 = Normal (RMS_DS_2)
shape_gamma_5, loc_gamma_5, scale_gamma_5,gamma_5 = gamma (RMS_DS_2)
shape_log_5, loc_log_5, scale_log_5,log_5 = log (RMS_DS_2)
loc_laplace_5, scale_laplace_5,laplace_5 = laplace (RMS_DS_2)
hist_5, bin_edges_5 = np.histogram(RMS_DS_2, bins=num_bin)
plt.figure(figsize=(10,6))
#Histogram
plt.hist(RMS_DS_2,bins=num_bin,density=True,label='RMS DS')  
#Weibull                                                 
plt.plot(weibull_5, pdf_5, 'r-', lw=2, alpha=0.6, label='Weibull PDF')                          
#plt.plot(weibull_p1, cdf_p1, 'b-', lw=2, alpha=0.6, label='Weibull CDF')       
#Normal             
plt.plot(normal_5, stats.norm.pdf(normal_5, mu_5, std_5), lw=2, label='Normal Distribution')  
#Gamma
plt.plot(gamma_5, stats.gamma.pdf(gamma_5, shape_gamma_5, loc_gamma_5, scale_gamma_5), lw=2, label='Gamma Distribution')
#Log
plt.plot(log_5, stats.lognorm.pdf(log_5, shape_log_5, loc_log_5, scale_log_5), lw=2, label='Log-Normal Distribution')
#Laplace
plt.plot(laplace_5, stats.laplace.pdf(laplace_5, loc_laplace_5,scale_laplace_5), lw=2, label='Laplace Distribution')
#RMSE
rmse_weibull5 = calculate_rmse(hist_5, pdf_5)
rmse_normal5 = calculate_rmse(hist_5, normal_5)
rmse_gamma5 = calculate_rmse(hist_5, gamma_5)
rmse_log5 = calculate_rmse(hist_5, log_5)
rmse_laplace5 = calculate_rmse(hist_5, laplace_5)
print("Weibull RMSE for RMS DS : ", rmse_weibull5)
print("Normal RMSE for RMS DS : ", rmse_normal5)
print("Gamma RMSE for RMS DS : ", rmse_gamma5)
print("Log RMSE for RMS DS : ", rmse_log5)
print("Laplace RMSE for RMS DS : ", rmse_laplace5)
#AIC 
aic_weibull5 = calculate_aic(weibull_5, 3)  # Weibull has 3 parameters: shape, loc, scale
aic_normal5 = calculate_aic(normal_5, 2)    # Normal has 2 parameters: mu, std
aic_gamma5 = calculate_aic(gamma_5, 3)      # Gamma has 3 parameters: shape, loc, scale
aic_log5 = calculate_aic(log_5, 3)          # Log-Normal has 3 parameters: shape, loc, scale
aic_laplace5 = calculate_aic(laplace_5, 2)  # Laplace has 2 parameters: loc, scale
print("Weibull AIC for RMS DS 2: ", aic_weibull5)
print("Normal AIC for  RMS DS 2: ", aic_normal5)
print("Gamma AIC for RMS DS 2: ", aic_gamma5)
print("Log AIC for RMS DS 2: ", aic_log5)
print("Laplace AIC for RMS DS 2: ", aic_laplace5)
#KS Test
ks_weibull5, p_weibull5 = kstest(RMS_DS_2, 'weibull_min', args=(shape_5, loc_5, scale_5))
ks_normal5, p_normal5 = kstest(RMS_DS_2, 'norm', args=(mu_5,std_5))
ks_gamma5, p_gamma5 = kstest(RMS_DS_2, 'gamma', args=(shape_gamma_5, loc_gamma_5, scale_5))
ks_log5, p_log5 = kstest(RMS_DS_2, 'lognorm', args=(shape_log_5, loc_log_5, scale_log_5))
ks_laplace5, p_laplace5 = kstest(RMS_DS_2, 'laplace', args=(loc_laplace_5, scale_laplace_5))
print(f"KS-weibull for RMS DS : {ks_weibull5}, p-Wert for RMS DS : {p_weibull5}")
print(f"KS-Normal for RMS DS : {ks_normal5}, p-Wert for RMS DS : {p_normal5}")
print(f"KS-Gamma for RMS DS : {ks_gamma5}, p-Wert for RMS DS : {p_gamma5}")
print(f"KS-Log for RMS DS : {ks_log5}, p-Wert for RMS DS : {p_log5}")
print(f"KS-Laplace for RMS DS : {ks_laplace5}, p-Wert for RMS DS : {p_laplace5}")
plt.title("Probability of RMS Delay Spread")
plt.xlabel('RMS DS [us]')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)




################ coherence bandwidth Frank   ############################################################################### 
num_bin = update_num_bins (Bc_2)
Bc_2 = np.array(Bc_2).flatten()
pdf_7,cdf_7,weibull_7 = Weibull (Bc_2)
shape_7, loc_7, scale_7 = weibull_min.fit(Bc_2, floc=0)
mu_7,std_7,normal_7 = Normal (Bc_2)
shape_gamma_7, loc_gamma_7, scale_gamma_7,gamma_7 = gamma (Bc_2)
shape_log_7, loc_log_7, scale_log_7,log_7 = log (Bc_2)
loc_laplace_7, scale_laplace_7,laplace_7 = laplace (Bc_2)
hist_7, bin_edges_7 = np.histogram(Bc_2, bins=num_bin)
pdf_values = hist_7 / (np.sum(hist_7) * (bin_edges_7[1] - bin_edges_7[0]))
print(f'pdf co bandwidth: {pdf_values}')
plt.figure(figsize=(10,6))
#Histogram
plt.hist(Bc_2,bins=num_bin,density=True,label='Coherence Bandwidth') 
#Weibull                                                 
plt.plot(weibull_7, pdf_7, 'r-', lw=2, alpha=0.6, label='Weibull PDF')                          
#plt.plot(weibull_p1, cdf_p1, 'b-', lw=2, alpha=0.6, label='Weibull CDF')       
#Normal             
plt.plot(normal_7, stats.norm.pdf(normal_7, mu_7, std_7), lw=2, label='Normal Distribution')  
#Gamma
plt.plot(gamma_7, stats.gamma.pdf(gamma_7, shape_gamma_7, loc_gamma_7, scale_gamma_7), lw=2, label='Gamma Distribution')
#Log
plt.plot(log_7, stats.lognorm.pdf(log_7, shape_log_7, loc_log_7, scale_log_7), lw=2, label='Log-Normal Distribution')
#Laplace
plt.plot(laplace_7, stats.laplace.pdf(laplace_7, loc_laplace_7,scale_laplace_7), lw=2, label='Laplace Distribution')
#RMSE
rmse_weibull7 = calculate_rmse(hist_7, pdf_7)
rmse_normal7 = calculate_rmse(hist_7, normal_7)
rmse_gamma7 = calculate_rmse(hist_7, gamma_7)
rmse_log7 = calculate_rmse(hist_7, log_7)
rmse_laplace7 = calculate_rmse(hist_7, laplace_7)
print("Weibull RMSE for Coherence Bandwidth : ", rmse_weibull7)
print("Normal RMSE for Coherence Bandwidth : ", rmse_normal7)
print("Gamma RMSE for Coherence Bandwidth : ", rmse_gamma7)
print("Log RMSE for Coherence Bandwidth : ", rmse_log7)
print("Laplace RMSE for Coherence Bandwidth : ", rmse_laplace7)
#AIC 
aic_weibull7 = calculate_aic(weibull_7, 3)  # Weibull has 3 parameters: shape, loc, scale
aic_normal7 = calculate_aic(normal_7, 2)    # Normal has 2 parameters: mu, std
aic_gamma7 = calculate_aic(gamma_7, 3)      # Gamma has 3 parameters: shape, loc, scale
aic_log7 = calculate_aic(log_7, 3)          # Log-Normal has 3 parameters: shape, loc, scale
aic_laplace7 = calculate_aic(laplace_7, 2)  # Laplace has 2 parameters: loc, scale
print("Weibull AIC for Coherence Bandwidth : ", aic_weibull7)
print("Normal AIC for  Coherence Bandwidth : ", aic_normal7)
print("Gamma AIC for Coherence Bandwidth : ", aic_gamma7)
print("Log AIC for Coherence Bandwidth : ", aic_log7)
print("Laplace AIC for Coherence Bandwidth : ", aic_laplace7)
#KS Test
ks_weibull7, p_weibull7 = kstest(Bc_2, 'weibull_min', args=(shape_7, loc_7, scale_7))
ks_normal7, p_normal7 = kstest(Bc_2, 'norm', args=(mu_7,std_7))
ks_gamma7, p_gamma7 = kstest(Bc_2, 'gamma', args=(shape_gamma_7, loc_gamma_7, scale_7))
ks_log7, p_log7 = kstest(Bc_2, 'lognorm', args=(shape_log_7, loc_log_7, scale_log_7))
ks_laplace7, p_laplace7 = kstest(Bc_2, 'laplace', args=(loc_laplace_7, scale_laplace_7))
print(f"KS-weibull for Coherence Bandwidth : {ks_weibull7}, p-Wert for Coherence Bandwidth : {p_weibull7}")
print(f"KS-Normal for Coherence Bandwidth : {ks_normal7}, p-Wert for Coherence Bandwidth : {p_normal7}")
print(f"KS-Gamma for Coherence Bandwidth : {ks_gamma7}, p-Wert for Coherence Bandwidth : {p_gamma7}")
print(f"KS-Log for Coherence Bandwidth : {ks_log7}, p-Wert for Coherence Bandwidth : {p_log7}")
print(f"KS-Laplace for Coherence Bandwidth : {ks_laplace7}, p-Wert for Coherence Bandwidth : {p_laplace7}")

plt.title("Probability of Coherence Bandwidth")
plt.xlabel('Coherence Bandwidth 2 [Hz]')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)


#plt.show()

'''

