import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.signal import find_peaks

# Informationen
load_path = "/media/campus/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_AGVHorizontal/"
round_numbers = [77,78,79,80,81,82]
second = 2


# die Daten für bestimmte Round und Zeit herunterladen
data = []
for round_number in round_numbers:
    filename = f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"
    full_filename = os.path.join(load_path, filename)
    if os.path.exists(full_filename):
      mat = scipy.io.loadmat(full_filename)
      cirs_data = mat["cirs"]
      data.append((np.abs(cirs_data)** 2))
    else:
       print(f"File {filename} not found.")
data = np.array(data) 

#coherence Time
c_licht = 3e8
v = 0.6
f_c = 3.75e9  #Hz
#Doppler shift
f_m = (v * f_c) / c_licht 
T_c = 1 / f_m
print(f"Coherence Time: {T_c} s")


# Sampling interval (in seconds)
sampling_interval = 10e-9
num_delays = data.shape[2]
delays = np.arange(num_delays) * sampling_interval

#Delay Spread
delay_spread = np.max(delays) - np.min(delays)
print(f'Delay Spread: {delay_spread*1e6} us') 
num_milliseconds = data.shape[1]
mean_apdp = np.zeros(num_milliseconds)
#APDP_db_all = np.zeros((num_milliseconds, num_delays))

# Calculate Delay Spread and RMS Delay Spread
for ms in range(num_milliseconds):
    APDP = np.mean(data[:, ms, :], axis=0)  # jeder round : APDP
    APDP_db = 10 * np.log10(APDP)
    mean_apdp[ms] = np.mean(APDP_db)
    #APDP_db_all[ms, :] = APDP_db

    # Calculate RMS Delay Spread
    rms_delay_spread = np.sqrt(np.sum(delays **2 *APDP_db)/np.sum (APDP_db)- (np.sum(APDP_db * delays) / np.sum(APDP_db))** 2 )
    print(f'Round {round_number}: RMS Delay Spread: {rms_delay_spread*1e6} us')

    co_bandwidth = 1 / (2 * math.pi * rms_delay_spread)
    print(f'Round {round_number}: coherence Banwidth: {co_bandwidth} ')
    # Plot APDP for each round
    #plt.plot(delays*1000, APDP_db, label=f'Round {round_number}')


time_array = np.arange(num_milliseconds) * sampling_interval 
#time_array = np.arange(num_milliseconds) * 1e-3 
print(f"Length of time_array: {len(time_array)}")
print(f"Length of mean_apdp: {len(mean_apdp)}")


# Figure
plt.figure(figsize=(20, 6))
plt.plot (time_array, mean_apdp , label='APDP')
#for i in range(num_delays):
    #plt.plot(time_array, APDP_db_all[:, i], label=f'Delay {i}')
xticks = np.arange(0,np.max(time_array) ,0.2)
yticks = np.arange (np.min(mean_apdp) - 1, np.max(mean_apdp) + 1,5)
plt.xlabel("Delay Time (seconds)")
plt.xticks(xticks)
plt.yticks(yticks)
plt.tight_layout()
plt.ylim(np.min(mean_apdp) - 1, np.max(mean_apdp) + 1)
plt.ylabel("APDP (dB)")
plt.title(f"APDP for Rounds {round_numbers} at Second {second}")
plt.legend()

plt.grid(True)
plt.show()

