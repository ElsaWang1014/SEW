import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Informationen
load_path = "/media/campus/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_AGVHorizontal/"
round_numbers = [77,78,79,80,81,82]
second = 2

# die Daten für bestimmte Round und Zeit herunterladen
data_db = []
for round_number in round_numbers:
    filename = f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"
    full_filename = os.path.join(load_path, filename)
    if os.path.exists(full_filename):
      mat = scipy.io.loadmat(full_filename)
      cirs_data = mat["cirs"]
      data_db.append(10 * np.log10(np.abs(cirs_data)** 2))
    else:
       print(f"File {filename} not found.")
data_db = np.array(data_db) 

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
num_delays = data_db.shape[2]
delays = np.arange(num_delays) * sampling_interval

# Calculate Delay Spread and RMS Delay Spread
for i, round_number in enumerate(round_numbers):
    APDP = np.mean(data_db[i], axis=0)  # jeder round : APDP

    delay_spread = np.max(delays) - np.min(delays)
    print(f'Delay Spread: {delay_spread*1e6} us') 
    
    # Calculate RMS Delay Spread
    rms_delay_spread = np.sqrt(np.sum(delays **2 *APDP)/np.sum (APDP)- (np.sum(APDP * delays) / np.sum(APDP))** 2 )
    print(f'Round {round_number}: RMS Delay Spread: {rms_delay_spread*1e6} us')

    co_bandwidth = 1 / (2*math.pi*rms_delay_spread)
    print(f'Round {round_number}: coherence Banwidth: {co_bandwidth} ')
    # Plot APDP for each round
    plt.plot(delays*1000, APDP, label=f'Round {round_number}')



# Figure
plt.xlabel("Delay Time (seconds)")
plt.ylabel("APDP (dB)")
plt.title(f"APDP for Rounds {round_numbers} at Second {second}")
plt.legend()

plt.grid(True)
plt.show()

