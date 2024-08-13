import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# Informationen
load_path = "/media/campus/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_AGVHorizontal/"
round_numbers = [77]
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

APDP = np.mean(np.mean(data_db, axis=0) , axis=1) 


# Sampling interval (in seconds)
sampling_interval = 1e-3
num_delays = len(APDP)
delays = np.arange(num_delays) * sampling_interval
# Calculate Delay Spread (max delay - min delay)
delay_spread = np.max(delays) - np.min(delays)
print(f'Delay Spread: {delay_spread} seconds') 

#  APDP_value  
APDP_value = np.sum(delays * APDP) / np.sum(APDP)

#  RMS 
rms_delay_spread = np.sqrt(np.sum((delays - APDP_value) ** 2 * APDP) / np.sum(APDP))
print(f'RMS Delay Spread: {rms_delay_spread} seconds')

# Figur
plt.figure(figsize=(20, 10))
plt.plot(delays*1000, APDP, color='b')
plt.xlabel("Delay Zeit")
plt.ylabel("APDP in dB")
plt.title(f"APDP in Rounds {round_numbers} and Second {second}")



plt.grid(True)
plt.show()
