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

mean_power = np.mean(data_db, axis=0)  # 计算每个延迟点的平均值，结果形状为 (512, 1000)

# 对 mean_power 进行平均，得到每个延迟点的平均功率
mean_power = np.mean(mean_power, axis=1) 

# Sampling interval (in seconds)
sampling_interval = 1e-3
num_delays = len(mean_power)
delays = np.arange(num_delays) * sampling_interval
# Calculate Delay Spread (max delay - min delay)
delay_spread = np.max(delays) - np.min(delays)
print(f'Delay Spread: {delay_spread} seconds')

# 计算 APDP
APDP = mean_power  # APDP 就是 mean_power

# 计算 APDP 的值和 RMS 延迟扩展
APDP_value = np.sum(delays * mean_power) / np.sum(mean_power)

# 计算 RMS 延迟扩展
rms_delay_spread = np.sqrt(np.sum((delays - APDP_value) ** 2 * mean_power) / np.sum(mean_power))
print(f'RMS Delay Spread: {rms_delay_spread} seconds')

# Figur
plt.figure(figsize=(20, 10))

plt.plot(delays*1000, APDP, color='b')
plt.xlabel("Delay Zeit")
plt.ylabel("APDP in dB")
plt.title(f"APDP in Rounds {round_numbers} and Second {second}")



plt.grid(True)
plt.show()
