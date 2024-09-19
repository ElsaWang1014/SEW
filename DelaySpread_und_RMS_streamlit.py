import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.signal import find_peaks
import streamlit as st


# load data
def load_data(load_path, round_numbers, num_seconds=25):
    data = []
    for second in range(1, num_seconds + 1):
        second_data = []
        for round_number in round_numbers:
            filename = f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"
            full_filename = os.path.join(load_path, filename)
            if os.path.exists(full_filename):
                mat = scipy.io.loadmat(full_filename)
                cirs_data = mat["cirs"]
                second_data.append(np.abs(cirs_data[28:440, :]) ** 2)  
            else:
                print(f" File {filename} not found.")
        data.append(second_data)
    
    data = np.array(data)
    data = np.concatenate(data, axis=2)
    return data

# APDP  &  RMS DS
def compute_apdp_and_peaks(data, num_milliseconds):
    delays = np.arange(data.shape[1]) * sampling_interval - 10e-9
    apdp_db_all = []
    all_peaks_all = []
    rms_delay_spread_array = np.zeros(num_milliseconds)
    co_bandwidth_array = np.zeros(num_milliseconds)

    for ms in range(num_milliseconds):
        apdp = np.mean(data[:, :, ms], axis=0)
        apdp_db = 10 * np.log10(apdp)

        '''max_index = np.argmax(apdp_db)
        apdp_db_after_max = apdp_db[max_index:]'''

        min_height = np.max(apdp_db[200:]) + 3
        peaks, _ = find_peaks(apdp_db, height=min_height, prominence=(0.1, None))
        #peaks = peaks + max_index

        total_power = np.sum(apdp[:200])
        time_weighted_power = np.sum(delays[:200] * apdp[:200])
        tau_bar = time_weighted_power / total_power
        squared_delays = (delays[:200] - tau_bar) ** 2
        rms_delay_spread = np.sqrt(np.sum(squared_delays * apdp[:200]) / total_power)
        rms_delay_spread_array[ms] = rms_delay_spread
        co_bandwidth_array[ms] = 1 / (2 * math.pi * rms_delay_spread)

        apdp_db_all.append(apdp_db)
        all_peaks_all.append(peaks)

    return apdp_db_all, all_peaks_all, rms_delay_spread_array, co_bandwidth_array

# DATA PATH
load_path = "/media/student/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_AGVHorizontal/"
round_numbers = [77, 78, 79, 80, 81, 82]
sampling_interval = 10e-9
data = load_data(load_path, round_numbers)
num_milliseconds = data.shape[2]
apdp_db_all, all_peaks_all, rms_delay_spread_array, co_bandwidth_array = compute_apdp_and_peaks(data, num_milliseconds)

# Streamlit APP
st.title("Average Power Delay Profile and RMS Delay Spread")

# Slider for milliseconds
ms = st.slider(
    'Select Millisecond',
    min_value=1,
    max_value=num_milliseconds,
    step=1
)
index = ms - 1
# Plot APDP
apdp_db = apdp_db_all[index]
peaks = all_peaks_all[index]
delays = np.arange(data.shape[1]) * sampling_interval - 10e-9

fig_apdp, ax_apdp = plt.subplots()
ax_apdp.plot(delays * 1e9, apdp_db, label='APDP (dB)')
ax_apdp.scatter(delays[peaks] * 1e9, apdp_db[peaks], color='red', label='Peaks')
ax_apdp.set_title(f"APDP for Rounds {round_numbers} at Millisecond {ms}")
ax_apdp.set_xlabel('Delay Time (ns)')
ax_apdp.set_ylabel('APDP (dB)')
ax_apdp.legend()

st.pyplot(fig_apdp)

# Plot RMS Delay Spread
fig_rms, ax_rms = plt.subplots()
ax_rms.plot(np.arange(1, num_milliseconds + 1), rms_delay_spread_array * 1e9, label='RMS Delay Spread (ns)')
ax_rms.set_title('RMS Delay Spread over Time')
ax_rms.set_xlabel('Time (ms)')
ax_rms.set_ylabel('RMS Delay Spread (ns)')
ax_rms.legend()

st.pyplot(fig_rms)