import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.signal import find_peaks
from matplotlib.widgets import Slider


# Informationen
load_path = "/media/student/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_AGVHorizontal/"
#load_path = "/media/student/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_Langzeitmessungen/"

#wechselbar
#round_numbers = [22]#,26,30,35,38]
round_numbers = [77,78,79,80,81,82]
seconds = []
#RF_index = 0
#APDP_index = 1


for filenames in os.listdir(load_path):
  for round_number in round_numbers:
      #print(f"all file:{filenames}")
      if filenames.startswith(f"Round_{round_number}_AP_1_RF_1_Sec_") and filenames.endswith(".mat"):
        #print(filenames)
        r = int (filenames.split("_")[7].replace(".mat",""))
        seconds.append((round_number,r))

        #ordnen wie 1,2,3,...
seconds.sort(key=lambda x: (x[0], x[1]))
#print (f"second ist {seconds}")


# die Daten für bestimmte Round und Zeit herunterladen

data = []


for second in range(1,26):
  data_1 = []
  for round_number in round_numbers:
    
    filename = f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"
    full_filename = os.path.join(load_path, filename)
    if os.path.exists(full_filename):
      mat = scipy.io.loadmat(full_filename)
      cirs_data = mat["cirs"]
      #print(f"cir shape : {cirs_data.shape}") 
      data_1.append((np.abs(cirs_data[28:440,:])**2))
      
    else:
      print(f"File {filename} not found.")
    
  data.append(data_1)
  
data = np.array(data) 
#print (f"data shape: {data.shape}")
data = np.concatenate(data, axis=2)
#print (f"data shape: {data.shape}")


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
num_delays = data.shape[1]
delays = np.arange(num_delays) * sampling_interval
delays = delays - 10e-9
#print(num_delays)
#print("Initial shape of delays:", delays.shape)



#Parameters
num_milliseconds = data.shape[2]
#print(f"num_milliseconds:{data.shape[2]}")
APDP_db_all = []
all_peaks_all = []

rms_delay_spread_2_array = np.zeros(num_milliseconds)

co_bandwidth_2 = np.zeros(num_milliseconds)
all_peaks = np.zeros(num_delays*num_milliseconds)

# Calculate Delay Spread and RMS Delay Spread


#for s in [s for r, s in seconds if r == round_number]:
for ms in range(num_milliseconds):
        APDP_1 = np.zeros(num_delays)
        for i in range(num_delays):
          #print(num_delays)

        # Calculate APDP

          APDP_1[i] = np.mean(data[:,  i, ms], axis=0)  #  APDP for all delay position in a certain ms 
          #print (f"apdp 1:{APDP.shape}")
        
        #print (f"apdp:{APDP.shape}")

        APDP_db = 10 * np.log10(APDP_1)
        #print (f"apdp DB:{APDP_db.shape}")
        #peaks
        max_index = np.argmax(APDP_db)
        #print (f"max index :{max_index}")
        APDP_db_after_max = APDP_db[max_index:]
                
        min_height_2 = np.max(APDP_db[200:]) + 3
        peaks_2, peak_heights_2 = find_peaks(APDP_db_after_max, height = min_height_2, prominence = (0.1, None))
        peaks_2 = peaks_2 + max_index
        all_peaks = np.append(peaks_2, max_index)
        all_peaks = np.sort(all_peaks)
        APDP_db_all.append(APDP_db)
        all_peaks_all.append(all_peaks)
        #print(np.mean(APDP_db[200:]))


        
        # Calculate RMS Delay Spread (von Frank)

        total_power = np.sum(APDP_1[0:200])
        time_weighted_power = np.sum(delays[0:200] * APDP_1[0:200])
        tau_bar = time_weighted_power / total_power
        squared_delays = (delays[0:200] - tau_bar)**2
        rms_delay_spread_2 = np.sqrt(np.sum(squared_delays*APDP_1[0:200]) / total_power)
        rms_delay_spread_2_array[ms] = rms_delay_spread_2 
                #print(rms_delay_spread_2)
                #print(f'Round {round_number}: RMS Delay Spread: {rms_delay_spread_2*1e6} us')
  
        co_bandwidth_2[ms] = 1 / (2 * math.pi * rms_delay_spread_2)
                #print(f'Round {round_number}: coherence Banwidth: {co_bandwidth_2} ')


'''print (f"peaks shape: {len(all_peaks_all)}")
time_array = delays*num_milliseconds
all_peaks_all_1 = np.concatenate(all_peaks_all)
print(f"all peaks :{all_peaks_all_1.shape}")'''

# Figure with slider
fig, ax = plt.subplots(figsize=(20, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Initial plot
current_ms = 0
APDP_db = APDP_db_all[current_ms]
peaks = all_peaks_all[current_ms]
line, = ax.plot(delays * 1e9, APDP_db, label='APDP (dB)')
peaks_plot, = ax.plot(delays[peaks] * 1e9, APDP_db[peaks], 'rx', label='Peaks')
ax.set_xlabel("Delay Time (ns)")
ax.set_ylabel("APDP (dB)")
ax.set_title(f"APDP for Rounds {round_numbers} at Millisecond {current_ms}")
ax.legend()
ax.grid(True)

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Millisecond', 1, num_milliseconds , valinit=current_ms, valfmt='%d ms')

def update(val):
    ms = int(slider.val)
    APDP_db = APDP_db_all[ms]
    peaks = all_peaks_all[ms]
    line.set_ydata(APDP_db)
    peaks_plot.set_xdata(delays[peaks] * 1e9)
    peaks_plot.set_ydata(APDP_db[peaks])
    ax.relim()
    ax.autoscale_view()
    ax.set_title(f"APDP for Rounds {round_numbers} at Millisecond {ms}")
    fig.canvas.draw_idle()

slider.on_changed(update)



plt.figure(figsize=(20, 6))
plt.plot(rms_delay_spread_2_array * 1e9, label='RMS Delay Spread')
plt.xlabel("Time (milliseconds)")
plt.ylabel("RMS Delay Spread (ns)")
plt.title("RMS Delay Spread over Time")
plt.legend()
plt.grid(True)


plt.show()

