#%%
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import math
from scipy.signal import find_peaks
from matplotlib.widgets import Slider
import pickle


# Informationen
load_path = "/media/student/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_AGVHorizontal/"
#load_path = "/media/student/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_Langzeitmessungen/"

#wechselbar
#round_numbers = [22]#,26,30,35,38
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
T_c_1 = (9 * c_licht) / (16 * math.pi *f_c * v )
print(f"Coherence Time 1 : {T_c_1} s")
co_time_1 = round(T_c_1 * 1e3) 
print(f"Coherence Time: {co_time_1} ms")

#Doppler shift
f_d =  v * f_c / c_licht
T_c_2 =  1 / f_d
print(f"Coherence Time 2: {T_c_2} s")
co_time_2 = round(T_c_2 * 1e3) 
print(f"Coherence Time: {co_time_2} ms")

# Sampling interval (in seconds)
sampling_interval = 10e-9
num_delays = data.shape[1]
delays = np.arange(num_delays) * sampling_interval
delays = delays - 10e-9
#print(num_delays)
#print("Initial shape of delays:", delays_1.shape)
num_milliseconds = data.shape[2]
#print(f"num_milliseconds:{data.shape[2]}")
number_1 = int(np.ceil(num_milliseconds / co_time_1))
number_2 =  int(np.ceil(num_milliseconds / co_time_2))
#print(f"Number of samples for coherence time 1: {number_1}")

#Parameters

APDP_ms = []
#APDP_db_all = []
all_peaks_all_1 = []
all_peaks_all_2 = []

new_APDP_1 =  np.zeros(( number_1 * co_time_1 ,num_delays))
new_APDP_2 =  np.zeros(( number_2 * co_time_2 ,num_delays))
rms_delay_spread_array = np.zeros(num_milliseconds)
rms_delay_spread_1_array = np.zeros(number_1)
rms_delay_spread_2_array = np.zeros(number_2)
co_bandwidth = np.zeros(num_milliseconds)
co_bandwidth_1 = np.zeros(number_1)
co_bandwidth_2 = np.zeros(number_2)
all_peaks = np.zeros(num_delays*num_milliseconds)



#Figure
APDP_db_all_1 = np.load('Power_of_APDP_of_every_24.npy')
APDP_db_all_2 = np.load('Power_of_APDP_of_every_133.npy')
ms_final_1 = np.load('ms_of_every_24.npy')
ms_final_2 = np.load('ms_of_every_133.npy')
with open('all_peaks_for_every_24.pkl', 'rb') as file:
    all_peaks_all_1 = pickle.load(file)
with open('all_peaks_for_every_133.pkl', 'rb') as file:
    all_peaks_all_2 = pickle.load(file)

#%%
def plot_apdp_with_slider(data, peaks_all, ms_delay, co_time, round_numbers):
    # Create figure and axis
    
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    # Initial plot
    current_index = 0
    #APDP_db = data[current_index]
    peaks = peaks_all[current_index]
    
    # Plot the APDP (dB) line and peaks
    line, = ax.plot(ms_delay * 1e9, data[current_index], label='APDP (dB)')
    peaks_plot, = ax.plot(ms_delay[peaks] * 1e9, data[current_index, peaks], 'rx', label='Peaks')

    # Set labels and title
    ax.set_xlabel("Delay Time (ns)")
    ax.set_ylabel("APDP (dB)")
    ax.set_title(f"APDP for Rounds {round_numbers} from {current_index * co_time} ms to {current_index * co_time + co_time} ms")
    ax.legend()
    ax.grid(True)

    # Create a slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Millisecond', 1, len(data) , valinit=1, valfmt='%d.th')
    
    # Update function for the slider
    def update(val):
        index = int(slider.val) - 1
        #APDP_db = data[index]
        peaks = peaks_all[index]
        
        # Update the plot data
        line.set_ydata(data[index, :])
        peaks_plot.set_xdata(ms_delay[peaks] * 1e9)
        peaks_plot.set_ydata(data[index, peaks])
        
        # Rescale and redraw
        
        ax.set_title(f"APDP for Rounds {round_numbers} from {index * co_time} ms to {index * co_time + co_time} ms")
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    # Connect slider to the update function
    slider.on_changed(update)
    #interact(update);
    
plot_apdp_with_slider(APDP_db_all_1, all_peaks_all_1, ms_final_1, co_time_1, round_numbers)
plot_apdp_with_slider(APDP_db_all_2, all_peaks_all_2, ms_final_2, co_time_2, round_numbers)


rms_delay_spread_array = np.load('rms_delay_spread_of_every_1000_ms.npy')
plt.figure(figsize=(20, 6))
plt.plot(rms_delay_spread_array * 1e9, label='RMS Delay Spread (jede ms)')
plt.xlabel("Time (milliseconds)")
plt.ylabel("RMS Delay Spread (ns)")
plt.title("RMS Delay Spread every ms over Time")
plt.legend()
plt.grid(True)

rms_delay_spread_array_1 = np.load('rms_delay_spread_of_every_24_ms.npy')
plt.figure(figsize=(20, 6))
plt.plot(rms_delay_spread_array_1 * 1e9, label='RMS Delay Spread (jede 24ms)')
plt.xlabel("Time (milliseconds)")
plt.ylabel("RMS Delay Spread (ns)")
plt.title("RMS Delay Spread every 24 ms over Time")
plt.legend()
plt.grid(True)

rms_delay_spread_array_2 = np.load('rms_delay_spread_of_every_133_ms.npy')
plt.figure(figsize=(20, 6))
plt.plot(rms_delay_spread_array_2 * 1e9, label='RMS Delay Spread (jede 133ms)')
plt.xlabel("Time (milliseconds)")
plt.ylabel("RMS Delay Spread (ns)")
plt.title("RMS Delay Spread every 133 ms over Time")
plt.legend()
plt.grid(True)

plt.show()



'''# Calculate the APDP

for ms in range(0, num_milliseconds):
    APDP_1 = np.zeros(num_delays)
    for i in range(num_delays):
        # Compute APDP for each delay position at a specific millisecond
        APDP_1[i] = np.mean(data[:, i, ms], axis=0)  # APDP for all delay positions at a certain ms
    APDP_ms.append(APDP_1)
    #print (f" shape of APDP mean :{APDP_mean.shape}")
APDP_ms = np.array(APDP_ms)
    #print(f"shape of APDP_power_array: {APDP_power_array.shape}")
APDP_ms_power = 10 * np.log10(APDP_ms)
np.save ('APDP_of_every_ms',APDP_ms)
np.save ('Power_of_APDP_of_every_ms',APDP_ms_power)

#Calculate the APDP with coherence time
def  APDP_with_coherence_time(data, co_time, num_delays, number):

  new_APDP =  np.zeros(( number * co_time ,num_delays))
  new_APDP[:num_milliseconds,:] = data
  reshaped_APDP = new_APDP.reshape( number, co_time ,num_delays)
  mean_APDP_time = reshaped_APDP.mean(axis=1)
  APDP_power = 10 * np.log10(mean_APDP_time)
  APDP_db_all =  APDP_power

   # Calculate the number of milliseconds
  num_ms = APDP_db_all.shape[1]

  # Calculate the ms_final array
  ms_final = np.arange(num_ms) * sampling_interval

  power_filename = f'Power_of_APDP_of_every_{co_time}.npy'
  MS_filename = f'ms_of_every_{co_time}.npy'
  np.save (power_filename,APDP_db_all)
  np.save (MS_filename,ms_final)

 

  # Return APDP_db_all and ms_final
  return mean_APDP_time, APDP_db_all, ms_final
  
#peaks 
def  find_peaks_in_data(data,co_time):
  all_peaks_all = []
  for index in range(len(data)):
    current_segment = data[index]
    max_index = np.argmax(current_segment)
    APDP_db_after_max = current_segment[max_index +1:] 
                    
    min_height = np.max(data[:,200:]) + 3
    peaks, peak_heights = find_peaks(APDP_db_after_max, height = min_height, prominence = (0.1, None))
    peaks = peaks + max_index +1
    all_peaks = np.append(peaks, max_index)
    all_peaks = np.sort(all_peaks)
            
    all_peaks_all.append(all_peaks)
    #print(all_peaks.shape)

  filename = f'all_peaks_for_every_{co_time}.pkl'
  
  with open(filename, 'wb') as f:
      pickle.dump(all_peaks_all, f)

  return  all_peaks_all

      
#Calculate the RMS Delay Spread
def  calculate_rms_delay_spread(data,num_time,num_delay,co_time):
    rms_delay_spread_array = np.zeros(num_time)
    co_bandwidth = np.zeros(num_time)

    for time in range(num_time):
        
        total_power = np.sum(data[time ,:])
        time_weighted_power = np.sum(num_delay * data[time,:])
        tau_bar = time_weighted_power / total_power
        squared_delays = (num_delay - tau_bar)**2
        rms_delay_spread = np.sqrt(np.sum(squared_delays*data[time,:]) / total_power)
        rms_delay_spread_array[time] = rms_delay_spread 
                #print(rms_delay_spread_2)
                #print(f'Round {round_number}: RMS Delay Spread: {rms_delay_spread_2*1e6} us')
  
        co_bandwidth[time] = 1 / (2 * math.pi * rms_delay_spread)        
                #print(f'Round {round_number}: coherence Banwidth: {co_bandwidth_2} '        )

    rms_filename = f'rms_delay_spread_of_every_{co_time}_ms.npy'
    bandwidth_filename = f'co_bandwidth_of_every_{co_time}_ms.npy'

    np.save(rms_filename, rms_delay_spread_array)
    np.save(bandwidth_filename, co_bandwidth)


    return rms_delay_spread_array, co_bandwidth

def plot_apdp_with_slider(data, peaks_all, ms_delay, co_time, round_numbers):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    # Initial plot
    current_index = 0
    #APDP_db = data[current_index]
    peaks = peaks_all[current_index]
    
    # Plot the APDP (dB) line and peaks
    line, = ax.plot(ms_delay * 1e9, data[current_index], label='APDP (dB)')
    peaks_plot, = ax.plot(ms_delay[peaks] * 1e9, data[current_index, peaks], 'rx', label='Peaks')

    # Set labels and title
    ax.set_xlabel("Delay Time (ns)")
    ax.set_ylabel("APDP (dB)")
    ax.set_title(f"APDP for Rounds {round_numbers} from {current_index * co_time} ms to {current_index * co_time + co_time} ms")
    ax.legend()
    ax.grid(True)

    # Create a slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Millisecond', 1, len(data) , valinit=1, valfmt='%d.th')

    # Update function for the slider
    def update(val):
        index = int(slider.val) - 1
        #APDP_db = data[index]
        peaks = peaks_all[index]
        
        # Update the plot data
        line.set_ydata(data[index, :])
        peaks_plot.set_xdata(ms_delay[peaks] * 1e9)
        peaks_plot.set_ydata(data[index, peaks])
        
        # Rescale and redraw
        
        ax.set_title(f"APDP for Rounds {round_numbers} from {index * co_time} ms to {index * co_time + co_time} ms")
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    # Connect slider to the update function
    slider.on_changed(update)
    plt.show()

#Results
mean_APDP_time_1,APDP_db_all_1 , ms_final_1 = APDP_with_coherence_time(APDP_ms,co_time_1,num_delays,number_1)
mean_APDP_time_2,APDP_db_all_2 , ms_final_2 = APDP_with_coherence_time(APDP_ms,co_time_2,num_delays,number_2)

all_peaks_all_1 =  find_peaks_in_data(APDP_db_all_1,co_time_1)
all_peaks_all_2 =  find_peaks_in_data(APDP_db_all_2,co_time_2)

rms_delay_spread_array ,co_bandwidth  =  calculate_rms_delay_spread(APDP_ms,num_milliseconds,delays,1000)
rms_delay_spread_array_1 ,co_bandwidth_1  =  calculate_rms_delay_spread(mean_APDP_time_1,number_1,ms_final_1,co_time_1)
rms_delay_spread_array_2 ,co_bandwidth_2  =  calculate_rms_delay_spread(mean_APDP_time_2,number_2,ms_final_2,co_time_2)

#plot_apdp_with_slider(APDP_db_all_1, all_peaks_all_1, ms_final_1, co_time_1, round_numbers)
#plot_apdp_with_slider(APDP_db_all_2, all_peaks_all_2, ms_final_2, co_time_2, round_numbers)
'''



'''
num_ms_1 = APDP_db_all_1.shape[1]
ms_final_1 = np.arange(num_ms_1) * sampling_interval
#print("Initial shape of delay:", ms_final.shape)

num_ms_2 = APDP_db_all_2.shape[1]
ms_final_2 = np.arange(num_ms_2) * sampling_interval
#print("Initial shape of delay:", ms_final.shape)

# Calculate RMS Delay Spread (mit jede ms)
for ms in range(num_milliseconds):
        
        total_power_1 = np.sum(APDP_ms[ ms ,:])
        time_weighted_power_1 = np.sum(delays * APDP_ms[ ms,:])
        tau_bar_1 = time_weighted_power_1 / total_power_1
        squared_delays_1 = (delays - tau_bar_1)**2
        rms_delay_spread_1 = np.sqrt(np.sum(squared_delays_1*APDP_ms[ ms,:]) / total_power_1)
        rms_delay_spread_1_array[ms] = rms_delay_spread_1 
                #print(rms_delay_spread_2)
                #print(f'Round {round_number}: RMS Delay Spread: {rms_delay_spread_2*1e6} us')
  
        co_bandwidth_1[ms] = 1 / (2 * math.pi * rms_delay_spread_1)        
                #print(f'Round {round_number}: coherence Banwidth: {co_bandwidth_2} '        )
np.save ('rms_delay_spread_of_every_ms',rms_delay_spread_1_array)
np.save ('co_bandwidth_of_every_ms',co_bandwidth_1)


# Calculate RMS Delay Spread (für Eike Wunsch)
for time in range(number):
        
        total_power = np.sum( mean_APDP_time[ time ,:])
        time_weighted_power = np.sum(ms_final * mean_APDP_time[ time,:])
        tau_bar = time_weighted_power / total_power
        squared_delays = (ms_final - tau_bar)**2
        rms_delay_spread_2 = np.sqrt(np.sum(squared_delays*mean_APDP_time[ time,:]) / total_power)
        rms_delay_spread_2_array[time] = rms_delay_spread_2 
                #print(rms_delay_spread_2)
                #print(f'Round {round_number}: RMS Delay Spread: {rms_delay_spread_2*1e6} us')
  
        co_bandwidth_2[time] = 1 / (2 * math.pi * rms_delay_spread_2 )       
                #print(f'Round {round_number}: coherence Banwidth: {co_bandwidth_2} '        )

np.save ('rms_delay_spread_of_every_24_ms',rms_delay_spread_2_array)
np.save ('co_bandwidth_of_every_24_ms',co_bandwidth_2)

# Calculate RMS Delay Spread (mit jede T_c_2)
for time in range(number_2):
        
        total_power_3 = np.sum(mean_APDP_time_2[time ,:])
        time_weighted_power_3= np.sum(ms_final_2 * mean_APDP_time_2[ time ,:])
        tau_bar_3 = time_weighted_power_3 / total_power_3
        squared_delays_3 = (ms_final_2 - tau_bar_3)**2
        rms_delay_spread_3 = np.sqrt(np.sum(squared_delays_3*mean_APDP_time_2[time,:]) / total_power_3)
        rms_delay_spread_3_array[ms] = rms_delay_spread_3 
                #print(rms_delay_spread_2)
                #print(f'Round {round_number}: RMS Delay Spread: {rms_delay_spread_2*1e6} us')
  
        co_bandwidth_3[ms] = 1 / (2 * math.pi * rms_delay_spread_3)        
                #print(f'Round {round_number}: coherence Banwidth: {co_bandwidth_2} '        )
np.save ('rms_delay_spread_of_every_co_time2_ms',rms_delay_spread_3_array)
np.save ('co_bandwidth_of_every_co_time2_ms',co_bandwidth_3)
'''


'''
# Figure je ms
fig, ax = plt.subplots(figsize=(20, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Initial plot
current_ms = 0

APDP_db = APDP_ms_power[current_ms]
peaks_1 = all_peaks_all_1[current_ms]
line_1, = ax.plot(delays* 1e9, APDP_ms_power[current_ms, :], label='APDP (dB)')
peaks_plot_1, = ax.plot(delays[peaks_1] , APDP_ms_power[current_ms, peaks_1], 'rx', label='Peaks')

ax.set_xlabel("Delay Time (ns)")
ax.set_ylabel("APDP (dB)")
#ax.set_title(f"APDP for Rounds {round_numbers} at {current_ms} ms")
ax.legend()
ax.grid(True)

# Slider
ax_slider_1 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_1 = Slider(ax_slider_1, 'Millisecond', 1, num_milliseconds , valinit=current_ms + 1, valfmt='%d ms')

def update_1(val):
    ms = int(slider_1.val)
    #print(f"Slider index: {index}") 
    APDP_db = APDP_ms_power[ms]
    peaks = all_peaks_all_1[ms]
    #line.set_xdata([ms_final[index] * 1e9])
    line_1.set_ydata(APDP_ms_power[ms,:])
    #print (f"ydata: {APDP_db}")
    peaks_plot_1.set_xdata(delays[peaks] * 1e9)
    peaks_plot_1.set_ydata(APDP_ms_power[ms, peaks])
    ax.relim()
    ax.autoscale_view()
    ax.set_title(f"APDP for Rounds {round_numbers} at {ms + 1} ms")
    fig.canvas.draw_idle()

slider_1.on_changed(update_1)'''

'''# Sampling interval (in seconds)
sampling_interval = 10e-9
num_delays = data.shape[1]
delays = np.arange(num_delays) * sampling_interval
delays = delays - 10e-9


    # Show the plot
    #plt.show()

# Figure je 24 ms
fig, ax = plt.subplots(figsize=(20, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Initial plot
current_idex = 0
APDP_db = APDP_db_all_1[current_idex]
peaks = all_peaks_all_1[current_idex]
line, = ax.plot(ms_final_1* 1e9, APDP_db_all_1[current_idex, :], label='APDP (dB)')
peaks_plot, = ax.plot(ms_final_1[peaks] , APDP_db_all_1[current_idex, peaks], 'rx', label='Peaks')

ax.set_xlabel("Delay Time (ns)")
ax.set_ylabel("APDP (dB)")
ax.set_title(f"APDP for Rounds {round_numbers} from {current_idex*co_time_1} ms to  {current_idex*co_time_1+co_time_1} ms")
ax.legend()
ax.grid(True)

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Millisecond', 0, len(APDP_db_all_1) -1, valinit=current_idex + 1, valfmt='%d.th 24 ms')

def update(val):
    index = int(slider.val)
    #print(f"Slider index: {index}") 
    APDP_db = APDP_db_all_1[index]
    peaks = all_peaks_all_1[index]
    #line.set_xdata([ms_final[index] * 1e9])
    line.set_ydata(APDP_db_all_1[index,:])
    #print (f"ydata: {APDP_db}")
    peaks_plot.set_xdata(ms_final_1[peaks] * 1e9)
    peaks_plot.set_ydata(APDP_db_all_1[index, peaks])
    ax.relim()
    ax.autoscale_view()
    ax.set_title(f"APDP for Rounds {round_numbers} from {index*co_time_1} ms to  {index*co_time_1+co_time_1} ms")
    fig.canvas.draw_idle()

slider.on_changed(update)

# Figure je 133 ms
fig, ax = plt.subplots(figsize=(20, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Initial plot
current_idex = 0
APDP_db = APDP_db_all_2[current_idex]
peaks = all_peaks_all_2[current_idex]
line, = ax.plot(ms_final_2* 1e9, APDP_db_all_2[current_idex, :], label='APDP (dB)')
peaks_plot, = ax.plot(ms_final_2[peaks] , APDP_db_all_2[current_idex, peaks], 'rx', label='Peaks')

ax.set_xlabel("Delay Time (ns)")
ax.set_ylabel("APDP (dB)")
ax.set_title(f"APDP for Rounds {round_numbers} from {current_idex*co_time_2} ms to  {current_idex*co_time_2+co_time_2} ms")
ax.legend()
ax.grid(True)

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Millisecond', 0, len(APDP_db_all_2) -1, valinit=current_idex + 1, valfmt='%d.th 133 ms')

def update(val):
    index = int(slider.val)
    #print(f"Slider index: {index}") 
    APDP_db = APDP_db_all_2[index]
    peaks = all_peaks_all_2[index]
    #line.set_xdata([ms_final[index] * 1e9])
    line.set_ydata(APDP_db_all_2[index,:])
    #print (f"ydata: {APDP_db}")
    peaks_plot.set_xdata(ms_final_2[peaks] * 1e9)
    peaks_plot.set_ydata(APDP_db_all_2[index, peaks])
    ax.relim()
    ax.autoscale_view()
    ax.set_title(f"APDP for Rounds {round_numbers} from {index*co_time_2} ms to  {index*co_time_2+co_time_2} ms")
    fig.canvas.draw_idle()

slider.on_changed(update)
'''

'''# Create the first figure and slider
fig1, ax1 = plt.subplots(figsize=(20, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Initial plotting for the first dataset
current_index_1 = 0  # Start from the first index
APDP_db_1 = APDP_db_all_1[current_index_1]
peaks_1 = all_peaks_all_1[current_index_1]
line1, = ax1.plot(ms_final_1, APDP_db_1, label='APDP 1 (dB)')
peaks_plot1, = ax1.plot(ms_final_1[peaks_1], APDP_db_1[peaks_1], 'rx', label='Peaks 1')
ax1.set_xlabel("Delay Time (ns)")
ax1.set_ylabel("APDP 1 (dB)")
ax1.set_title(f"APDP 1 for Rounds {round_numbers} from {current_index_1 * co_time_1} ms to {current_index_1 * co_time_1 + co_time_1} ms")
ax1.legend()
ax1.grid(True)

# Slider for the first figure
ax_slider1 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider1 = Slider(ax_slider1, 'Milliseconds', 1, len(APDP_db_all_1), valinit=1, valfmt='%d.th 24 ms')

# Update function for the first figure
def update1(val):
    index = int(slider1.val) - 1  # Convert slider value to zero-based index
    APDP_db_1 = APDP_db_all_1[index]
    peaks_1 = all_peaks_all_1[index]
    line1.set_ydata(APDP_db_1)
    peaks_plot1.set_xdata(ms_final_1[peaks_1])
    peaks_plot1.set_ydata(APDP_db_1[peaks_1])
    ax1.relim()
    ax1.autoscale_view()
    ax1.set_title(f"APDP 1 for Rounds {round_numbers} from {index * co_time_1} ms to {index * co_time_1 + co_time_1} ms")
    fig1.canvas.draw_idle()  # Refresh the figure

slider1.on_changed(update1)

# Create the second figure and slider
fig2, ax2 = plt.subplots(figsize=(20, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Initial plotting for the second dataset
current_index_2 = 0  # Start from the first index
APDP_db_2 = APDP_db_all_2[current_index_2]
peaks_2 = all_peaks_all_2[current_index_2]
line2, = ax2.plot(ms_final_2, APDP_db_2, label='APDP 2 (dB)')
peaks_plot2, = ax2.plot(ms_final_2[peaks_2], APDP_db_2[peaks_2], 'rx', label='Peaks 2')
ax2.set_xlabel("Delay Time (ns)")
ax2.set_ylabel("APDP 2 (dB)")
ax2.set_title(f"APDP 2 for Rounds {round_numbers} from {current_index_2 * co_time_2} ms to {current_index_2 * co_time_2 + co_time_2} ms")
ax2.legend()
ax2.grid(True)

# Slider for the second figure
ax_slider2 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider2 = Slider(ax_slider2, 'Milliseconds', 1, len(APDP_db_all_2), valinit=1, valfmt='%d.th 133 ms')

# Update function for the second figure
def update2(val):
    index = int(slider2.val) - 1  # Convert slider value to zero-based index
    APDP_db_2 = APDP_db_all_2[index]
    peaks_2 = all_peaks_all_2[index]
    line2.set_ydata(APDP_db_2)
    peaks_plot2.set_xdata(ms_final_2[peaks_2])
    peaks_plot2.set_ydata(APDP_db_2[peaks_2])
    ax2.relim()
    ax2.autoscale_view()
    ax2.set_title(f"APDP 2 for Rounds {round_numbers} from {index * co_time_2} ms to {index * co_time_2 + co_time_2} ms")
    fig2.canvas.draw_idle()  # Refresh the figure

slider2.on_changed(update2)
'''