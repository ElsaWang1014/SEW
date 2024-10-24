import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import math


rsp_position = []
delta_time = []
delta_dis = []
all_speed = []
i=0
delta_x = 0


rsp_position_4m_2_0_v = np.load('rsp_position_4m_RX2_RF0_vertikal.npy',allow_pickle=True)  
#print(rsp_position_4m_2_0_v.shape[0])

correlation_means_2_0_v = np.load('correlation_mean_position_1mm_RX2_RF0_vertikal.npy')
correlation_means_2_0_h = np.load('correlation_mean_position_1mm_RX2_RF0_horizontal.npy')
correlation_means_2_1_v = np.load('correlation_mean_position_1mm_RX2_RF1_vertikal.npy')
correlation_means_2_1_h = np.load('correlation_mean_position_1mm_RX2_RF1_horizontal.npy')
correlation_means_3_0_v = np.load('correlation_mean_position_1mm_RX3_RF0_vertikal.npy')
correlation_means_3_0_h = np.load('correlation_mean_position_1mm_RX3_RF0_horizontal.npy')
correlation_means_3_1_v = np.load('correlation_mean_position_1mm_RX3_RF1_vertikal.npy')
correlation_means_3_1_h = np.load('correlation_mean_position_1mm_RX3_RF1_horizontal.npy')

num_k_pos = rsp_position_4m_2_0_v.shape[0] - 1
 
# Figure 
plt.subplots(figsize=(60, 10))
x_values = np.arange(0, num_k_pos + 1) 
plt.plot(x_values,correlation_means_2_0_v,marker='o',label = 'RX2_RF0 at time point')
plt.plot(x_values,correlation_means_2_1_v,marker='o',label = 'RX2_RF1 at time point')
plt.plot(x_values,correlation_means_3_0_v,marker='o',label = 'RX3_RF0 at time point')
plt.plot(x_values,correlation_means_3_1_v,marker='o',label = 'RX3_RF1 at time point')
plt.title('Mean Correlation Over Position  ---  vertikal')
plt.xlabel('Distance in mm')
#plt.xticks(np.arange(0,max(x_values)+100,100)) 
plt.axhline(y=0.5, color ='purple', linestyle='--')
plt.grid(True)
plt.legend()

plt.subplots(figsize=(60, 10))
x_values = np.arange(0, num_k_pos + 1) 
plt.plot(x_values,correlation_means_2_0_h, marker='o',label = 'RX2_RF0 at time point')
plt.plot(x_values,correlation_means_2_1_h, marker='o',label = 'RX2_RF1 at time point')
plt.plot(x_values,correlation_means_3_0_h, marker='o',label = 'RX3_RF0 at time point')
plt.plot(x_values,correlation_means_3_1_h, marker='o',label = 'RX3_RF1 at time point')
plt.title('Mean Correlation Over Position  ---  horizontal')
plt.xlabel('Distance in mm')
#plt.xticks(np.arange(0,max(x_values)+100,100)) 
plt.axhline(y=0.5, color ='purple', linestyle='--')
plt.grid(True)
plt.legend()

plt.show()





###########################################################     data-process   ###########################################################
'''data = np.load('data_for_BesondereMessungen_RX2_RF0_vertikal.npy')
position = np.load('position_scenario3.npy', allow_pickle=True)
time1, x1, y1 = position[0]['Time (s)'], position[0]['X (m)'], position[0]['Y (m)']   
for j in range(len(position)):
    
        
        time2, x2, y2 = position[j]['Time (s)'], position[j]['X (m)'], position[j]['Y (m)']
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
        if distance >= 4 :
            delta_time.append(time2)
           
            print (f"Time : {time2} seconds")

            rsp_position.append(data[:,:j])
            print(rsp_position)
            
            rsp_filename = f'data_4m_RX2_RF0_vertikal.npy'
            np.save (rsp_filename,rsp_position)
            break


'''

'''print(data.shape)



position = []
time_indices = data.shape[2]
print(data.shape[2])
for time_index in range(time_indices):
    t = time_index / 1000
    if t>=0 and t<0.6:
                    alpha = 270 + 90 - (160 / (2 * np.pi * 72)) * 90 + (180 / (np.pi * 0.72)) * 0.5 * t ** 2
                    x = -3.35 - 0.72 + 0.72 * np.cos(np.deg2rad(alpha))
                    y = 11.26 + 0.72 * np.sin(np.deg2rad(alpha))
    elif t>=0.6 and t< (0.6+0.22/0.6):
                    alpha = 270 + 90 - (160 / (2 * np.pi * 72)) * 90 + (180 / (np.pi * 0.72)) * 0.5 * 0.6 ** 2 + (180 / (np.pi * 0.72)) * 0.6 * (t - 0.6)
                    x = -3.35 - 0.72 + 0.72 * np.cos(np.deg2rad(alpha))
                    y = 11.26 + 0.72 * np.sin(np.deg2rad(alpha))
    elif t>=0.6+0.22/0.6 and t<=20:
                    alpha = 0
                    x = -3.35
                    y = 11.26 + 0.6 * (t - (0.6+0.22/0.6))

    position.append({
                'Time (s)': time_index / 1000,
                'X (m)': x,
                'Y (m)': y
            })
print (position)
filename = f'position_scenario3_4m.npy'
np.save (filename,position)'''


###########################################################     Correlation-process   ###########################################################
'''
data = np.load('data_4m_RX2_RF1_horizontal.npy')
position = np.load('position_scenario3_4m.npy',allow_pickle=True)


while i < len(position)-1:  
    delta_x = 0
    
    for j in range(i + 1, len(position)):
    
        time1, x1, y1 = position[i]['Time (s)'], position[i]['X (m)'], position[i]['Y (m)']
        time2, x2, y2 = position[j]['Time (s)'], position[j]['X (m)'], position[j]['Y (m)']
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        #print(f"Distance between time {time1}s and {time2}s: {distance} meters")
        #time_diff = time2 - time1
        #speed = distance / time_diff
        #all_speed.append(speed)
        #print(f"Speed between time {time1}s and {time2}s: {speed}")
    
        if distance >= 0.001 :
            delta_time.append(time2)
            i = j
            #print (f"Time : {time2} seconds")
            break
    if j == len(position) -1 :
        i = len(position)
        break 
    

for time in delta_time:
    rsp_position.append(data[:,:,int(time*1000)])
rsp_position =  np.array(rsp_position)
#print(rsp_position)
#print(f"shape:{rsp_position.shape}")       
rsp_filename = f'rsp_position_4m_RX2_RF1_horizontal.npy'
np.save (rsp_filename,rsp_position)
'''
'''
#Correlation
#print (f"number k :{num_k_pos}")

rsp_position  = np.load('rsp_position_4m_RX3_RF1_vertikal.npy',allow_pickle=True)

correlations = []
for k  in range(rsp_position.shape[0]):
        temp_correlations = []
        for i in range (rsp_position.shape[0]- k ):
            
                corr = np.corrcoef(rsp_position[i,:],rsp_position[i + k  ,:])
                temp_correlations.append(corr[0][1])
            #print(f"k:{k},i,{i},temp correlation : {temp_correlations}")
        correlations.append(np.mean(temp_correlations))
        #print(correlations)
correlation_means = correlations
#print(correlation_means)
corr_filename = f'correlation_mean_position_1mm_RX3_RF1_vertikal.npy'
np.save (corr_filename,correlation_means)
'''