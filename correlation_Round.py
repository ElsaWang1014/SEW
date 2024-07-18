import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

#Inforamtionen
load_path = "/media/campus/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_Normal/Round_3_AP_1_RF_0_Sec_20.mat"
round_number = 1
second = 10
cirs_data = []


#die Daten für bestimmte Round und Zeit herunterladen
filename = load_path.format(round_number,second)
mat = scipy.io.loadmat(filename)
data = mat["cirs"]



# dB berechnen
data_db = 10 * np.log10(np.abs(data))
cirs_data.append(data_db)

#Die erste 1000ms
num_samples = 1000
#data_first_millisecond = data_db[:,0]


#Correlation
correlations = []
cnt = 0

for i in range(1,num_samples):  
   if cnt ==0:
      data_first_millisecond = data_db[:,0]
      cnt =1
    #data_current_round = data_db[f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"][:,ms]
   
   data_current_microsecond = data_db[:,i]

   normalized = np.sqrt(np.sum(data_first_millisecond**2)*np.sum(data_current_microsecond**2))

    #if normalized == 0:
       #correlation = np.correlate(data_first_microsecond,data_current_microsecond,mode='valid')[0]
   # else:
       #correlation = np.correlate(data_first_microsecond,data_current_microsecond,mode='valid')[0]/ normalized

   correlation = np.correlate(data_first_millisecond,data_current_microsecond,mode='valid')/ normalized

   correlations.append(correlation)

# Figur
time = np.arange(1,num_samples)
xticks = np.arange(0,1050,50)
yticks = np.arange(0.994,1,0.0005)
plt.figure(figsize=(12, 6))
plt.plot(time, correlations,color='b')
plt.xlabel("Milliseconds [ms]")
plt.xlim(0,1000)
plt.xticks(xticks)
plt.ylabel("Correlation Coefficient")
plt.ylim(0.994,1)
plt.yticks(yticks)
plt.title(f"Correlation of 1st Millisecond with Sebsequent Milliseconds in Round {round_number}")
plt.grid(True)
plt.show()

