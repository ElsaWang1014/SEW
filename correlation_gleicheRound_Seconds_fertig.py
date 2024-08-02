import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

#Inforamtionen
#load_path = "/media/campus/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_Normal/Round_3_AP_1_RF_0_Sec_20.mat"
load_path_1 = "/media/campus/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_Normal/"
round_number = 1
cirs_data = []
seconds = []
data = {}
data_db = {}

#wie viele Rounds in diesem Ordner
for filenames in os.listdir(load_path_1):
    if filenames.startswith(f"Round_{round_number}_AP_1_RF_0_Sec_") and filenames.endswith(".mat"):
       r = int (filenames.split("_")[7].replace(".mat",""))
       seconds.append(r)

       #ordnen wie 1,2,3,...
       seconds.sort()
       print (f"second ist {seconds}")

#print(seconds)


#die Daten für bestimmte Round und Zeit herunterladen

for second in seconds:
   
 filename = f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"
 full_filename = os.path.join(load_path_1,filename)
 mat = scipy.io.loadmat(full_filename)
 cirs_data = mat["cirs"]
 
 data[filename] = cirs_data
 #print(filename)
 #print(data)



# dB berechnen
for key, value in data.items():
 data_db [key]= 10 * np.log10(np.abs(value))


#Die erste 1ms

first_second_filename = f"Round_{round_number}_AP_1_RF_0_Sec_{seconds[0]}.mat"
data_first_millisecond = data_db[first_second_filename][:, 0]

#Correlation
correlations = []
ms = 0
for second in seconds:

    data_current_millisecond = data_db[f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"][:, ms]
     
    
    correlation = np.corrcoef(data_first_millisecond,data_current_millisecond)
    #print(correlation[0][1])
    correlations.append(correlation[0][1])

    #correlations.append(correlation)
    print(f"correlations ist:{correlations}")

# Figur
#plt.plot(data_first_millisecond)
#plt.show()
time = np.arange(1,len(data_db)+1)
max_seconds = max(seconds)
xticks = np.arange(1,max_seconds)
yticks = np.arange(0,1,0.1)
plt.figure(figsize=(100, 50))
plt.plot(time, correlations,color='b')
plt.xlabel("Seconds [s]")
plt.xlim(1,max_seconds)
plt.xticks(xticks)
plt.ylabel("Correlation Coefficient")
#plt.ylim(0,1)
#plt.yticks(yticks)
plt.title(f"Correlation of 1st Millisecond with Sebsequent Milliseconds in Round {round_number}")
plt.grid(True)
plt.show()
