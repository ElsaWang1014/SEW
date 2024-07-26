import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

#Inforamtionen
#load_path = "/media/campus/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_Normal/Round_3_AP_1_RF_0_Sec_20.mat"
load_path_1 = "/media/campus/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_Normal/"
round_number = 1
#second = 10
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
       print(f"filename is：{filenames}")
       print(f"repr filename： {repr(filenames)}")



#die Daten für bestimmte Round und Zeit herunterladen

for second in seconds:
   
 filename = f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"
 full_filename = os.path.join(load_path_1,filename)
 if os.path.exists(full_filename):
        mat = scipy.io.loadmat(full_filename)
        cirs_data = mat["cirs"]
        
        # Durch jede Millisecond dB berechnen
        for ms in range(cirs_data.shape[1]):
            data_db[(second, ms)] = 10 * np.log10(np.abs(cirs_data[:, ms]))
 #data[filename] = cirs_data
 print(filename)
 #print(data)
print(f"Available keys in data_db:", data_db.keys())


#Die erste 1ms
first_second_filename = f"Round_{round_number}_AP_1_RF_0_Sec_{seconds[0]}.mat"
first_millisecond_key = (seconds[0], 0)
data_first_millisecond = data_db[first_millisecond_key]


#Correlation
correlations = []

for key in data_db.keys():  
   
    data_current_millisecond = data_db[key]
     
    normalized = np.sqrt(np.sum(data_first_millisecond**2)*np.sum(data_current_millisecond**2))

    #if normalized == 0:
       #correlation = np.correlate(data_first_microsecond,data_current_microsecond,mode='valid')[0]
   # else:
       #correlation = np.correlate(data_first_microsecond,data_current_microsecond,mode='valid')[0]/ normalized

    correlation = np.correlate(data_first_millisecond,data_current_millisecond,mode='valid')[0]/ normalized

    correlations.append(correlation)
    #print(f"correlations ist:{correlations}")

# Figur

time = np.arange(1,len(data_db)+1)
xticks = np.arange(1,25)
yticks = np.arange(0.999,1,0.0001)
plt.figure(figsize=(100, 55))
plt.plot(time, correlations,color='b')
plt.xlabel("Seconds [s]")
plt.xlim(1,25)
plt.xticks(xticks)
plt.ylabel("Correlation Coefficient")
plt.ylim(0.999,1.0001)
plt.yticks(yticks)
plt.title(f"Correlation of 1st Millisecond with Sebsequent Milliseconds in Round {round_number}")
plt.grid(True)
plt.show()
