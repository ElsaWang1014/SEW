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

#print(seconds)


#die Daten für bestimmte Round und Zeit herunterladen
#filename = load_path_1.format(round_number,second)
#mat = scipy.io.loadmat(filenames)
#data = mat["cirs"]
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
#cirs_data.append(data_db)
# durch jede Millisecond
#for ms in range(cirs_data.shape[1]): 
        #data_db[(second,ms)] = 10 * np.log10(np.abs(cirs_data[:, ms]))
       #for key, value in data.items():
       # data_db[key] = 10 * np.log10(np.abs(value))
        #print (f"data_db ist: {data_db}")
        #print (f"Zahle ist {len(data_db)}")

#Die erste 1ms
#num_samples = 1000
#data_first_millisecond = data_db[:,0]
#data_first_millisecond = data_db[0][:, 0]
#first_millisecond_key = (second[0],0)
#data_first_millisecond = data_db[first_millisecond_key]
first_second_filename = f"Round_{round_number}_AP_1_RF_0_Sec_{seconds[0]}.mat"
data_first_millisecond = data_db[first_second_filename][:, 0]

#Correlation
correlations = []
ms = 0
for second in seconds:
#for key in data_db.keys():  
    #if key != first_millisecond_key:
    #data_first_millisecond = data_db[f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"][:,ms]
      #data_current_millisecond = data_db[f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"][:,ms]
    data_current_millisecond = data_db[f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"][:, ms]
     
    normalized = np.sqrt(np.sum(data_first_millisecond**2)*np.sum(data_current_millisecond**2))

    #if normalized == 0:
       #correlation = np.correlate(data_first_microsecond,data_current_microsecond,mode='valid')[0]
   # else:
       #correlation = np.correlate(data_first_microsecond,data_current_microsecond,mode='valid')[0]/ normalized

    correlation = np.correlate(data_first_millisecond,data_current_millisecond,mode='valid')[0]/ normalized

    correlations.append(correlation)
    print(f"correlations ist:{correlations}")

# Figur
#plt.plot(data_first_millisecond)
#plt.show()
time = np.arange(1,len(data_db)+1)
xticks = np.arange(1,25,0.5)
yticks = np.arange(0.994,1,0.001)
plt.figure(figsize=(100, 50))
plt.plot(time, correlations,color='b')
plt.xlabel("Seconds [s]")
plt.xlim(1,25)
plt.xticks(xticks)
plt.ylabel("Correlation Coefficient")
#plt.ylim(0.994,1.001)
#plt.yticks(yticks)
plt.title(f"Correlation of 1st Millisecond with Sebsequent Milliseconds in Round {round_number}")
plt.grid(True)
plt.show()
