import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

#Inforamtionen
load_path = "/media/campus/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_AGVHorizontal/"
rounds = set()
round_numbers = [77,78,79,80,81,82]
second = 2
data = {}
data_db = {}


#die Daten für bestimmte Round und Zeit herunterladen
for round_number in round_numbers:
   
 filename = f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"
 full_filename = os.path.join(load_path,filename)
 mat = scipy.io.loadmat(full_filename)
 cirs_data = mat["cirs"]
 data[filename] = cirs_data
 #print(f"Attempting to load file: {full_filename}")

#wie viele Rounds in diesem Ordner
for filename in os.listdir(load_path):
   r = int (filename.split("_")[1])
   rounds.add(r)
rounds = list(rounds)
rounds.sort()
#print(data)
print(rounds)

# dB berechnen
for key, value in data.items():
    data_db[key] = 10 * np.log10(np.abs(value))



#Correlation
correlations = []
cnt = 0
ms = 500
for round_number in round_numbers:  
    if cnt ==0:
      data_first_round = data_db[f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"][:,ms]
      cnt =1
    data_current_round = data_db[f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"][:,ms]

  
    correlation = np.corrcoef(data_first_round,data_current_round)
  
    correlations.append(correlation[0][1])
    
    print(round_number,correlations)
    print(round_number,correlation)

#plt.plot(data_first_round)
#plt.show()

# Figur
round = np.arange(1,len(round_numbers)+1)
plt.figure(figsize=(10, 6))
plt.plot(round_numbers, correlations,color='b',marker='o', linestyle='-', linewidth=2)
plt.xlabel("Rounds")
plt.xticks(rounds)
plt.ylabel("Correlation Coefficient")
#plt.ylim([0,1])
plt.title(f"Correlation of 1st Round with Rounds in Second {second}")
plt.grid(True)
plt.show()
