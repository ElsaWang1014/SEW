import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
#print(rounds)

# dB berechnen
for key, value in data.items():
    data_db[key] = 10 * np.log10(np.abs(value))
    

data_array = np.array(list(data_db.values()))
#print(f'data_db shape',data_array.shape)

num_milliseconds = data_array.shape[2]

corr = np.zeros((len(round_numbers), len(round_numbers), num_milliseconds))


# cross correlation berechnen
for t in range(num_milliseconds):

    #print (f'Millisecond:', t)

    for i in range(len(round_numbers)):

        #print(f'Round number:', round_number)

        data_i = data_array[i, :, t]

        for j in range(i, len(round_numbers)):
          
          #print(f'Round number:', round_number)

          data_j = data_array[j, :, t]

          corr_coef = np.corrcoef(data_i, data_j)
          #print(f'corr_coef',corr_coef)

          if i == j:
                corr[i, j, t] = corr_coef[0, 0]  # calculate correlation with itself
          else:
                corr[i, j, t] = corr_coef[0, 1]
                corr[j, i, t] = corr_coef[1, 0]

    #print (f'corr matrix', corr)
    
        


# Visualisierung
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# colorbar
colors = [(0, 0, 1), (1, 0, 0)]  # blue to red
cmap_name = 'blue_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors)


# grid
X, Y = np.meshgrid(round_numbers, round_numbers)
for t in range(num_milliseconds):
    Z = np.full_like(X, t)  
    corr_t = corr[:, :, t]
    surf=ax.plot_surface(X, Y, Z, facecolors=cm(corr_t),vmin=0,vmax=1)


mappable = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=1))
cbar = plt.colorbar(mappable, ax=ax)
cbar.set_label('Cross-Correlation Coefficient', labelpad=10)


ax.set_title('4D Heatmap of Cross-Correlation Matrix of CIRs')
ax.set_xlabel('Round')
ax.set_ylabel('Round')
ax.set_zlabel('Milliseconds')
ax.invert_xaxis()



plt.show()