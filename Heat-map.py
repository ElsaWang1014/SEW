import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from scipy import stats
from numpy import percentile, zeros
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
print(data)
print(rounds)

# dB berechnen
for key, value in data.items():
    data_db[key] = 10 * np.log10(np.abs(value))
    

data_array = np.array(list(data_db.values()))
print(f'data_db',data_array)

num_realizations = len(data_array)
corr = np.zeros((num_realizations, num_realizations))


# 计算交叉相关矩阵
for i in range(num_realizations):
    for j in range(num_realizations):
        data_i = data_array[i]
        data_j = data_array[j]

        corr_coef = np.corrcoef(data_i, data_j)
        print(f'corr_coef',corr_coef)
        if i == j:
            corr[i,j] = corr_coef[0, 0]  # calculate correlation with itself
            print(f'corr coef with itself',corr)
        else:
           corr[i, j] = corr_coef[0, 1]
           corr[j, i] = corr_coef[1, 0]
    print (f'corr matrix', corr)
    
        


# Visualisierung

# Create figure and add axis
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111, projection='3d')

# grid
X, Y = np.meshgrid(range(num_realizations), range(num_realizations))
Z = corr
# colorbar
colors = [(0, 0, 1), (1, 0, 0)]  # blue to red
cmap_name = 'blue_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors)

surf = ax.plot_surface(X, Y, Z, cmap=cm,vmin=0, vmax=1)
#cbar = fig.colorbar(surf, ax=ax)
#cbar.set_clim(0, 1)

fig.colorbar(surf, ax=ax, label='Cross-Correlation Coefficient')

# title and ticks and label
plt.title('3D Heat-Map for Cross-Correlation Matrix of CIRs')


x_ticks = range(len(rounds))
x_tick_labels = rounds
y_ticks = range(len(rounds))
y_tick_labels = rounds
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xticklabels(x_tick_labels)
ax.set_yticklabels(y_tick_labels)


ax.invert_xaxis()
ax.set_xlabel('Rounds')
ax.set_ylabel('Rounds')

ax.set_zlabel('Cross-Correlation Coefficient')
ax.set_zlim(0,1)

plt.show()