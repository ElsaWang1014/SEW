
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import correlation_Zeit
 

# Example data

cir_realizations = correlation_Zeit.correlations # Dict mit round und cirs für alle Sekunden/
print(type(cir_realizations))  
print(cir_realizations)  
cir_realizations = np.array(cir_realizations)
print(type(cir_realizations))  
print(cir_realizations.shape) 

 

def compute_cross_correlation(data):

    num_realizations = data.shape[0]

    cross_corr_matrix = np.zeros((num_realizations, num_realizations))

 

    for i in range(num_realizations):

        print(i)

        for j in range(i, num_realizations):

            print(j)

            #flattened_i = data[i].flatten()
            #flattened_j = data[j].flatten()

            # Compute cross-correlation
            corr = np.correlate(np.array(data[i], ndmin=1), np.array(data[j], ndmin=1), mode='full')
            norm_corr = corr / np.sqrt(np.correlate(np.array(data[i], ndmin=1), np.array(data[i], ndmin=1), mode='full')[-1] * np.correlate(np.array(data[j], ndmin=1), np.array(data[j], ndmin=1), mode='full')[-1])
            max_corr = np.max(norm_corr)
            
            cross_corr_matrix[i, j] = max_corr

            cross_corr_matrix[j, i] = max_corr

 

    return cross_corr_matrix

 

# Compute cross-correlation matrix

cross_corr_matrix = compute_cross_correlation(cir_realizations)

 

# Visualize the cross-correlation matrix

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111,projection='3d')

X,Y = np.meshgrid(np.arange(cir_realizations.shape[0]),np.arange(cir_realizations.shape[0]))

surf = ax.plot_surface(X, Y, cross_corr_matrix, cmap='hot')
#plt.imshow(cross_corr_matrix, cmap='hot', interpolation='nearest')

fig.colorbar(surf, ax=ax, label='Cross-Correlation Coefficient')

plt.title('Cross-Correlation Matrix of CIRs')

ax.set_xlabel('Realization Index')
ax.set_ylabel('Realization Index')
ax.set_zlabel('Realization Index')
ax.set_zlim(0.995,1)

plt.show()