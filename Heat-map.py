
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import correlation_Zeit_Millisecond_fertig
import correlation_Zeit
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker


 

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
min_corr = np.min(cross_corr_matrix)
max_corr = np.max(cross_corr_matrix)
normalized_matrix = (cross_corr_matrix - min_corr) / (max_corr - min_corr)
#rank_matrix = cross_corr_matrix.argsort(axis=1).argsort(axis=1)
#normalized_matrix = rank_matrix / (cross_corr_matrix.shape[1] - 1)


colors = [(0, 0, 1), (1, 0, 0)]  # B -> R
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'blue_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Visualize the cross-correlation matrix

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111,projection='3d')


#X, Y = np.meshgrid(np.arange(enhanced_matrix.shape[0]), np.arange(enhanced_matrix.shape[1]))

#surf = ax.plot_surface(X, Y, enhanced_matrix, cmap=cm)

X, Y = np.meshgrid(np.arange(normalized_matrix.shape[0]), np.arange(normalized_matrix.shape[1]))
surf = ax.plot_surface(X, Y, normalized_matrix, cmap=cm)
#plt.imshow(cross_corr_matrix, cmap='hot', interpolation='nearest')

fig.colorbar(surf, ax=ax, label='Normalized Cross-Correlation Coefficient')

plt.title('Cross-Correlation Matrix of CIRs')


#rounds = list(correlation_Zeit.rounds)
#ax.set_xticks(np.arange(len(rounds)))
#ax.set_yticks(np.arange(len(rounds)))
#ax.set_xticklabels(rounds)
#ax.set_yticklabels(rounds)

# set the major ticks to every 50th label
#ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

ax.set_xlabel('Round')
ax.set_ylabel('Round')

ax.set_zlabel('Normalized Cross-Correlation Coefficient')
ax.set_zlim(0,1)

plt.show()