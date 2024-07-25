<<<<<<< HEAD
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
=======

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
<<<<<<< HEAD
>>>>>>> f24c6cc052b0cd7233956b3a27714149a1c81c4d
import correlation_Zeit_Millisecond_fertig
import correlation_Zeit
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
<<<<<<< HEAD
from scipy import stats
from numpy import percentile

=======


=======
import correlation_Zeit
>>>>>>> origin/neu
>>>>>>> f24c6cc052b0cd7233956b3a27714149a1c81c4d
 

# Example data

<<<<<<< HEAD
cir_realizations = correlation_Zeit_Millisecond_fertig.correlations 
print(correlation_Zeit.correlations )  
=======
cir_realizations = correlation_Zeit.correlations # Dict mit round und cirs für alle Sekunden/
>>>>>>> f24c6cc052b0cd7233956b3a27714149a1c81c4d
print(type(cir_realizations))  
print(cir_realizations)  
cir_realizations = np.array(cir_realizations)
print(type(cir_realizations))  
print(cir_realizations.shape) 

 

def compute_cross_correlation(data):

<<<<<<< HEAD
    num_realizations = len(data)
    cross_corr_matrix = np.zeros((num_realizations, num_realizations))
=======
    num_realizations = data.shape[0]

    cross_corr_matrix = np.zeros((num_realizations, num_realizations))

>>>>>>> f24c6cc052b0cd7233956b3a27714149a1c81c4d
 

    for i in range(num_realizations):

        print(i)

        for j in range(i, num_realizations):

            print(j)
<<<<<<< HEAD
            if i == j:
                corr = 1.0
            else:
                corr = np.corrcoef(data, data)[0, 1]
                for k in range(1, num_realizations - i):
                 corr += np.corrcoef(data[:-k], data[k:])[0, 1]
                corr /= num_realizations - i
            cross_corr_matrix[i, j] = corr
            cross_corr_matrix[j, i] = corr

 
    return cross_corr_matrix



# Compute cross-correlation matrix
# Select every 500th millisecond from cir_realizations
#cir_realizations_500ms = cir_realizations[::100]
cross_corr_matrix = compute_cross_correlation(cir_realizations)

# Compute cross-correlation matrix
#cross_corr_matrix = compute_cross_correlation(cir_realizations_500ms)

print("cross_corr_matrix:", cross_corr_matrix)
#print("Normalized matrix min and max values:", np.min(normalized_matrix), np.max(normalized_matrix))
=======

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
<<<<<<< HEAD
min_corr = np.min(cross_corr_matrix)
max_corr = np.max(cross_corr_matrix)
normalized_matrix = (cross_corr_matrix - min_corr) / (max_corr - min_corr)
#rank_matrix = cross_corr_matrix.argsort(axis=1).argsort(axis=1)
#normalized_matrix = rank_matrix / (cross_corr_matrix.shape[1] - 1)
>>>>>>> f24c6cc052b0cd7233956b3a27714149a1c81c4d


colors = [(0, 0, 1), (1, 0, 0)]  # B -> R
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'blue_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
<<<<<<< HEAD
=======
=======

 
>>>>>>> origin/neu
>>>>>>> f24c6cc052b0cd7233956b3a27714149a1c81c4d

# Visualize the cross-correlation matrix

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111,projection='3d')

<<<<<<< HEAD

X, Y = np.meshgrid(np.arange(cross_corr_matrix.shape[0]), np.arange(cross_corr_matrix.shape[1]))
surf = ax.plot_surface(X, Y, cross_corr_matrix, cmap=cm)
=======
<<<<<<< HEAD

#X, Y = np.meshgrid(np.arange(enhanced_matrix.shape[0]), np.arange(enhanced_matrix.shape[1]))

#surf = ax.plot_surface(X, Y, enhanced_matrix, cmap=cm)

X, Y = np.meshgrid(np.arange(normalized_matrix.shape[0]), np.arange(normalized_matrix.shape[1]))
surf = ax.plot_surface(X, Y, normalized_matrix, cmap=cm)
>>>>>>> f24c6cc052b0cd7233956b3a27714149a1c81c4d
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
<<<<<<< HEAD
=======
=======
X,Y = np.meshgrid(np.arange(cir_realizations.shape[0]),np.arange(cir_realizations.shape[0]))

surf = ax.plot_surface(X, Y, cross_corr_matrix, cmap='hot')
#plt.imshow(cross_corr_matrix, cmap='hot', interpolation='nearest')

fig.colorbar(surf, ax=ax, label='Cross-Correlation Coefficient')

plt.title('Cross-Correlation Matrix of CIRs')

ax.set_xlabel('Realization Index')
ax.set_ylabel('Realization Index')
ax.set_zlabel('Realization Index')
ax.set_zlim(0.995,1)
>>>>>>> origin/neu
>>>>>>> f24c6cc052b0cd7233956b3a27714149a1c81c4d

plt.show()