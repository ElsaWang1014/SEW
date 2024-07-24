import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import correlation_Zeit_Millisecond_fertig
import correlation_Zeit
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from scipy import stats
from numpy import percentile

 

# Example data

cir_realizations = correlation_Zeit_Millisecond_fertig.correlations 
print(correlation_Zeit.correlations )  
print(type(cir_realizations))  
print(cir_realizations)  
cir_realizations = np.array(cir_realizations)
print(type(cir_realizations))  
print(cir_realizations.shape) 

 

def compute_cross_correlation(data):

    num_realizations = len(data)
    cross_corr_matrix = np.zeros((num_realizations, num_realizations))
 

    for i in range(num_realizations):

        print(i)

        for j in range(i, num_realizations):

            print(j)
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


#min_val = np.min(cross_corr_matrix)
#max_val = np.max(cross_corr_matrix)
#normalized_matrix = (cross_corr_matrix - min_val) / (max_val - min_val)
#print(f"min= {min_val}, max = {max_val}")

# Compute cross-correlation matrix
#cross_corr_matrix = compute_cross_correlation(cir_realizations_500ms)

print("cross_corr_matrix:", cross_corr_matrix)
#print("Normalized matrix min and max values:", np.min(normalized_matrix), np.max(normalized_matrix))


colors = [(0, 0, 1), (1, 0, 0)]  # B -> R
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'blue_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Visualize the cross-correlation matrix

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111,projection='3d')


X, Y = np.meshgrid(np.arange(cross_corr_matrix.shape[0]), np.arange(cross_corr_matrix.shape[1]))
surf = ax.plot_surface(X, Y, cross_corr_matrix, cmap=cm)
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