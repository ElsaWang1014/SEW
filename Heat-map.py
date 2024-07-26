import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import correlation_Zeit_Millisecond_fertig
import correlation_Zeit
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from scipy import stats
from numpy import percentile, zeros

 

# Example data

cir_realizations = [np.array([value]) for value in correlation_Zeit.correlations]

def ensure_array(data):
    """Stelle sicher, dass jeder Eintrag in der Liste ein eindimensionales numpy Array ist"""
    return [np.asarray(item).flatten() for item in data]

def standardize(data):
    # Standardisieren correlation, weil die differenz zu klein ist
    data = np.asarray(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data) 
    return (data - mean) / std


def compute_cross_correlation(data):

    num_realizations = len(data)
    cross_corr_matrix = np.zeros((num_realizations, num_realizations))

 

    for i in range(num_realizations):

        print(f"Processing realization {i}")
        
        for j in range(i, num_realizations):

            print(f"Comparing with realization {j}")
            
            corr = np.correlate(data[i], data[j], mode='full')
            print(f"correlation", corr)

            #norm_corr = corr / np.sqrt(np.correlate(data[i], data[i], mode='full')[len(data[i]) - 1] * np.correlate(data[j], data[j], mode='full')[len(data[j]) - 1])
            #print(f"nor_correlation", norm_corr)


            max_corr = np.max(corr)

            cross_corr_matrix[i, j] = max_corr

            cross_corr_matrix[j, i] = max_corr


    print("Shape of data[i]:", data[i].shape)
    print("Shape of data[j]:", data[j].shape)
    


    return cross_corr_matrix
 
 

cir_realizations = ensure_array(cir_realizations)


# cross correlation matrix
cross_corr_matrix = compute_cross_correlation(cir_realizations)
print("cross_corr_matrix:", cross_corr_matrix)


# Visualisierung

# Create figure and add axis
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111, projection='3d')

# grid
X, Y = np.meshgrid(np.arange(cross_corr_matrix.shape[0]), np.arange(cross_corr_matrix.shape[1]))

# colorbar
colors = [(0, 0, 1), (1, 0, 0)]  # blue to red
cmap_name = 'blue_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors)

surf = ax.plot_surface(X=X, Y=Y, Z=cross_corr_matrix, cmap=cm,vmin=0, vmax=1)
cbar = fig.colorbar(surf, ax=ax)
cbar.set_clim(0, 1)

fig.colorbar(surf, ax=ax, label='Cross-Correlation Coefficient')

# title and ticks and label
plt.title('3D Heat-Map for Cross-Correlation Matrix of CIRs')


x_ticks = range(len(correlation_Zeit.rounds))
x_tick_labels = correlation_Zeit.rounds
y_ticks = range(len(correlation_Zeit.rounds))
y_tick_labels = correlation_Zeit.rounds
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xticklabels(x_tick_labels)
ax.set_yticklabels(y_tick_labels)


ax.invert_xaxis()
ax.set_xlabel('Rounds')
ax.set_ylabel('Rounds')

ax.set_zlabel('Cross-Correlation Coefficient')
ax.set_zlim(0.99,1)

plt.show()