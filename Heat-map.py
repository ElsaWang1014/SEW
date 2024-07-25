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

cir_realizations = [np.array([value]) for value in correlation_Zeit_Millisecond_fertig.correlations]

def ensure_array(data):
    #jede Data als 1-dimensional
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
            
            # make sure 1-dimensional
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

# 检查数据长度和内容
print("Checking data lengths and contents:")
for i, realization in enumerate(cir_realizations):
    print(f"Realization {i}: {realization}")
    print(f"Length of Realization {i}: {len(realization)}")

# 标准化
# 只有在数据长度足够大的情况下才执行标准化
if any(len(realization) > 1 for realization in cir_realizations):
    cir_realizations_nor = [standardize(realization) for realization in cir_realizations]
else:
    cir_realizations_nor = cir_realizations

print("Standardized cir_realizations:")
for i, realization in enumerate(cir_realizations_nor):
    print(f"Realization {i}: {realization}")
    print(f"Shape of Realization {i}: {realization.shape}")

# 计算交叉相关性矩阵
cross_corr_matrix = compute_cross_correlation(cir_realizations_nor)
print("cross_corr_matrix:", cross_corr_matrix)


# Visualisierung

# Create figure and add axis
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111, projection='3d')

# 创建网格
X, Y = np.meshgrid(np.arange(cross_corr_matrix.shape[0]), np.arange(cross_corr_matrix.shape[1]))

# 绘制表面图
colors = [(0, 0, 1), (1, 0, 0)]  # 从蓝色到红色
n_bins = 100  # 将插值离散化为100个色阶
cmap_name = 'blue_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

surf = ax.plot_surface(X=X, Y=Y, Z=cross_corr_matrix, cmap=cm)

# 添加颜色条
fig.colorbar(surf, ax=ax, label='Normalized Cross-Correlation Coefficient')

# 设置标题和标签
plt.title('Cross-Correlation Matrix of CIRs')

#ax.set_xticks(np.arange(len(correlation_Zeit.rounds)))
#ax.set_yticks(np.arange(len(correlation_Zeit.rounds)))
#ax.set_xticklabels(correlation_Zeit.rounds)
#ax.set_yticklabels(correlation_Zeit.rounds)

# set the major ticks to every 50th label
#ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.invert_xaxis()
ax.set_xlabel('Millisecond')
ax.set_ylabel('Millisecond')

ax.set_zlabel('Normalized Cross-Correlation Coefficient')
ax.set_zlim(0.99,1)

plt.show()