import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats as stats
import os

# Informationen
load_path = "/media/campus/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_AGVHorizontal/"
round_numbers = [77]
second = 2

# die Daten für bestimmte Round und Zeit herunterladen
data_db = []
for round_number in round_numbers:
    filename = f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"
    full_filename = os.path.join(load_path, filename)
    if os.path.exists(full_filename):
      mat = scipy.io.loadmat(full_filename)
      cirs_data = mat["cirs"]
      data_db.append(10 * np.log10(np.abs(cirs_data)** 2))
    else:
       print(f"File {filename} not found.")
data_db = np.array(data_db) 
num_samples = data_db.shape[0]

#one dimension
data_flat = data_db.flatten()



# Normal Distribution
plt.figure(figsize=(10, 6))
mu, std = stats.norm.fit(data_flat)
x = np.linspace(min(data_flat), max(data_flat), 1000)
plt.plot(x, stats.norm.pdf(x, mu, std), label='Normal Distribution')
plt.hist(data_flat, bins=50, density=True, alpha=0.6, color='g', label='Data Histogram')
plt.title('Normal Distribution')
plt.xlabel('Data Value in dB')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# Gamma Distribution
plt.figure(figsize=(10, 6))
shape, loc, scale = stats.gamma.fit(data_flat)
x = np.linspace(min(data_flat), max(data_flat), 1000)
plt.plot(x, stats.gamma.pdf(x, shape, loc, scale), label='Gamma Distribution')
plt.hist(data_flat, bins=50, density=True, alpha=0.6, color='g', label='Data Histogram')
plt.title('Gamma Distribution')
plt.xlabel('Data Value in dB')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# Log-Normal Distribution
plt.figure(figsize=(10, 6))
abs_data = np.abs(data_flat)
shape, loc, scale = stats.lognorm.fit(abs_data, floc=0)
x = np.linspace(min(abs_data), max(abs_data), 1000)
plt.plot(x, stats.lognorm.pdf(x, shape, loc, scale), label='Log-Normal Distribution')
plt.hist(abs_data, bins=50, density=True, alpha=0.6, color='g', label='Data Histogram')
plt.title('Log-Normal Distribution')
plt.xlabel('Data Value in dB')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# Rayleigh Distribution
plt.figure(figsize=(10, 6))
abs_data = np.abs(data_flat)
scale = stats.rayleigh.fit(abs_data, floc=0)[0]
x = np.linspace(min(abs_data), max(abs_data), 1000)
plt.plot(x, stats.rayleigh.pdf(x, scale=scale), label='Rayleigh Distribution')
plt.hist(abs_data, bins=50, density=True, alpha=0.6, color='g', label='Data Histogram')
plt.title('Rayleigh Distribution')
plt.xlabel('Data Value in dB')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# Poisson Distribution
plt.figure(figsize=(10, 6))
data_poisson = np.round(data_flat).astype(int)
mu = np.mean(data_poisson)
x = np.arange(min(data_poisson), max(data_poisson) + 1)
plt.bar(x, stats.poisson.pmf(x, mu), alpha=0.75, label='Poisson Distribution')
plt.hist(data_poisson, bins=50, density=True, alpha=0.6, color='g', label='Data Histogram')
plt.title('Poisson Distribution')
plt.xlabel('Data Value in dB')
plt.ylabel('Probability Mass')
plt.legend()
plt.grid(True)
plt.show()

# Kafa Distribution
plt.figure(figsize=(10, 6))

plt.title('kafa Distribution')
plt.xlabel('Data Value in dB')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# Laplace Distribution
plt.figure(figsize=(10, 6))
loc, scale = stats.laplace.fit(data_flat)
x = np.linspace(min(data_flat), max(data_flat), 1000)
plt.plot(x, stats.laplace.pdf(x, loc, scale), label='Laplace Distribution')
plt.hist(data_flat, bins=50, density=True, alpha=0.6, color='g', label='Data Histogram')
plt.title('Laplace Distribution')
plt.xlabel('Data Value in dB')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()