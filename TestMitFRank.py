import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.signal import find_peaks
from matplotlib.widgets import Slider


Nt1 = -114

Nt1_linear = 10**(-114/10)

Nt20_linear = Nt1_linear*100

Nt20 = 10*np.log10(Nt20_linear)

print(Nt20)