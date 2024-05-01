import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os


curr_dir = os.getcwd()
data_file = os.path.join(curr_dir, 'all_data.txt')
f = np.loadtxt(data_file, usecols=7)

# plot true values' histogram
bins = np.linspace(0, 1.0, 100)
plt.hist(f, histtype = 'bar', bins = bins)
plt.xlabel('Alpha values')
plt.ylabel('Number of alpha values')
plt.legend()
plt.savefig('true_histogram.png')
plt.show()


