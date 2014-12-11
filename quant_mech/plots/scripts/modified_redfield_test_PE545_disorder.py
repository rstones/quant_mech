import numpy as np
import matplotlib.pyplot as plt

data = np.load('../../data/modified_redfield_test_PE545_disorder_data.npz')
site_histories = data['site_histories']
time = data['time']
num_realisations = data['num_realisations']

site_history_average = np.zeros((site_histories.shape[1], site_histories.shape[2]))
for n in range(num_realisations):
    site_history_average += site_histories[n]
site_history_average /= num_realisations
 
for i,row in enumerate(site_history_average):
    plt.plot(time, row, label=str(i+1))
plt.legend()
plt.show()
