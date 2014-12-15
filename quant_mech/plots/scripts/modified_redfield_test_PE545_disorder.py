import numpy as np
import matplotlib.pyplot as plt

data = np.load('../../data/modified_redfield_test_PE545_disorder_data2.npz')
site_histories_sum = data['site_histories_sum']
time = data['time']
num_realisations = data['num_realisations']

site_history_average = site_histories_sum / num_realisations
 
for i,row in enumerate(site_history_average):
    plt.plot(time, row, label=str(i+1))
plt.legend()
plt.show()
