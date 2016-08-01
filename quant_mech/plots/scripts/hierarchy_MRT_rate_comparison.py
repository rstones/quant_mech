import numpy as np
import matplotlib.pyplot as plt

data = np.load('../../data/hierarchy_MRT_comparison_rate_data.npz')
MRT_rates = data['MRT_rates']
HEOM_rates = data['HEOM_rates']
energy_gap_values = data['energy_gap_values']

plt.loglog(energy_gap_values, MRT_rates, label='MRT')
plt.loglog(energy_gap_values, HEOM_rates, label='HEOM')
plt.legend().draggable()
plt.show()