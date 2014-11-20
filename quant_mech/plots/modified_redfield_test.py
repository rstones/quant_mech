'''
Script to plot data generated from modified_redfield_test module
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils

data = np.load('../data/modified_redfield_test_simps_data.npz')
rates = data['rates']
delta_E_values = data['delta_E_values']
coupling_values = data['coupling_values']

for i,V in enumerate(coupling_values):
    plt.subplot(1,3,i+1)
    plt.loglog(delta_E_values, utils.WAVENUMS_TO_INVERSE_PS*rates[i], label='V = ' + str(V))
    plt.xlim(5,1000)
    plt.ylim(0.01,200)
    plt.legend()

plt.show()