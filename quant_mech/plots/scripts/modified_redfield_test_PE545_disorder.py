import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils

# load disorder data
data = np.load('../../data/modified_redfield_test_PE545_disorder_data5.npz')
site_histories_sum = data['site_histories_sum']
time = data['time']
num_realisations = data['num_realisations']
site_thermal_states = data['site_thermal_states']

site_history_average = site_histories_sum / num_realisations
site_history_average /= np.sum(site_history_average[:,0]) # normalise to compare with thermal state

thermal_state_sum = np.sum(site_thermal_states, axis=1)
thermal_state_average = thermal_state_sum / num_realisations

'''
Calculates thermal state of average Hamiltonian
'''
# # basis { PEB_50/61C, DBV_A, DVB_B, PEB_82C, PEB_158C, PEB_50/61D, PEB_82D, PEB_158D }
# site_energies = np.array([18532., 18008., 17973., 18040., 18711., 19574., 19050., 18960.])
# 
# couplings = np.array([[0, 1., -37., 37., 23., 92., -16., 12.],
#                       [0, 0, 4., -11., 33., -39., -46., 3.],
#                       [0, 0, 0, 45., 3., 2., -11., 34.],
#                       [0, 0, 0, 0, -7., -17., -3., 6.],
#                       [0, 0, 0, 0, 0, 18., 7., 6.],
#                       [0, 0, 0, 0, 0, 0, 40., 26.],
#                       [0, 0, 0, 0, 0, 0, 0, 7.],
#                       [0, 0, 0, 0, 0, 0, 0, 0]])
# 
# site_hamiltonian = site_energies + couplings + couplings.T
# 
# evals, evecs = utils.sorted_eig(site_hamiltonian)
# exciton_hamiltonian = np.diag(evals)
# evecs = evecs.T
# temperature = 77.
# beta = 1. / (utils.KELVIN_TO_WAVENUMS * temperature)
# exciton_thermal_state = np.exp(-beta * evals) / np.sum([np.exp(-beta*E) for E in evals])
# site_thermal_state = np.diag(np.dot(evecs, np.dot(np.diag(exciton_thermal_state), evecs.T)))

 
for i,row in enumerate(site_history_average):
    plt.plot(time, row, label=str(i+1))
    plt.axhline(thermal_state_average[i], ls='--', label=str(i+1))
#plt.xlim(0,5)
plt.legend()
plt.show()
