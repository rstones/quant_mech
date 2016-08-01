'''
Created on 16 Dec 2014

@author: rstones
'''
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import quant_mech.open_systems as os
import quant_mech.utils as utils
import quant_mech.time_utils as te
from datetime import datetime

time_log = False

# basis { PEB_50/61C, DBV_A, DVB_B, PEB_82C, PEB_158C, PEB_50/61D, PEB_82D, PEB_158D }
average_site_energies = np.array([18532., 18008., 17973., 18040., 18711., 19574., 19050., 18960.])

couplings = np.array([[0, 1., -37., 37., 23., 92., -16., 12.],
                      [0, 0, 4., -11., 33., -39., -46., 3.],
                      [0, 0, 0, 45., 3., 2., -11., 34.],
                      [0, 0, 0, 0, -7., -17., -3., 6.],
                      [0, 0, 0, 0, 0, 18., 7., 6.],
                      [0, 0, 0, 0, 0, 0, 40., 26.],
                      [0, 0, 0, 0, 0, 0, 0, 7.],
                      [0, 0, 0, 0, 0, 0, 0, 0]])

average_site_hamiltonian = np.diag(average_site_energies) + couplings + couplings.T

temperature = 77.
reorg_energy1 = 40.
reorg_energy2 = 70.
cutoff_freq1 = 30.
cutoff_freq2 = 90.
mode_damping = 20.

def PE545_mode_params(damping):
    return np.array([(207., 0.0013, damping),
                     (244., 0.0072, damping),
                     (312., 0.0450, damping),
                     (372., 0.0578, damping),
                     (438., 0.0450, damping),
                     (514., 0.0924, damping),
                     (718., 0.0761, damping),
                     (813., 0.0578, damping),
                     (938., 0.0313, damping),
                     (1111., 0.0578, damping),
                     (1450., 0.1013, damping),
                     (1520., 0.0265, damping),
                     (1790., 0.0072, damping),
                     (2090., 0.0113, damping)])

data = io.loadmat('../../data/data_for_Richard.mat')

site_shifts = data['site_shift']
exciton_init_states = data['D0_save_ex']
num_realisations = site_shifts.shape[0]

time_interval = 5 #10
integration_time = np.linspace(0, time_interval, time_interval*32000) # 16000
num_expansion_terms = 40
mode_params = PE545_mode_params(mode_damping)
coeffs = os.lbf_coeffs(reorg_energy1, cutoff_freq1, temperature, mode_params, num_expansion_terms)
coeffs = np.concatenate((coeffs, os.lbf_coeffs(reorg_energy2, cutoff_freq2, temperature, None, num_expansion_terms)))
g_site = os.site_lbf_ed(integration_time, coeffs)
g_site_dot = os.site_lbf_dot_ed(integration_time, coeffs)
g_site_dot_dot = os.site_lbf_dot_dot_ed(integration_time, coeffs)
total_site_reorg_energy = reorg_energy1 + reorg_energy2 + np.sum([mode[0]*mode[1] for mode in mode_params])
# parameters for time evolution
duration = 5.
timestep = 0.01
time = np.arange(0, duration+timestep, timestep)
#init_dv = np.array([0.35, 0.12, 0.1, 0.1, 0.34, 0.61, 0.46, 0.5]) # init state in site basis

site_history_sum = np.zeros((average_site_hamiltonian.shape[0], time.size))

for n in range(num_realisations):
    print 'Calculating realisation ' + str(n+1) + ' at time ' +str(datetime.now().time())
    
    # calculate Hamiltonian for this realisation ie. add site_shifts to average site energies and construct Hamiltonian
    site_hamiltonian = np.diag(average_site_energies + total_site_reorg_energy + site_shifts[n]) + couplings + couplings.T # now including reorganisation shift
    evals, evecs = utils.sorted_eig(site_hamiltonian) # make sure to return excitons in basis going from lowest to highest energy with sorted_eig
    site_reorg_energies = np.zeros(site_hamiltonian.shape[0])
    site_reorg_energies.fill(total_site_reorg_energy)
    exciton_reorg_energies = np.zeros(site_hamiltonian.shape[0])
    for i in range(site_hamiltonian.shape[0]):
        exciton_reorg_energies[i] = os.exciton_reorg_energy(evecs[i], site_reorg_energies) # calculate exciton reorg energies
    evals = evals - exciton_reorg_energies # shift exciton energies down by exciton reorg energies
    
    # calculate modified Redfield rates
    rates = os.MRT_rate_PE545_quick(evals, evecs, g_site, g_site_dot, g_site_dot_dot, total_site_reorg_energy, temperature, integration_time)
    # construct Liouvillian for system
    liouvillian = np.zeros((rates.shape[0], rates.shape[1]))
    for i,row in enumerate(rates.T):
        liouvillian[i,i] = -np.sum(row)
    liouvillian += rates
    
    # calculate time evolution starting from provided initial state in exciton basis
    init_dv = exciton_init_states[n]
    dv_history = te.liouvillian_time_evolution(init_dv, liouvillian, duration, timestep)
    # transform density matrix time evolution to site basis and add to other realisations
    evecs = evecs.T
    site_history = np.zeros((site_hamiltonian.shape[0], time.size))
    for i,dv in enumerate(dv_history):
        exciton_dm = np.diag(dv)
        site_dm = np.dot(evecs, np.dot(exciton_dm, evecs.T))
        site_history[:,i] = np.diag(site_dm)
        
    site_history_sum += site_history
    
# divide sum of transient dynamics by num_realisations to get averaged dynamics
site_history_average = site_history_sum / num_realisations

# plot
for i in range(average_site_hamiltonian.shape[0]):
    plt.plot(time, site_history_average[i], label=str(i+1))
plt.legend()
plt.show()
