'''
Created on 15 Mar 2017

@author: richard
'''
import numpy as np
import quant_mech.utils as utils
from quant_mech.OBOscillator import OBOscillator
from quant_mech.UBOscillator import UBOscillator
from quant_mech.hierarchy_solver import HierarchySolver

import scipy as sp
print sp.__version__

'''Ishizaki and Fleming params'''
electronic_coupling = 0.1
system_hamiltonian = np.array([[0, 0, 0],
                               [0, 0.2, electronic_coupling],
                               [0, electronic_coupling, 0]])
reorg_energy = 100.
cutoff_freq = 5.
temperature = 2.7 # Kelvin
beta = 4.297 # 0.4 #1. / (utils.KELVIN_TO_WAVENUMS * temperature)
mode_params = [] #[(200., 0.25, 10.)]

jump_ops = np.array([np.array([[0, 0, 0],
                              [1., 0, 0],
                              [0, 0, 0]]), np.array([[0, 0, 1.],
                                                     [0, 0, 0],
                                                     [0, 0, 0]])])
jump_rates = np.array([0.1, 0.0025])

K = 4
environment = []
if mode_params: # assuming that there is a single identical mode on each site 
    environment = [(OBOscillator(reorg_energy, cutoff_freq, beta, K=K), UBOscillator(mode_params[0][0], mode_params[0][1], mode_params[0][2], beta, K=K)), \
                   (OBOscillator(reorg_energy, cutoff_freq, beta, K=K), UBOscillator(mode_params[0][0], mode_params[0][1], mode_params[0][2], beta, K=K))]
else:
    environment = [(),
                   (OBOscillator(reorg_energy, cutoff_freq, beta, K=K),),
                   (OBOscillator(reorg_energy, cutoff_freq, beta, K=K),)]
hs = HierarchySolver(system_hamiltonian, environment, beta, jump_ops, jump_rates, num_matsubara_freqs=K, temperature_correction=True)
hs.truncation_level = 7

hm = hs.construct_hierarchy_matrix_super_fast()
print 'hierarchy matrix shape: ' + str(hm.shape)
print hs.dm_per_tier()

np.savez('DQD_heom_matrix_N7_K4.npz', hm=hm)

import scipy.sparse.linalg as spla
np.set_printoptions(precision=6, linewidth=150, suppress=True)
v0 = np.zeros(hm.shape[0])
v0[0] = 1./3
v0[4] = 1./3
v0[8] = 1./3
evals,evec = spla.eigs(hm.tocsc(), k=1, sigma=0, which='LM', v0=v0)#, ncv=100)

print evals
evec = evec[:9]
evec.shape = 3,3
evec /= np.trace(evec)
print evec


