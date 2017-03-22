'''
Created on 22 Mar 2017

@author: richard
'''
import numpy as np
import quant_mech.hierarchy_solver as hs
import quant_mech.hierarchy_solver_filtering_truncation as hsft
from quant_mech.OBOscillator import OBOscillator

time_step = 0.01
#duration = 1. # picoseconds
duration = 15.7 # inverse wavenums

'''Shi J. Chem. Phys. 130, 2009'''
beta = 1.
electronic_coupling = 0.1 / beta
system_hamiltonian = np.array([[0, electronic_coupling], [electronic_coupling, 0]])
reorg_energy = 5. / beta
cutoff_freq = 1. / beta
K = 0
environment = [(OBOscillator(reorg_energy, cutoff_freq, beta, K=K),),
                   (OBOscillator(reorg_energy, cutoff_freq, beta, K=K),)]

solver = hs.HierarchySolver(system_hamiltonian, environment, beta, num_matsubara_freqs=K, temperature_correction=False)