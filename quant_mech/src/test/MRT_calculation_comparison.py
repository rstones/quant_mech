'''
Created on 28 Nov 2014

Module to compare two methods of calculating modified Redfield rates from open_systems module, my initial version and the one
translated from Ed's Mathematica code. 

@author: rstones
'''
import numpy as np
import numpy.fft as fft
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import quant_mech.utils as utils
import quant_mech.open_systems as os

e1 = 1000.
delta_E_values = np.logspace(0,3.,100) # wavenumbers
coupling_values = np.array([20., 100., 500.]) # wavenumbers
 
def hamiltonian(delta_E, V):
    return np.array([[delta_E/2., V],
                    [V, -delta_E/2.]])
 
reorg_energy = 100. # wavenumbers
cutoff_freq = 53. # wavenumbers
temperature = 300.

time = np.linspace(0,0.5,1000)
site_lbf = os.site_lbf_ed(time, os.lbf_coeffs(reorg_energy, cutoff_freq, temperature, None, 0))

exciton_reorg_energies1 = []
exciton_reorg_energies2 = []
for i,delta_E in enumerate(delta_E_values):
    evals, evecs = utils.sorted_eig(hamiltonian(delta_E, 20.))
    exciton_reorg_energies1.append(os.generalised_exciton_reorg_energy(np.array([evecs[0], evecs[0], evecs[0], evecs[0]]), np.array([reorg_energy, reorg_energy])))
    exciton_reorg_energies2.append(np.sum(evecs[0]**4) * reorg_energy)
    
plt.plot(delta_E_values, exciton_reorg_energies1, label='1')
plt.plot(delta_E_values, exciton_reorg_energies2, label='2')
plt.legend()
plt.show()