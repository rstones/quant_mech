'''
Created on 18 Dec 2014

Module to calculate thermal state (ie. steady state) of PE545 system to compare to long time transient dynamics

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils
import quant_mech.open_systems as os
import quant_mech.time_evolution as te

# basis { PEB_50/61C, DBV_A, DVB_B, PEB_82C, PEB_158C, PEB_50/61D, PEB_82D, PEB_158D }
site_energies = np.array([18532., 18008., 17973., 18040., 18711., 19574., 19050., 18960.]) # no reorganisation shift included

couplings = np.array([[0, 1., -37., 37., 23., 92., -16., 12.],
                      [0, 0, 4., -11., 33., -39., -46., 3.],
                      [0, 0, 0, 45., 3., 2., -11., 34.],
                      [0, 0, 0, 0, -7., -17., -3., 6.],
                      [0, 0, 0, 0, 0, 18., 7., 6.],
                      [0, 0, 0, 0, 0, 0, 40., 26.],
                      [0, 0, 0, 0, 0, 0, 0, 7.],
                      [0, 0, 0, 0, 0, 0, 0, 0]])

site_hamiltonian = np.diag(site_energies) + couplings + couplings.T

evals, evecs = utils.sorted_eig(site_hamiltonian)

site_reorg_shift = 110.
site_reorg_energies = np.zeros(site_energies.size)
site_reorg_energies.fill(site_reorg_shift)
shifted_site_hamiltonian = site_hamiltonian + np.diag(site_reorg_energies)

shifted_evals, shifted_evecs = utils.sorted_eig(shifted_site_hamiltonian)
exciton_reorg_energies = os.exciton_reorg_energy(shifted_evecs, site_reorg_energies)

print evals
print shifted_evals - exciton_reorg_energies
print exciton_reorg_energies
for i,el in enumerate(evecs.flatten()):
    print shifted_evecs.flatten()[i] + 0.0000001 > el > shifted_evecs.flatten()[i] - 0.0000001

