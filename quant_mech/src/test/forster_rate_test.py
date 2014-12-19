'''
Created on 19 Dec 2014

Module to test Forster rate calculation in open_systems module

@author: rstones
'''
import numpy as np
import quant_mech.open_systems as os

# set up data
model = PSIIRCCurrentNoiseModelIntermediateCTState() # model should be a subclass of DataModel

# params
site_reorg_energy = 35.
cutoff_freq = 40.
temperature = 77.
mode_damping = 10.
mode_params = model.high_energy_mode_params(mode_damping)

CT_reorg_energy_scale = 3
CT_reorg_energy = CT_reorg_energy_scale * site_reorg_energy
CT_lifetime = 22. * utils.WAVENUMS_TO_INVERSE_PS # wavenumbers


# Hamiltonian in basis { P_D1, P_D2, Chl_D1, Chl_D2, Phep_D1, Pheo_D2, Chl_D1+Pheo_D1- } including reorganisation shift of 540cm-1 for sites and 1620cm-1 for CT state
num_sites = 6
H_site_CT = np.array([[15260.,150.,-42.,-55.,-6.,17.,0],
                     [150.,15190.,-56.,-36.,20.,-2.,0],
                     [-42.,-56.,15000.,7.,46.,-4.,70.],
                     [-55.,-36.,7.,15100.,-5.,37.,0],
                     [-6.,20.,46.,-5.,15030.,-3.,70.],
                     [17.,-2.,-4.,37.,-3.,15020.,0],
                     [0,0,70.,0,70.,0,15992.]])

# 6 site Hamiltonian in basis { P_D1, P_D2, Chl_D1, Chl_D2, Phep_D1, Pheo_D2 } including reorganisation shift of 540cm-1 for all sites
H_site = H_site_CT[:num_sites,:num_sites]
H_site_dim = H_site.shape[0]
exciton_energies, exciton_states = utils.sorted_eig(H_site) # diagonalise site Hamiltonian including reorganisation shifts
exciton_states = np.array([np.append(state, 0) for state in exciton_states])
CT_state = np.array([0, 0, 0, 0, 0, 0, 1.])

# calculate modified Redfield rates
time_interval = 20
time = np.linspace(0,time_interval,time_interval*32000)
num_expansion_terms = 10
site_lbf, site_lbf_dot, site_lbf_dot_dot, total_site_reorg_energy = os.modified_redfield_params(time, site_reorg_energy, cutoff_freq, temperature, mode_params, num_expansion_terms)

# calculate total exciton reorg energies
total_site_reorg_energies = np.zeros(num_sites)
print 'total site reorg energy: ' + str(total_site_reorg_energy)
total_site_reorg_energies.fill(total_site_reorg_energy)
total_exciton_reorg_energies = np.array([os.exciton_reorg_energy(exciton_states[i][:num_sites], total_site_reorg_energies) for i in range(exciton_energies.size)])
exciton_energies = exciton_energies - total_exciton_reorg_energies # get correct bare exction energies
