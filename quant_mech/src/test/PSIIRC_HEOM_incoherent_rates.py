'''
Created on 11 Jun 2016

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils
import quant_mech.open_systems as os
import quant_mech.time_utils as tutils
from quant_mech.hierarchy_solver import HierarchySolver
import matplotlib
 
font = {'size':16}
matplotlib.rc('font', **font)

#np.set_printoptions(precision=3, suppress=True, linewidth=150)

print '[Starting script execution...]'

# system Hamiltonian and bath parameters
average_site_CT_energies = np.array([15260., 15190., 15000., 15100., 15030., 15020., 15992., 16132.])
# site-CT couplings
couplings = np.array([[0,150.,-42.,-55.,-6.,17.,0,0],
                     [0,0,-56.,-36.,20.,-2.,0,0],
                     [0,0,0,7.,46.,-4.,70.,0],
                     [0,0,0,0,-5.,37.,0,0],
                     [0,0,0,0,0,-3.,70.,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,40.],
                     [0,0,0,0,0,0,0,0]])
coherent_hamiltonian = np.diag(average_site_CT_energies) + couplings + couplings.T

system_dimension = 10
system_hamiltonian = np.zeros((system_dimension,system_dimension))
system_hamiltonian[1:7,1:7] = coherent_hamiltonian[:6,:6]

evalues, evectors = utils.sorted_eig(coherent_hamiltonian[:6,:6])
site_exciton_transform = np.eye(10)
site_exciton_transform[1:7, 1:7] = evectors.T

reorg_energy = 35.
cutoff_freq = 40.
temperature = 300.

# jump operators. basis { ground , P_D1 , P_D2 , Chl_D1 , Chl_D2 , Phe_D1 , Phe_D2 , CT1 , CT2 , empty }
# 2 for pumping from ground state
# 12 for primary CT
# 2 for secondary CT
# 2 for coupling to leads
jump_operators = np.zeros((18, system_dimension, system_dimension))
# excitation: ground -> Chl_D1
jump_operators[0] = np.dot(site_exciton_transform, np.dot(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [1., 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), site_exciton_transform.T))
# deexcitation: Chl_D1 -> ground
jump_operators[1] = np.dot(site_exciton_transform, np.dot(np.array([[0, 1., 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), site_exciton_transform.T))
# jump_operators[0] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [1., 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# # deexcitation: Chl_D1 -> ground
# jump_operators[1] = np.array([[0, 0, 0, 1., 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# generate forward and backward operators for primary CT in site basis
for i in range(1,7):
    op_forward = np.zeros((10,10))
    op_forward[7,i] = 1.
    op_forward = np.dot(site_exciton_transform, np.dot(op_forward, site_exciton_transform.T))
    jump_operators[2*i] = op_forward
    op_backward = np.zeros((10,10))
    op_backward[i,7] = 1.
    op_backward = np.dot(site_exciton_transform, np.dot(op_backward, site_exciton_transform.T))
    jump_operators[2*i+1] = op_backward

# # Chl_D1 -> CT1
# jump_operators[2] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 1., 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# # CT1 -> Chl_D1
# jump_operators[3] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 1., 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# # Phe_D1 -> CT1
# jump_operators[4] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 1., 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# # CT1 -> Phe_D1
# jump_operators[5] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 1., 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# CT1 -> CT2
jump_operators[14] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1., 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# CT2 -> CT1
jump_operators[15] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1., 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# drain lead: CT2 -> empty
jump_operators[16] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1., 0]])
# source lead: empty -> ground
jump_operators[17] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# jump rates
# excitation rates to Chl_D1
# Forster rates from sites Chl_D1 and Phe_D1
# lead coupling rates
gamma_ex = 0.125 * 1.24e-6 * utils.EV_TO_WAVENUMS
n_ex = 60000
excitation_rate = n_ex * gamma_ex
deexcitation_rate = (n_ex+1.) * gamma_ex

# forster params
total_reorg_energy = 540.
CT1_scaling = 3.
CT2_scaling = 4.

bare_chl_energy = system_hamiltonian[4,4] - total_reorg_energy
bare_phe_energy = system_hamiltonian[6,6] - total_reorg_energy
bare_CT1_energy = average_site_CT_energies[6] - total_reorg_energy*CT1_scaling
bare_CT2_energy = average_site_CT_energies[7] - total_reorg_energy*CT2_scaling

chl_state = np.array([0, 0, 1., 0, 0, 0, 0, 0])
phe_state = np.array([0, 0, 0, 0, 1., 0, 0, 0])
CT1_state = np.array([0, 0, 0, 0, 0, 0, 1., 0])
CT2_state = np.array([0, 0, 0, 0, 0, 0, 0, 1.])

time = np.linspace(0,2.,256000)
site_lbf = os.site_lbf_ed(time, os.lbf_coeffs(reorg_energy, cutoff_freq, temperature, None, 100))

# primary CT rates in site basis
chl_CT1_rate = os.forster_rate(bare_chl_energy, bare_CT1_energy, reorg_energy, reorg_energy*CT1_scaling, \
                                    site_lbf, site_lbf*CT1_scaling, 0, 0, chl_state, CT1_state, coherent_hamiltonian, time)
CT1_chl_rate = os.forster_rate(bare_CT1_energy, bare_chl_energy, reorg_energy, reorg_energy*CT1_scaling, \
                                    site_lbf, site_lbf*CT1_scaling, 0, 0, chl_state, CT1_state, coherent_hamiltonian, time)
phe_CT1_rate = os.forster_rate(bare_phe_energy, bare_CT1_energy, reorg_energy, reorg_energy*CT1_scaling, \
                                    site_lbf, site_lbf*CT1_scaling, 0, 0, phe_state, CT1_state, coherent_hamiltonian, time)
CT1_phe_rate = os.forster_rate(bare_CT1_energy, bare_phe_energy, reorg_energy, reorg_energy*CT1_scaling, \
                                    site_lbf, site_lbf*CT1_scaling, 0, 0, phe_state, CT1_state, coherent_hamiltonian, time)

forward_exciton_CT_rates = [2.86359127e+00, 2.09614196e-02, 7.89708095e+01, 5.47195323e+00, 2.91393623e+00, 2.92997893e-01]
backward_exciton_CT_rates = [2.32445332e+00, 1.40848357e-02, 4.18754762e+01, 2.46325661e+00, 1.25847348e+00, 3.69904601e-02]

# secondary CT rates
CT1_CT2_rate = os.forster_rate(bare_CT1_energy, bare_CT2_energy, reorg_energy*CT1_scaling, reorg_energy*CT2_scaling, \
                                    site_lbf*CT1_scaling, site_lbf*CT2_scaling, 0, 0, CT1_state, CT2_state, coherent_hamiltonian, time)
CT2_CT1_rate = os.forster_rate(bare_CT2_energy, bare_CT1_energy, reorg_energy*CT1_scaling, reorg_energy*CT2_scaling, \
                                    site_lbf*CT1_scaling, site_lbf*CT2_scaling, 0, 0, CT1_state, CT2_state, coherent_hamiltonian, time)

# check Forster rates have converged and obey detailed balance

CT2_empty_rate = 25.e-3 * utils.EV_TO_WAVENUMS
empty_ground_rate = 25.e-3 * utils.EV_TO_WAVENUMS
# jump_rates = np.array([excitation_rate, deexcitation_rate, chl_CT1_rate, CT1_chl_rate, \
#                        phe_CT1_rate, CT1_phe_rate, CT1_CT2_rate, CT2_CT1_rate, CT2_empty_rate, empty_ground_rate])

jump_rates = np.array([excitation_rate, deexcitation_rate])
for i in range(6):
    jump_rates = np.append(jump_rates, [forward_exciton_CT_rates[i], backward_exciton_CT_rates[i]])
#jump_rates[2:] *= 0.1
jump_rates = np.append(jump_rates, [CT1_CT2_rate, CT2_CT1_rate, CT2_empty_rate, empty_ground_rate])

single_mode_params = []#[(342., 342.*0.4, 100.)]

#print jump_rates
#jump_rates = np.zeros(18)
hs = HierarchySolver(system_hamiltonian, reorg_energy, cutoff_freq, temperature, jump_operators=jump_operators, jump_rates=jump_rates, underdamped_mode_params=single_mode_params)
hs.truncation_level = 6
init_state = np.zeros((10, 10))
init_state[0,0] = 1.
init_state = np.dot(site_exciton_transform, np.dot(init_state, site_exciton_transform.T))
hs.init_system_dm = init_state
dm_history, time = hs.calculate_time_evolution(0.01, 5.)
# transform back to exciton basis
exciton_dm_history = np.zeros((dm_history.shape[0], dm_history.shape[1], dm_history.shape[2]))
for i,dm in enumerate(dm_history):
    exciton_dm_history[i] = np.dot(site_exciton_transform.T, np.dot(dm, site_exciton_transform))
np.savez('../../data/PSIIRC_HEOM_incoherent_rates_slow_jump_rates_data.npz', exciton_dm_history=exciton_dm_history, \
                            time=time, reorg_energy=reorg_energy, cutoff_freq=cutoff_freq, temperature=temperature, \
                            site_exciton_transform=site_exciton_transform, jump_rates=jump_rates)

# np.savez('../../data/PSIIRC_HEOM_incoherent_rates_mode_data.npz', exciton_dm_history=exciton_dm_history, \
#                             time=time, reorg_energy=reorg_energy, cutoff_freq=cutoff_freq, temperature=temperature, \
#                             site_exciton_transform=site_exciton_transform, jump_rates=jump_rates, mode_params=single_mode_params)


# truncation_level = 5
# 
# # try different reorg energies
# reorg_energy_values = [35., 70., 100.]
# cutoff_freq = 40.
# temperature = 300.
# 
# for reorg_energy in reorg_energy_values:
#     print 'Started calculating HEOM dynamics for reorg_energy = ' + str(reorg_energy) + 'cm-1 at ' + str(tutils.getTime())
#     hs = HierarchySolver(system_hamiltonian, reorg_energy, cutoff_freq, temperature, jump_operators=jump_operators, jump_rates=jump_rates)
#     init_state = np.zeros((10, 10))
#     init_state[0,0] = 1.
#     init_state = np.dot(site_exciton_transform, np.dot(init_state, site_exciton_transform.T))
#     dm_history, time = hs.converged_time_evolution(init_state, truncation_level, truncation_level, 0.001, 5.)
#     # transform back to exciton basis
#     exciton_dm_history = np.zeros((dm_history.shape[0], dm_history.shape[1], dm_history.shape[2]))
#     for i,dm in enumerate(dm_history):
#         exciton_dm_history[i] = np.dot(site_exciton_transform.T, np.dot(dm, site_exciton_transform))
#     np.savez('../../data/PSIIRC_HEOM_incoherent_rates_reorg_energy_'+str(int(reorg_energy))+'_wavenums.npz', exciton_dm_history=exciton_dm_history, \
#                                 time=time, reorg_energy=reorg_energy, cutoff_freq=cutoff_freq, temperature=temperature, \
#                                 site_exciton_transform=site_exciton_transform)
#     
# # try different reorg energies
# reorg_energy = 35.
# cutoff_freq_values = [40., 70., 100.]
# temperature = 300.
# 
# for cutoff_freq in cutoff_freq_values:
#     print 'Started calculating HEOM dynamics for cutoff_freq = ' + str(cutoff_freq) + 'cm-1 at ' + str(tutils.getTime())
#     hs = HierarchySolver(system_hamiltonian, reorg_energy, cutoff_freq, temperature, jump_operators=jump_operators, jump_rates=jump_rates)
#     init_state = np.zeros((10, 10))
#     init_state[0,0] = 1.
#     init_state = np.dot(site_exciton_transform, np.dot(init_state, site_exciton_transform.T))
#     dm_history, time = hs.converged_time_evolution(init_state, truncation_level, truncation_level, 0.001, 5.)
#     # transform back to exciton basis
#     exciton_dm_history = np.zeros((dm_history.shape[0], dm_history.shape[1], dm_history.shape[2]))
#     for i,dm in enumerate(dm_history):
#         exciton_dm_history[i] = np.dot(site_exciton_transform.T, np.dot(dm, site_exciton_transform))
#     np.savez('../../data/PSIIRC_HEOM_incoherent_rates_cutoff_freq_'+str(int(cutoff_freq))+'_wavenums.npz', exciton_dm_history=exciton_dm_history, \
#                                 time=time, reorg_energy=reorg_energy, cutoff_freq=cutoff_freq, temperature=temperature, \
#                                 site_exciton_transform=site_exciton_transform)

# print 'Saving data...'
# np.savez('PSIIRC_incoherent_HEOM_full_system_data.npz', dm_history=dm_history, system_hamiltonian=system_hamiltonian, time=time, \
#                     reorg_energy=reorg_energy, cutoff_freq=cutoff_freq, temperature=temperature, jump_operators=jump_operators, jump_rates=jump_rates)
  
print '[Script execution finished]'
  
labels = ['g', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'CT1', 'CT2', 'empty']
 
print time.shape
print exciton_dm_history.shape
 
for i in range(hs.system_dimension):
    plt.plot(time, exciton_dm_history[:-1,i,i], label=labels[i], linewidth=2)
plt.legend().draggable()
plt.xlabel('time (ps)')
plt.ylabel('population')
plt.xlim(0,5)
plt.show()

'''
Steady state calculation
'''
# start_time = tutils.getTime()
# steady_state = hs.calculate_steady_state(3,3)
# print steady_state
# 
# end_time = tutils.getTime()
# print 'Calculation took ' + str(tutils.duration(end_time, start_time))

#np.savez('../../data/PSIIRC_steady_state.npz', steady_state=steady_state)