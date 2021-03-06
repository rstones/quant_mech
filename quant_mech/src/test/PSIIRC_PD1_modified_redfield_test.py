'''
Created on 12 May 2016

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils
import quant_mech.open_systems as os
import quant_mech.time_evolution as te
import quant_mech.time_utils as time_utils

np.set_printoptions(precision=3, linewidth=120, suppress=True)

# system parameters
# average site and CT energies
average_site_CT_energies = np.array([15260., 15190., 15000., 15100., 15030., 15020., 15182.])

# site-CT couplings
couplings = np.array([[0,150.,-42.,-55.,-6.,17.,45.],
                     [0,0,-56.,-36.,20.,-2.,45.],
                     [0,0,0,7.,46.,-4.,0],
                     [0,0,0,0,-5.,37.,0],
                     [0,0,0,0,0,-3.,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,]])

def mode_params(damping):
        # freq and damping constant in wavenumbers
        return np.array([(97.,0.02396,damping),
                         (138.,0.02881,damping),
                         (213.,0.03002,damping),
                         (260.,0.02669,damping),
                         (298.,0.02669,damping),
                         (342.,0.06035,damping),
                         (388.,0.02487,damping),
                         (425.,0.01486,damping),
                         (518.,0.03942,damping),
                         (546.,0.00269,damping),
                         (573.,0.00849,damping),
                         (585.,0.00303,damping),
                         (604.,0.00194,damping),
                         (700.,0.00197,damping),
                         (722.,0.00394,damping),
                         (742.,0.03942,damping),
                         (752.,0.02578,damping),
                         (795.,0.00485,damping),
                         (916.,0.02123,damping),
                         (986.,0.01031,damping),
                         (995.,0.02274,damping),
                         (1052.,0.01213,damping),
                         (1069.,0.00636,damping),
                         (1110.,0.01122,damping),
                         (1143.,0.04094,damping),
                         (1181.,0.01759,damping),
                         (1190.,0.00667,damping),
                         (1208.,0.01850,damping),
                         (1216.,0.01759,damping),
                         (1235.,0.00697,damping),
                         (1252.,0.00636,damping),
                         (1260.,0.00636,damping),
                         (1286.,0.00454,damping),
                         (1304.,0.00576,damping),
                         (1322.,0.03032,damping),
                         (1338.,0.00394,damping),
                         (1354.,0.00576,damping),
                         (1382.,0.00667,damping),
                         (1439.,0.00667,damping),
                         (1487.,0.00788,damping),
                         (1524.,0.00636,damping),
                         (1537.,0.02183,damping),
                         (1553.,0.00909,damping),
                         (1573.,0.00454,damping),
                         (1580.,0.00454,damping),
                         (1612.,0.00454,damping),
                         (1645.,0.00363,damping),
                         (1673.,0.00097,damping)])

hamiltonian = np.diag(average_site_CT_energies) + couplings + couplings.T

site_drude_reorg_energy = 35.
total_site_reorg_energy = 540.
site_reorg_energies = np.array([total_site_reorg_energy, total_site_reorg_energy, total_site_reorg_energy, \
                                total_site_reorg_energy, total_site_reorg_energy, total_site_reorg_energy, total_site_reorg_energy*1.5])
cutoff_freq = 40.
temperature = 77.
mode_damping = 10.
time = np.linspace(0,10,1000000) # need loads of time steps to get MRT rates to converge

lbf_fn = '../../data/PSIIRC_lbfs_' + str(int(temperature)) + 'K_data4.npz'
try:
    data = np.load(lbf_fn)
    lbf = data['lbf']
    lbf_dot = data['lbf_dot']
    lbf_dot_dot = data['lbf_dot_dot']
    time = data['time']
except IOError:
    print 'Starting lbf calculations at ' + str(time_utils.getTime())
    lbf_coeffs = os.lbf_coeffs(site_drude_reorg_energy, cutoff_freq, temperature, mode_params(mode_damping), 100)
    lbf = os.site_lbf_ed(time, lbf_coeffs)
    lbf_dot = os.site_lbf_dot_ed(time, lbf_coeffs)
    lbf_dot_dot = os.site_lbf_dot_dot_ed(time, lbf_coeffs)
    print 'Finished calculating lbfs at ' + str(time_utils.getTime())
    np.savez(lbf_fn, lbf=lbf, lbf_dot=lbf_dot, lbf_dot_dot=lbf_dot_dot, time=time)

# calculate MRT rates
evals,evecs = utils.sorted_eig(hamiltonian)

exciton_reorg_energies = np.array([os.exciton_reorg_energy(exciton, site_reorg_energies) for exciton in evecs])
# print evals - exciton_reorg_energies
# evals,evecs = utils.sort_evals_evecs(evals-exciton_reorg_energies, evecs)
# print evals

reorgs = np.array([total_site_reorg_energy, total_site_reorg_energy, total_site_reorg_energy,\
                   total_site_reorg_energy, total_site_reorg_energy, total_site_reorg_energy,\
                   1.5*total_site_reorg_energy])

MRT_rates2 = os.modified_redfield_rates_general_unordered(evals-exciton_reorg_energies, evecs,
                                                          np.array([lbf,lbf,lbf,lbf,lbf,lbf,1.5*lbf]), \
                                                          np.array([lbf_dot,lbf_dot,lbf_dot,lbf_dot,lbf_dot,lbf_dot,1.5*lbf_dot]), \
                                                          np.array([lbf_dot_dot,lbf_dot_dot,lbf_dot_dot,lbf_dot_dot,lbf_dot_dot,lbf_dot_dot,1.5*lbf_dot_dot]), \
                                                          reorgs,\
                                                          temperature, time)

liouvillian2 = MRT_rates2 - np.diag(np.sum(MRT_rates2, axis=0))

print evals
print MRT_rates2

MRT_rates,evals,evecs = os.modified_redfield_rates_general(evals-exciton_reorg_energies, evecs, lbf, lbf_dot, lbf_dot_dot, total_site_reorg_energy, temperature, time)
print evals
print MRT_rates

MRT_rates3,evals,evecs = os.modified_redfield_rates_general(evals, evecs,\
                                                           np.array([lbf,lbf,lbf,lbf,lbf,lbf,1.5*lbf]),\
                                                           np.array([lbf_dot,lbf_dot,lbf_dot,lbf_dot,lbf_dot,lbf_dot,1.5*lbf_dot]),\
                                                           np.array([lbf_dot_dot,lbf_dot_dot,lbf_dot_dot,lbf_dot_dot,lbf_dot_dot,lbf_dot_dot,1.5*lbf_dot_dot]),\
                                                           reorgs,\
                                                           temperature, time)
print MRT_rates3

# time evolution
liouvillian = MRT_rates - np.diag(np.sum(MRT_rates, axis=0))

# print liouvillian
# for col in liouvillian.T:
#     print np.sum(col)

# init_state = np.array([1., 0, 0, 0, 0, 0])
# duration = 15. # ps
# time_step = 0.01 # ps
# dv_dynamics = te.liouvillian_time_evolution(init_state, liouvillian, duration, time_step)
# dv_dynamics = np.array(dv_dynamics)
# 
# init_state = np.array([0, 1., 0, 0, 0, 0])
# dv_dynamics2 = te.liouvillian_time_evolution(init_state, liouvillian2, duration, time_step)
# dv_dynamics2 = np.array(dv_dynamics2)
# 
# # save data
# 
# # plot
# plt.subplot(121)
# time = np.arange(0, duration+time_step, time_step)
# labels = ['P_D1', 'P_D2', 'Chl_D1', 'Chl_D2', 'Phe_D1', 'Phe_D2']
# for i in range(6):
#     plt.plot(time, dv_dynamics.T[i], label=labels[i])
# plt.plot(time, np.sum(dv_dynamics, axis=1), label='trace')
# plt.ylim(0, 1.2)
# plt.legend().draggable()
# 
# 
# plt.subplot(122)
# labels = ['P_D1', 'P_D2', 'Chl_D1', 'Chl_D2', 'Phe_D1', 'Phe_D2']
# for i in range(6):
#     plt.plot(time, dv_dynamics2.T[i], label=labels[i])
# plt.plot(time, np.sum(dv_dynamics2, axis=1), label='trace')
# plt.ylim(0, 1.2)
# plt.legend().draggable()
# 
# plt.show()