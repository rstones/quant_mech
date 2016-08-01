'''
Created on 19 Apr 2016

@author: rstones
'''
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import quant_mech.utils as utils
import quant_mech.open_systems as os
import quant_mech.time_evolution as te

np.set_printoptions(precision=3, linewidth=120, suppress=True)

# average site and CT energies
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



site_drude_reorg_energy = 35.
total_site_reorg_energy = 540.
site_reorg_energies = np.array([total_site_reorg_energy, total_site_reorg_energy, total_site_reorg_energy, \
                                total_site_reorg_energy, total_site_reorg_energy, total_site_reorg_energy])
cutoff_freq = 40.
temperature = 77.
mode_damping = 10.
time = np.linspace(0,10,1000000) # need loads of time steps to get MRT rates to converge

lbf_fn = '../../data/PSIIRC_lbfs_' + str(int(temperature)) + 'K_data.npz'
try:
    data = np.load(lbf_fn)
    lbf = data['lbf']
    lbf_dot = data['lbf_dot']
    lbf_dot_dot = data['lbf_dot_dot']
except IOError:
    lbf_coeffs = os.lbf_coeffs(site_drude_reorg_energy, cutoff_freq, temperature, mode_params(mode_damping), 5)
    lbf = os.site_lbf_ed(time, lbf_coeffs)
    lbf_dot = os.site_lbf_dot_ed(time, lbf_coeffs)
    lbf_dot_dot = os.site_lbf_dot_dot_ed(time, lbf_coeffs)
    np.savez(lbf_fn, lbf=lbf, lbf_dot=lbf_dot, lbf_dot_dot=lbf_dot_dot, time=time)
    
num_realisations = 5000
PSIIRC_disorder = np.array([95., 95., 95., 95., 95., 95., 380., 570.])

def standard_deviation(fwhm):
    return fwhm / (2.*np.sqrt(2.*np.log(2.)))

energy_samples = np.zeros((average_site_CT_energies.size, num_realisations))
for i in range(average_site_CT_energies.size):
    energy_samples[i] = random.normal(average_site_CT_energies[i], standard_deviation(PSIIRC_disorder[i]), num_realisations)

dynamics_duration = 15.
dynamics_timestep = 0.01
dynamics_time = np.arange(0, dynamics_duration+dynamics_timestep, dynamics_timestep)
population_dynamics_realisations = np.zeros((num_realisations, average_site_CT_energies.size, dynamics_time.size))
liouvillians = np.zeros((num_realisations, average_site_CT_energies.size, average_site_CT_energies.size))

site_init_state = np.array([0.07, 0.14, 0.79, 0.4, 0.58, 0.56, 0, 0])
normalisation = np.sum(site_init_state)
    
for n in range(num_realisations):
    
    if n % 100 == 0:
        print "Calculating for realisation " + str(n)
    
    hamiltonian = np.diag(energy_samples.T[n]) + couplings + couplings.T
    
    site_hamiltonian = hamiltonian[:6,:6]
    
    evals, evecs = utils.sorted_eig(site_hamiltonian)
    exciton_reorg_energies = np.array([os.exciton_reorg_energy(exciton, site_reorg_energies) for exciton in evecs])
    
    '''
    # remove reorg energy and reorder evals/evecs from lowest to highest energy
    evals, evecs = utils.sort_evals_evecs(evals-exciton_reorg_energies, evecs)
    
    # calculate modified Redfield rates
    MRT_rates = os.modified_redfield_rates_general(evals, evecs, lbf, lbf_dot, lbf_dot_dot, total_site_reorg_energy, temperature, time)[0]
    
    
    # need to reorder exciton_reorg_energies too! easier to just recalculate with new evecs
    exciton_reorg_energies = np.array([os.exciton_reorg_energy(exciton, site_reorg_energies) for exciton in evecs])
    '''
    
    '''
    temporary to test that this reproduces the results for incorrect ordering of exciton levels
    '''
    MRT_rates = os.modified_redfield_rates(evals-exciton_reorg_energies, evecs, lbf, lbf_dot, lbf_dot_dot, total_site_reorg_energy, temperature, time)
    
    
    site_lbfs = np.array([lbf, lbf, lbf, lbf, lbf, lbf])
    exciton_lbfs = np.array([os.exciton_lbf(exciton, site_lbfs) for exciton in evecs])
     
    primary_CT_scaling = 3.
    primary_CT_reorg_energy = primary_CT_scaling * total_site_reorg_energy
    primary_CT_state = np.array([0, 0, 0, 0, 0, 0, 1., 0])
    secondary_CT_scaling = 4.
    secondary_CT_reorg_energy = secondary_CT_scaling * total_site_reorg_energy
    secondary_CT_state = np.array([0, 0, 0, 0, 0, 0, 0, 1.])
     
    # calculate Forster rates to primary and secondary CT states
    forward_primary_forster_rates = np.zeros(site_hamiltonian.shape[0])
    backward_primary_forster_rates = np.zeros(site_hamiltonian.shape[0])
    for i,E in enumerate(evals):
        # double check the effect of the lifetimes on the Forster rates
        # also check Forster rates converge with the number of time points used
        forward_primary_forster_rates[i] = os.forster_rate(E, hamiltonian[6,6]-primary_CT_reorg_energy, exciton_reorg_energies[i], primary_CT_reorg_energy, \
                                                           exciton_lbfs[i], primary_CT_scaling*lbf, 0, 0, \
                                                           np.append(evecs[i], [0,0]), primary_CT_state, hamiltonian, time)
        backward_primary_forster_rates[i] = os.forster_rate(hamiltonian[6,6]-primary_CT_reorg_energy, E, exciton_reorg_energies[i], primary_CT_reorg_energy, \
                                                           exciton_lbfs[i], primary_CT_scaling*lbf, 0, 0, \
                                                           np.append(evecs[i], [0,0]), primary_CT_state, hamiltonian, time)
    
    # construct Liouvillian
    liouvillian = np.zeros(hamiltonian.shape)
    liouvillian[:6,:6] = MRT_rates
    liouvillian[6,:6] = forward_primary_forster_rates
    liouvillian.T[6,:6] = backward_primary_forster_rates
    # forward secondary CT rate
    liouvillian[7,6] = os.forster_rate(hamiltonian[6,6]-primary_CT_reorg_energy, hamiltonian[7,7]-secondary_CT_reorg_energy, \
                                       primary_CT_reorg_energy, secondary_CT_reorg_energy, \
                                       primary_CT_scaling*lbf, secondary_CT_scaling*lbf, 0, 0, \
                                       primary_CT_state, secondary_CT_state, hamiltonian, time)
    # backward secondary CT rate
    liouvillian[6,7] = os.forster_rate(hamiltonian[7,7]-secondary_CT_reorg_energy, hamiltonian[6,6]-primary_CT_reorg_energy, \
                                       primary_CT_reorg_energy, secondary_CT_reorg_energy, \
                                       primary_CT_scaling*lbf, secondary_CT_scaling*lbf, 0, 0, \
                                       primary_CT_state, secondary_CT_state, hamiltonian, time)
    
    for i,col in enumerate(liouvillian.T):
        liouvillian[i,i] = -np.sum(col)
        
    liouvillians[n] = liouvillian
    
    #  propagate initial state for 15 ps
    site_exciton_transform = np.zeros(hamiltonian.shape)
    for i,exciton in enumerate(evecs):
        site_exciton_transform[i] = np.append(exciton, [0,0])
    site_exciton_transform[6,6] = 1.
    site_exciton_transform[7,7] = 1.
    
    exciton_init_state = np.diag(np.dot(site_exciton_transform, np.dot(np.diag(site_init_state), site_exciton_transform.T)))
    
    dv_history = te.liouvillian_time_evolution(exciton_init_state, liouvillian, dynamics_duration, dynamics_timestep)
    dv_history = np.array(dv_history)
    
    # loop over density vectors in history and transform to site basis
    site_dv_history = np.zeros(dv_history.shape)
    for i in range(dv_history.shape[0]):
        site_dv_history[i] = np.diag(np.dot(site_exciton_transform.T, np.dot(np.diag(dv_history[i]), site_exciton_transform)))
    
    population_dynamics_realisations[n] = site_dv_history.T
    
data_fn = '../../data/PSIIRC_ChlD1_pathway_dynamics_' + str(int(temperature)) + 'K__incorrect_ordering_data.npz'
try:
    saved_data = np.load(data_fn)
    saved_realisations = saved_data['population_dynamics_realisations']
    population_dynamics_realisations = np.append(population_dynamics_realisations, saved_realisations, axis=0)
    num_realisations += saved_data['num_realisations']
except IOError:
    pass
    
np.savez(data_fn, population_dynamics_realisations=population_dynamics_realisations, \
                        dynamics_time=dynamics_time, num_realisations=num_realisations, liouvillians=liouvillians)

# averaged_population_dynamics = np.sum(population_dynamics_realisations, axis=0) / num_realisations
#  
# for i in range(hamiltonian.shape[0]):
#     plt.plot(dynamics_time, averaged_population_dynamics[i])
# plt.show()

print '[Script execution complete]'

