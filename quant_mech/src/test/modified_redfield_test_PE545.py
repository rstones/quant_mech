'''
Created on 10 Dec 2014

@author: rstones
'''
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import quant_mech.utils as utils
import quant_mech.open_systems as os
import quant_mech.time_utils as te
import time

rand.seed()

# basis { PEB_50/61C, DBV_A, DVB_B, PEB_82C, PEB_158C, PEB_50/61D, PEB_82D, PEB_158D }
site_energies = np.array([18532., 18008., 17973., 18040., 18711., 19574., 19050., 18960.])

couplings = np.array([[0, 1., -37., 37., 23., 92., -16., 12.],
                      [0, 0, 4., -11., 33., -39., -46., 3.],
                      [0, 0, 0, 45., 3., 2., -11., 34.],
                      [0, 0, 0, 0, -7., -17., -3., 6.],
                      [0, 0, 0, 0, 0, 18., 7., 6.],
                      [0, 0, 0, 0, 0, 0, 40., 26.],
                      [0, 0, 0, 0, 0, 0, 0, 7.],
                      [0, 0, 0, 0, 0, 0, 0, 0]])

site_hamiltonian = np.diag(site_energies) + couplings + couplings.T

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

# to include disorder....
# generate distribution of energies for each site
def standard_deviation(fwhm):
    return fwhm / (2.*np.sqrt(2.*np.log(2.)))

num_realisations = 500
FWHM = 400.
site_energy_samples = []
for E in site_energies:
    site_energy_samples.append(rand.normal(E, standard_deviation(FWHM), num_realisations))
    
# parameters for time evolution
duration = 5.
timestep = 0.01
time = np.arange(0, duration+timestep, timestep)
init_dv = np.array([0.35, 0.12, 0.1, 0.1, 0.34, 0.61, 0.46, 0.5]) # init state in site basis

data_filename = '../../data/modified_redfield_test_PE545_disorder_data5.npz'


try:
    data = np.load(data_filename)
    print 'data file already exists, do you want to append to it?' 
except:
    pass

time_interval = 10
integration_time = np.linspace(0, time_interval, time_interval*16000) # time_interval*16000
num_expansion_terms = 10
mode_params = PE545_mode_params(mode_damping)

coeffs = os.lbf_coeffs(reorg_energy1, cutoff_freq1, temperature, mode_params, num_expansion_terms)
coeffs = np.concatenate((coeffs, os.lbf_coeffs(reorg_energy2, cutoff_freq2, temperature, None, num_expansion_terms)))
g_site = os.site_lbf_ed(integration_time, coeffs)
g_site_dot = os.site_lbf_dot_ed(integration_time, coeffs)
g_site_dot_dot = os.site_lbf_dot_dot_ed(integration_time, coeffs)
total_site_reorg_energy = reorg_energy1 + reorg_energy2 + np.sum([mode[0]*mode[1] for mode in mode_params])

shift_before_diagonalisation = True

# in each realisation pick the next value from the distribution for each site to construct the Hamiltonian
for n in range(num_realisations):
    print 'Calculating realisation number ' + str(n+1)
    realisation_energies = np.zeros(site_energies.size)
    for i in range(site_energies.size):
        realisation_energies[i] = site_energy_samples[i][n]
    if not shift_before_diagonalisation:
        hamiltonian = np.diag(realisation_energies) + couplings + couplings.T
        evals, evecs = utils.sorted_eig(hamiltonian)
    else:
        hamiltonian = np.diag(realisation_energies + total_site_reorg_energy) + couplings + couplings.T # shift site energies by reorg energy
        evals, evecs = utils.sorted_eig(hamiltonian) # diagonalise
        site_reorg_energies = np.zeros(hamiltonian.shape[0])
        site_reorg_energies.fill(total_site_reorg_energy)
        exciton_reorg_energies = np.zeros(hamiltonian.shape[0])
        for i in range(hamiltonian.shape[0]):
            exciton_reorg_energies[i] = os.exciton_reorg_energy(evecs[i], site_reorg_energies) # calculate exciton reorg energies
        evals = evals - exciton_reorg_energies # shift exciton energies down by exciton reorg energies
        
    # calculate the rates and time evolution for the realisation
    realisation_rates = os.MRT_rate_PE545_quick(evals, evecs, g_site, g_site_dot, g_site_dot_dot, total_site_reorg_energy, temperature, integration_time)
    
    liouvillian = np.zeros((realisation_rates.shape[0], realisation_rates.shape[1]))
    for i,row in enumerate(realisation_rates.T):
        liouvillian[i,i] = -np.sum(row)
    liouvillian += realisation_rates
    
    # make sure to return excitons in basis going from lowest to highest energy with sorted_eig
    evecs = evecs.T
    init_dv = np.diag(np.dot(evecs.T, np.dot(np.diag(init_dv), evecs))) # convert init dv in site basis to exciton basis before time evolution calculation
    
    dv_history = te.liouvillian_time_evolution(init_dv, liouvillian, duration, timestep)
    
    site_history = np.zeros((site_energies.size, time.size))
    for i,dv in enumerate(dv_history):
        exciton_dm = np.diag(dv)
        site_dm = np.dot(evecs, np.dot(exciton_dm, evecs.T))
        site_history[:,i] = np.diag(site_dm)
        
    beta = 1. / (utils.KELVIN_TO_WAVENUMS * temperature)
    exciton_thermal_state = np.exp(-beta * evals) / np.sum([np.exp(-beta*E) for E in evals])
    site_thermal_state = np.diag(np.dot(evecs, np.dot(np.diag(exciton_thermal_state), evecs.T)))
        
    # add the time evolution for each site to the previous realisations
    try:
        data = np.load(data_filename)
        site_histories_sum = data['site_histories_sum']
        site_histories_sum[:,:] += site_history
        site_thermal_states = data['site_thermal_states']
        site_thermal_states[:,n] = site_thermal_state
        np.savez(data_filename, site_histories_sum=site_histories_sum, time=time, site_thermal_states=site_thermal_states, num_realisations=n+1)
    except IOError:
        site_histories_sum = np.zeros((site_energies.size, time.size))
        site_thermal_states = np.zeros((site_energies.size, num_realisations))
        if n == 0: # n should be zero here as it is the first run of the code so check
            site_histories_sum[:,:] += site_history
            site_thermal_states[:,n] = site_thermal_state
            np.savez(data_filename, site_histories_sum=site_histories_sum, time=time, site_thermal_states=site_thermal_states, num_realisations=n+1)
        else:
            print 'n is not zero on first run of the code!'
    

# divide time evolution by number of realisations at the end
# plot disorder averaged time evolution




'''
# calculate rates for PE545 (rates are returned in basis of excitons going lowest to highest in energy)
rates = os.MRT_rate_PE545(site_hamiltonian, reorg_energy1, cutoff_freq1, reorg_energy2, cutoff_freq2, temperature, PE545_mode_params(mode_damping), 10, 10)

# construct Liouvillian from modified Redfield rates (in basis of excitons from lowest to highest in energy)
liouvillian = np.zeros((rates.shape[0], rates.shape[1]))
for i,row in enumerate(rates.T):
    liouvillian[i,i] = -np.sum(row)
liouvillian += rates

# check liouvillian validity
for row in liouvillian.T:
    print np.sum(row)
    
print rates
np.savez('../../data/modified_redfield_test_PE545_data.npz', rates=rates)

# run time evolution for 5ps to try and reproduce plot in supplementary info
duration = 5.
timestep = 0.01
init_dv = np.array([0.35, 0.12, 0.1, 0.1, 0.34, 0.61, 0.46, 0.5])
#init_dv /= np.sum(init_dv)
evals, evecs = utils.sorted_eig(site_hamiltonian) # make sure to return excitons in basis going from lowest to highest energy with sorted_eig
evecs = evecs.T
init_dv = np.diag(np.dot(evecs.T, np.dot(np.diag(init_dv), evecs))) # convert init dv in site basis to exciton basis before time evolution calculation
dv_history = te.liouvillian_time_evolution(init_dv, liouvillian, duration, timestep)

# dm_history gives us exciton populations so need to convert to site populations for plotting

site_history = np.zeros((evals.shape[0], len(dv_history)))
for i,dv in enumerate(dv_history):
    exciton_dm = np.diag(dv)
    site_dm = np.dot(evecs, np.dot(exciton_dm, evecs.T))
    site_history[:,i] = np.diag(site_dm)

time = np.arange(0, duration+timestep, timestep)
for i,row in enumerate(site_history):
    plt.plot(time, row, label=str(i+1))
plt.legend()
plt.show()
    
# include disorder to get better reproduction of plot
# so create new realisation of disorder in site energies, calculate rates and time evolution
# average time evolution of each realisation to get final plot
'''