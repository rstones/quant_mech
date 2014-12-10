'''
Created on 10 Dec 2014

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils
import quant_mech.open_systems as os
import quant_mech.time_evolution as te

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
