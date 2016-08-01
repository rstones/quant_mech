'''
Created on 28 Apr 2016

@author: rstones
'''
import numpy as np
from multiprocessing import Pool
from scipy.optimize import leastsq
import quant_mech.utils as utils
import quant_mech.open_systems as os
from quant_mech.hierarchy_solver import HierarchySolver
import matplotlib.pyplot as plt
import quant_mech.time_utils as time_utils

# define model
def hamiltonian(energy_gap, coupling):
    return np.array([[energy_gap/2., coupling],
                     [coupling, -energy_gap/2.]])
    
reorg_energy = 100.
cutoff_freq = 53.
temperature = 300.

energy_gap_values = np.logspace(0, 3, 10)
coupling = 20.

hamiltonians = [hamiltonian(energy_gap, coupling) for energy_gap in energy_gap_values]

print 'Calculating line broadening function at ' + str(time_utils.getTime())
time = np.linspace(0, 10, 1000000)
num_expansion_terms = 500
lbf_fn = '../../data/hierarchy_MRT_comparison_lbf_data.npz'
try:
    data = np.load(lbf_fn)
    lbf = data['lbf']
    lbf_dot = data['lbf_dot']
    lbf_dot_dot = data['lbf_dot_dot']
except IOError:
    lbf_coeffs = os.lbf_coeffs(reorg_energy, cutoff_freq, temperature, None, num_expansion_terms)
    lbf = os.site_lbf_ed(time, lbf_coeffs)
    lbf_dot = os.site_lbf_dot_ed(time, lbf_coeffs)
    lbf_dot_dot = os.site_lbf_dot_ed(time, lbf_coeffs)
    np.savez(lbf_fn, lbf=lbf, lbf_dot=lbf_dot, lbf_dot_dot=lbf_dot_dot, time=time)

print 'Calculating MRT rates at ' + str(time_utils.getTime())
MRT_rates = np.zeros(energy_gap_values.size)
for i,H in enumerate(hamiltonians):
    evals,evecs = utils.sorted_eig(H)
    MRT_rates[i] = os.modified_redfield_rates_general(evals, evecs, lbf, lbf_dot, lbf_dot_dot, reorg_energy, temperature, time)[0][0,1]

# define function for HEOM calculation pool
def HEOM_calculation(hamiltonian):
    reorg_energy = 100.
    cutoff_freq = 53.
    temperature = 300.
    init_state = np.array([[1., 0], [0, 0]])
    
    duration = 5.
    time_step = 0.00005
    
    hs = HierarchySolver(hamiltonian, reorg_energy, cutoff_freq, temperature)
    HEOM_history, time = hs.hierarchy_time_evolution(init_state, 18, time_step, duration)
    return HEOM_history

print 'Calculating HEOM dynamics at ' + str(time_utils.getTime())
pool = Pool(processes=4)
HEOM_dynamics = np.array(pool.map(HEOM_calculation, hamiltonians))

np.savez('../../data/hierarchy_MRT_comparison_dynamics_data.npz', HEOM_dynamics=HEOM_dynamics)

print 'Fitting rates to HEOM dynamics at ' + str(time_utils.getTime())

def residuals(p, y, pops1, pops2):
        k12, k21 = p
        return y - (-k12*pops1 + k21*pops2)

HEOM_rates = np.zeros(energy_gap_values.size)
for i,deltaE in enumerate(energy_gap_values):
    HEOM_history = HEOM_dynamics[i]
    pops1 = np.array([dm[0,0] for dm in HEOM_history], dtype='float64')
    pops2 = np.array([dm[1,1] for dm in HEOM_history], dtype='float64')
    y = np.array(utils.differentiate_function(pops1, time), dtype='float64')
 
    rates0 = np.array([0.1 ,0.1])
    result = leastsq(residuals, rates0, args=(y, pops1, pops2))
    rates = result[0]
    HEOM_rates[i] = rates[0]

# save data
np.savez('../../data/hierarchy_MRT_comparison_rate_data.npz', MRT_rates=MRT_rates, HEOM_rates=HEOM_rates, energy_gap_values=energy_gap_values)

print 'Plotting at ' + str(time_utils.getTime())
plt.semilogx(energy_gap_values, MRT_rates, label='MRT')
plt.semilogx(energy_gap_values, HEOM_rates, label='HEOM')
plt.legend().draggable()
plt.show()