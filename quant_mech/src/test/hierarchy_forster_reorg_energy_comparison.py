'''
Created on 19 Apr 2016

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils
import quant_mech.open_systems as os
from quant_mech.hierarchy_solver import HierarchySolver
from scipy.optimize import leastsq

from quant_mech import time_utils

from multiprocessing import Pool

reorg_energy_values = np.logspace(0, 3, 40)

forster_rates = np.zeros(reorg_energy_values.size)
HEOM_rates = np.zeros(reorg_energy_values.size)

HEOM_dynamics = []

time_step = 0.0001
duration = 2.
time = np.arange(0, duration+time_step, time_step)

system_hamiltonian = np.array([[100., 20.], [20., 0]])
cutoff_freq = 53.
temperature = 300.
init_state = np.array([[1., 0], [0, 0]])

hs = HierarchySolver(system_hamiltonian, 1., cutoff_freq, temperature)

# error function for least squares fit
def residuals(p, y, pops1, pops2):
        k12, k21 = p
        return y - (-k12*pops1 + k21*pops2)

print '[Calculating Forster rates...]'
print 'Started at: ' + str(time_utils.getTime())
for i,v in enumerate(reorg_energy_values):
    #print 'Calculating rates for reorg energy: ' + str(v)
    forster_time = np.linspace(0, 20., 16000)
    lbf = os.site_lbf_ed(forster_time, os.lbf_coeffs(v, cutoff_freq, temperature, None, 5))
    forster_rates[i] = os.forster_rate(100., 0, v, v, lbf, lbf, 0, 0, np.array([1., 0]), np.array([0, 1.]), system_hamiltonian, forster_time)
print 'Finished at: ' + str(time_utils.getTime())
    
def calculate_HEOM(reorg_energy):
    system_hamiltonian = np.array([[100., 20.], [20., 0]])
    cutoff_freq = 53.
    temperature = 300.
    init_state = np.array([[1., 0], [0, 0]])
    
    time_step = 0.00005
    duration = 5.

    hs = HierarchySolver(system_hamiltonian, reorg_energy, cutoff_freq, temperature)
    HEOM_history, time = hs.hierarchy_time_evolution(init_state, 18, time_step, duration)
    return HEOM_history

print '[Calculating HEOM in parallel...]'
print 'Started at: ' + str(time_utils.getTime())
pool = Pool(processes=8)
HEOM_dynamics = np.array(pool.map(calculate_HEOM, reorg_energy_values))
print 'Finished at: ' + str(time_utils.getTime())

print '[Fitting rates to HEOM dynamics...]'
print 'Started at: ' + str(time_utils.getTime())
for i,v in enumerate(reorg_energy_values):
#     hs.E_reorg = v
#     HEOM_history, time = hs.hierarchy_time_evolution(init_state, 15, time_step, duration)
#     HEOM_dynamics.append(HEOM_history)
    HEOM_history = HEOM_dynamics[i]
    pops1 = np.array([dm[0,0] for dm in HEOM_history], dtype='float64')
    pops2 = np.array([dm[1,1] for dm in HEOM_history], dtype='float64')
    y = np.array(utils.differentiate_function(pops1, time), dtype='float64')
 
    rates0 = np.array([0.1 ,0.1])
    result = leastsq(residuals, rates0, args=(y, pops1, pops2))
    rates = result[0]
    HEOM_rates[i] = rates[0]
print 'Finished at: ' + str(time_utils.getTime())
  
np.savez('../../data/HEOM_forster_rates_reorg_energy_5ps.npz', forster_rates=forster_rates, HEOM_rates=HEOM_rates, HEOM_dynamics=HEOM_dynamics, \
         time=time, reorg_energy_values=reorg_energy_values)
    
plt.semilogx(reorg_energy_values, forster_rates*utils.WAVENUMS_TO_INVERSE_PS, label='Forster')
plt.semilogx(reorg_energy_values, HEOM_rates, label='HEOM')
plt.show()