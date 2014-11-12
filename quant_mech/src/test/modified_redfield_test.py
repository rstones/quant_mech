'''
Created on 29 Oct 2014

Module to test code that calculates modified Redfield rates

@author: rstones
'''
import numpy as np
import numpy.fft as fft
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

a = os.modified_redfield_relaxation_rates(hamiltonian(100., 20.), np.array([reorg_energy, reorg_energy]), cutoff_freq, None, temperature, 20)

# rates_data = []
#    
# for i,V in enumerate(coupling_values):
#     print 'calculating rates for coupling ' + str(V)
#     rates = []
#     for delta_E in delta_E_values:
#         rates.append(os.modified_redfield_relaxation_rates(hamiltonian(delta_E, V), np.array([reorg_energy, reorg_energy]), cutoff_freq, None, temperature, 20)[0,1])
#     plt.subplot(1, 3, i+1)
#     rates_data.append(rates)
#     plt.loglog(delta_E_values, np.array(rates)*utils.WAVENUMS_TO_INVERSE_PS, label=V)
#     plt.xlabel(r'$\Delta E$ (cm$^{-1}$)')
#     plt.ylabel(r'rate')
#     plt.ylim(0.01, 100)
#     plt.legend()
#      
# np.savez('../../data/modified_redfield_test_quad_data.npz', rates=rates_data, delta_E_values=delta_E_values, coupling_values=coupling_values, \
#                                                         reorg_energy=reorg_energy, cutoff_freq=cutoff_freq, temperature=temperature)

# quad_data = np.load('../../data/modified_redfield_test_quad_data.npz')
# quad_rates = quad_data['rates']
# quad_delta_E_values = quad_data['delta_E_values']
# 
# simps_data = np.load('../../data/modified_redfield_test_simps_data.npz')
# simps_rates = simps_data['rates']
# simps_delta_E_values = simps_data['delta_E_values']
# 
# for i in range(coupling_values.shape[0]):
#     plt.subplot(1,3,i+1)
#     plt.loglog(quad_delta_E_values, quad_rates[i]*utils.WAVENUMS_TO_INVERSE_PS)
#     plt.loglog(simps_delta_E_values, simps_rates[i]*utils.WAVENUMS_TO_INVERSE_PS)
#     plt.ylim(0.01, 200)
#     plt.xlim(5,1000)
#  
# plt.show()
