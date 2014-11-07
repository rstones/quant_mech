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
delta_E_values = np.linspace(0,e1,100) # wavenumbers
coupling_values = np.array([20., 100., 500.]) # wavenumbers

def hamiltonian(delta_E, V):
    return np.array([[e1, V],
                    [V, e1-delta_E]])

reorg_energy = 100. # wavenumbers
cutoff_freq = 53. # wavenumbers
temperature = 300.

rates_data = []

# lbfs, time = os.modified_redfield_relaxation_rates(hamiltonian(0, 20.), np.array([reorg_energy, reorg_energy]), cutoff_freq, None, temperature, 0)
# plt.plot(time, lbfs[0])
# plt.plot(time[:-2], utils.differentiate_function(utils.differentiate_function(lbfs[0], time)[:-2], time))
# # plt.plot(time, np.exp(-utils.differentiate_function(lbfs[0], time)), label='0')
# # plt.plot(time, np.exp(-lbfs[1]), label='1')
# # plt.plot(time, np.exp(-lbfs[2]), label='2')
# # plt.plot(time, np.exp(-lbfs[3]), label='3')
# # plt.plot(time, np.exp(-lbfs[4]), label='4')
# # plt.plot(time, np.exp(-lbfs[5]), label='5')
# plt.legend()

for i,V in enumerate(coupling_values):
    print 'calculating rates for coupling ' + str(V)
    rates = []
    for delta_E in delta_E_values:
        rates.append(os.modified_redfield_relaxation_rates(hamiltonian(delta_E, V), np.array([reorg_energy, reorg_energy]), cutoff_freq, None, temperature, 0)[0,1])
    plt.subplot(1, 3, i+1)
    rates_data.append(rates)
    plt.loglog(delta_E_values, np.array(rates)*utils.WAVENUMS_TO_INVERSE_PS, label=V)
    plt.xlabel(r'$\Delta E$ (cm$^{-1}$)')
    plt.ylabel(r'rate')
    plt.ylim(0.01, 100)
    plt.legend()
 
np.savez('../../data/modified_redfield_test_data.npz', rates=rates_data, delta_E_values=delta_E_values, coupling_values=coupling_values, \
                                                        reorg_energy=reorg_energy, cutoff_freq=cutoff_freq, temperature=temperature)

plt.show()