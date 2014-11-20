'''
Created on 29 Oct 2014

Module to test code that calculates modified Redfield rates

@author: rstones
'''
import numpy as np
import numpy.fft as fft
import scipy.integrate as integrate
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

'''
Code to calculate rates for various coupling strengths and energy gaps between the monomers in a dimer, then saves data to file for plotting
'''
rates_data = []
    
for i,V in enumerate(coupling_values):
    print 'calculating rates for coupling ' + str(V)
    rates = []
    for delta_E in delta_E_values:
        rates.append(os.modified_redfield_relaxation_rates(hamiltonian(delta_E, V), np.array([reorg_energy, reorg_energy]), cutoff_freq, None, temperature, 20)[0,1])
    plt.subplot(1, 3, i+1)
    rates_data.append(rates)
    plt.loglog(delta_E_values, np.array(rates)*utils.WAVENUMS_TO_INVERSE_PS, label=V)
    plt.xlabel(r'$\Delta E$ (cm$^{-1}$)')
    plt.ylabel(r'rate')
    plt.ylim(0.01, 100)
    plt.legend()
      
np.savez('../../data/modified_redfield_test_simps_data.npz', rates=rates_data, delta_E_values=delta_E_values, coupling_values=coupling_values, \
                                                        reorg_energy=reorg_energy, cutoff_freq=cutoff_freq, temperature=temperature)

'''
Code to write/read integrand data from text files, then integrate to find rates and plot rates vs energy gap
(originally to send data to text file so integration could be tested against Mathematica integration routines)
'''
#coupling = 100.

# for i,delta_E in enumerate(delta_E_values):
#     integrand, time = os.modified_redfield_relaxation_rates(hamiltonian(delta_E, coupling), np.array([reorg_energy, reorg_energy]), cutoff_freq, None, temperature, 0)
#     np.savetxt('/home/rstones/numpy_external_data/modified_redfield_integrand'+str(i+1)+'_coupling_'+str(int(coupling))+'.txt', np.real(integrand), fmt='%.12f')
#     np.savetxt('/home/rstones/numpy_external_data/modified_redfield_time.txt', time, fmt='%.12f')
      
# np.savetxt('/home/rstones/numpy_external_data/modified_redfield_delta_E_values.txt', delta_E_values, fmt='%.12f')

# rates = []
# time = np.loadtxt('/home/rstones/numpy_external_data/modified_redfield_time.txt')
# delta_E_values = np.loadtxt('/home/rstones/numpy_external_data/modified_redfield_delta_E_values.txt')
#  
# for i,delta_E in enumerate(delta_E_values):
#     data = np.loadtxt('/home/rstones/numpy_external_data/modified_redfield_integrand'+str(i+1)+'_coupling_'+str(int(coupling))+'.txt')
#     rates.append(2.*integrate.simps(data, time))
#  
# plt.loglog(delta_E_values, 0.06*np.pi*np.array(rates))
# plt.show()
