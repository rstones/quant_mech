'''
Created on 27 Nov 2014

Module to test modified Redfield rates calculations in open_systems module. The specific functions being tested are those that
are copying Ed's Mathematica code.

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

# rates, integrands, time = os.MRT_rate_ed(hamiltonian(10.,100.), reorg_energy, cutoff_freq, temperature, 0, 0.5)
# plt.plot(time, integrands[1])
# plt.show()

# try to reproduce dimer rates using code copied from Mathematica
rates_data = []
       
for i,V in enumerate(coupling_values):
    print 'calculating rates for coupling ' + str(V)
    rates = []
    for delta_E in delta_E_values:
        rates.append(os.MRT_rate_ed(hamiltonian(delta_E, V), reorg_energy, cutoff_freq, temperature, 20)[0,1])
    plt.subplot(1, coupling_values.size, i+1)
    rates_data.append(rates)
    plt.loglog(delta_E_values, np.array(rates)*utils.WAVENUMS_TO_INVERSE_PS, label=V)
     
    # plot extracted data from Ed's thesis
    xdata, ydata = np.loadtxt('../../data/thieved_data'+str(i)+'.txt', delimiter=', ', unpack=True)
    plt.loglog(xdata, ydata, color='red')
    #s = interp.UnivariateSpline(xdata, ydata, k=2, s=None)
    #plt.loglog(xdata, s(xdata), color='red')
     
     
    plt.xlabel(r'$\Delta E$ (cm$^{-1}$)')
    plt.ylabel(r'rate')
    plt.ylim(0.01, 200)
    plt.xlim(5,1000)
    plt.legend()
     
np.savez('../../data/modified_redfield_test_ed_data.npz', delta_E_values=delta_E_values, coupling_values=coupling_values, rates=rates_data)
      
plt.show()    