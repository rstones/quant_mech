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

# rates, integrands, time = os.MRT_rate_ed(hamiltonian(10.,500.), reorg_energy, cutoff_freq, temperature, None, 0, 60.0)
# plt.plot(time, integrands[0,1])
# plt.show()

# try to reproduce dimer rates using code copied from Mathematica
rates_data = []
time_interval = 0.5
time = np.linspace(0, time_interval, int(time_interval*2000))
lbf_coeffs = os.lbf_coeffs(reorg_energy, cutoff_freq, temperature, None, 0)
g_site = os.site_lbf_ed(time, lbf_coeffs)
g_site_dot = os.site_lbf_dot_ed(time, lbf_coeffs)
g_site_dot_dot = os.site_lbf_dot_dot_ed(time, lbf_coeffs)
scaling = 1.

for i,V in enumerate(coupling_values):
    print 'calculating rates for coupling ' + str(V)
    rates = []
    for delta_E in delta_E_values:
        evals, evecs = utils.sorted_eig(hamiltonian(delta_E, V))
        print evals
        exciton_reorg_energies = np.array([os.exciton_reorg_energy(exciton, [reorg_energy, reorg_energy]) for exciton in evecs])
        print exciton_reorg_energies
        rates.append(os.modified_redfield_rates_general(evals, evecs, np.array([g_site, scaling*g_site]), np.array([g_site_dot,scaling*g_site_dot]), np.array([g_site_dot_dot,scaling*g_site_dot_dot]), np.array([reorg_energy,scaling*reorg_energy]), temperature, time)[0][0,1])
        #rates.append(os.MRT_rates(hamiltonian(delta_E, V), np.array([reorg_energy, reorg_energy]), cutoff_freq, temperature, None)[0,1])
    plt.subplot(1, coupling_values.size, i+1)
    rates_data.append(rates)
    plt.loglog(delta_E_values, np.array(rates)*utils.WAVENUMS_TO_INVERSE_PS, label=V)
       
    # plot extracted data from Ed's thesis
#     xdata, ydata = np.loadtxt('../../data/thieved_data'+str(i)+'.txt', delimiter=', ', unpack=True)
#     plt.loglog(xdata, ydata, color='red')
    #s = interp.UnivariateSpline(xdata, ydata, k=2, s=None)
    #plt.loglog(xdata, s(xdata), color='red')
       
    plt.xlabel(r'$\Delta E$ (cm$^{-1}$)')
    plt.ylabel(r'rate')
    plt.ylim(0.01, 200)
    plt.xlim(5,1000)
    plt.legend()
       
#np.savez('../../data/modified_redfield_test_ed_data.npz', delta_E_values=delta_E_values, coupling_values=coupling_values, rates=rates_data)
        
plt.show()    