'''
Created on 19 Dec 2014

Module to test Forster rate calculation in open_systems module

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import quant_mech.open_systems as os
import quant_mech.utils as utils

e1 = 1000.
delta_E_values = np.logspace(0,3.,100) # wavenumbers
coupling_values = np.array([20., 100., 500.]) # wavenumbers
 
def hamiltonian(delta_E, V):
    return np.array([[e1, V],
                    [V, e1-delta_E]])
 
reorg_energy = 100. # wavenumbers
cutoff_freq = 53. # wavenumbers
temperature = 300.

# H = hamiltonian(100., 20.)
# time_interval = 1.
# time = np.linspace(0, time_interval, time_interval*256000)
# site_lbf = os.site_lbf(time, os.lbf_coeffs(reorg_energy, cutoff_freq, temperature, None, 20))
# plt.plot(time, np.exp(-site_lbf))
# rate, integrand = os.forster_rate(H[0,0], H[1,1], reorg_energy, reorg_energy, site_lbf, site_lbf, 0, 0, np.array([1.,0]), np.array([0,1.]), H, time)
# print rate
# plt.plot(time, integrand)
# plt.show()

# E1 = 1000.
# E2 = 0
# E_reorg1 = 100.
# E_reorg2 = 100.
# 
# def integrand_func(t):
#     lbf = 0
#     for i,time_pt in enumerate(time):
#         if time_pt - 0.000005 < t < time_pt + 0.000005:
#             lbf = site_lbf[i]
#             break
#     return np.exp(1.j*(E1-E2)*t) * np.exp(-1.j*(E_reorg1+E_reorg2)*t) * np.exp(-lbf-lbf)
# 
# print 2. * integrate.simps(np.real(np.array([integrand_func(t) for t in time])))
# 
# plt.plot(time, np.array([integrand_func(t) for t in time]))
# plt.show()

# try to reproduce dimer rates using code copied from Mathematica
rates_data = []
time_interval = 20
time = np.linspace(0, time_interval, time_interval*16000)
site_lbf = os.site_lbf_ed(time, os.lbf_coeffs(reorg_energy, cutoff_freq, temperature, None, 20))
for i,V in enumerate(coupling_values):
    rates = []
    for delta_E in delta_E_values:
        H = hamiltonian(delta_E, V)
        rates.append(os.forster_rate(H[0,0], H[1,1], reorg_energy, reorg_energy, site_lbf, site_lbf, 0, 0, np.array([1.,0]), np.array([0,1.]), H, time))
    rates_data.append(rates)
     
for i,rates in enumerate(rates_data):
    plt.subplot(1,3,i+1)
    plt.loglog(delta_E_values, utils.WAVENUMS_TO_INVERSE_PS*np.array(rates))
    plt.ylim(0.01, 200)
    plt.xlim(5,1000)
plt.show()

    
    
    