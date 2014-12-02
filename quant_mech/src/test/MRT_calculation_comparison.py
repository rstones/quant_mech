'''
Created on 28 Nov 2014

Module to compare two methods of calculating modified Redfield rates from open_systems module, my initial version and the one
translated from Ed's Mathematica code. 

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

time = np.linspace(0,0.5,1000)
lbf_coeffs = os.lbf_coeffs(reorg_energy, cutoff_freq, temperature, None, 0)
site_lbf = os.site_lbf_ed(time, lbf_coeffs)
site_lbf_dot = os.site_lbf_dot_ed(time, lbf_coeffs)
site_lbf_dot_dot = os.site_lbf_dot_dot_ed(time, lbf_coeffs)

'''
# comparison of reorganisation energy calculations
exciton_reorg_energies1 = []
exciton_reorg_energies2 = []
for i,delta_E in enumerate(delta_E_values):
    evals, evecs = utils.sorted_eig(hamiltonian(delta_E, 20.))
    exciton_reorg_energies1.append(os.generalised_exciton_reorg_energy(np.array([evecs[0], evecs[0], evecs[1], evecs[1]]), np.array([reorg_energy, reorg_energy])))
    exciton_reorg_energies2.append(np.sum(evecs[0]**2 * evecs[1]**2) * reorg_energy)
    
plt.plot(delta_E_values, exciton_reorg_energies1, label='1')
plt.plot(delta_E_values, exciton_reorg_energies2, label='2', linewidth=2, ls='--', color='red')
plt.legend()
plt.show()
'''

'''
# comparison of line broadening function calculations
lbf_integral1 = []
lbf_integral2 = []
for i,delta_E in enumerate(delta_E_values):
    evals, evecs = utils.sorted_eig(hamiltonian(delta_E, 20.))
    lbf_integral1.append(integrate.simps(np.exp(-os.generalised_exciton_lbf(np.array([evecs[1], evecs[1], evecs[0], evecs[0]]), np.array([site_lbf, site_lbf]))), time))
    lbf_integral2.append(integrate.simps(np.exp(-np.sum(evecs[1]**2 * evecs[0]**2)*site_lbf), time))
    
plt.plot(delta_E_values, np.real(lbf_integral1), label='1')
plt.plot(delta_E_values, np.real(lbf_integral2), label='2', linewidth=2, ls='--', color='red')
plt.legend()
plt.show()
'''


# comparison of absorption and fluorescene part of integrand
def new_abs_fl(time, omega_ij, c_alphas, c_betas, site_reorg_energy, site_lbf):
    return np.array([np.exp(1.j*omega_ij*t - (np.sum(c_alphas**4) + np.sum(c_betas**4))*(1.j*site_reorg_energy*t + site_lbf[k])) for k,t in enumerate(time)])
'''
abs_fl_integral1 = []
abs_fl_integral2 = []
for i, delta_E in enumerate(delta_E_values):
    evals, evecs = utils.sorted_eig(hamiltonian(delta_E, 20.))
#     exciton0_lbf = np.sum(evecs[0]**4) * site_lbf
#     exciton1_lbf = np.sum(evecs[1]**4) * site_lbf
#     exciton1_reorg_energy = np.sum(evecs[1]**4) * reorg_energy
#     abs_fl_integral1.append(integrate.simps(os.absorption_line_shape(time, evals[0], exciton0_lbf) * os.fluorescence_line_shape(time, evals[1], exciton1_reorg_energy, exciton1_lbf).conj(), time))
    abs_lineshapes, fl_lineshapes, time2 = os.modified_redfield_relaxation_rates(hamiltonian(delta_E, 20.), np.array([reorg_energy, reorg_energy]), cutoff_freq, None, temperature, 0, 0.5)
    abs_fl_integral1.append(integrate.simps(abs_lineshapes[0]*fl_lineshapes[1].conj(), time2))
    abs_fl_integral2.append(integrate.simps(new_abs_fl(time, evals[1]-evals[0], evecs[0], evecs[1], reorg_energy, site_lbf), time))
    
plt.plot(delta_E_values, abs_fl_integral1, label='1')
plt.plot(delta_E_values, abs_fl_integral2, label='2', linewidth=2, ls='--', color='red')
plt.legend()
plt.show()
'''

# comparison of mixing function part of integrand
def new_mixing_function(time, c_alphas, c_betas, site_reorg_energy, g_site, g_site_dot, g_site_dot_dot):
    return np.array([np.exp(2. * np.sum(c_alphas**2 * c_betas**2) * (g_site[k] + 1.j*site_reorg_energy*t)) *
                    ((np.sum(c_alphas**2 * c_betas**2)*g_site_dot_dot[k]) - 
                    ((np.sum(c_alphas * c_betas**3) - np.sum(c_alphas**3 * c_betas))*g_site_dot[k] + 2.j*np.sum(c_betas**3 * c_alphas)*site_reorg_energy)**2) for k,t in enumerate(time)])
'''
mixing_integral1 = []
mixing_integral2 = []
for i,delta_E in enumerate(delta_E_values):
    evals, evecs = utils.sorted_eig(hamiltonian(delta_E, 20.))
    abs_lineshapes, fl_lineshapes, mixing_function, time2 = os.modified_redfield_relaxation_rates(hamiltonian(delta_E, 20.), np.array([reorg_energy, reorg_energy]), cutoff_freq, None, temperature, 0, 0.5)
    mixing_integral1.append(integrate.simps(mixing_function[0,1], time2[:-5]))
    mixing_integral2.append(integrate.simps(new_mixing_function(time, evecs[0], evecs[1], reorg_energy, site_lbf, site_lbf_dot, site_lbf_dot_dot)))
# plt.plot(delta_E_values, mixing_integral1, label='1')
# plt.plot(delta_E_values, mixing_integral2, label='2', linewidth=2, ls='--', color='red')
# plt.legend()
plt.plot(delta_E_values, [np.abs(mixing_integral1[i] - mixing_integral2[i]) for i in range(len(mixing_integral1))])
plt.show()
'''
    
# comparison of full rate calculations
rates1 = []
rates2 = []
rates3 = []

for i,delta_E in enumerate(delta_E_values):
    evals, evecs = utils.sorted_eig(hamiltonian(delta_E, 20.))
    rates, abs_lineshapes, fl_lineshapes, mixing_function, time2 = os.modified_redfield_relaxation_rates(hamiltonian(delta_E, 20.), np.array([reorg_energy, reorg_energy]), cutoff_freq, None, temperature, 0, 0.5)
    rates1.append(2.*np.real(integrate.simps(abs_lineshapes[0][:-5]*fl_lineshapes[1].conj()[:-5]*mixing_function[0,1], time2[:-5])))
    rates2.append(2.*np.real(integrate.simps(new_abs_fl(time, evals[1]-evals[0], evecs[0], evecs[1], reorg_energy, site_lbf)*new_mixing_function(time, evecs[0], evecs[1], reorg_energy, site_lbf, site_lbf_dot, site_lbf_dot_dot), time)))
    rates3.append(rates[0,1])
plt.loglog(delta_E_values, utils.WAVENUMS_TO_INVERSE_PS*np.array(rates1), label='1')
plt.loglog(delta_E_values, utils.WAVENUMS_TO_INVERSE_PS*np.array(rates2), label='2', linewidth=2, ls='--', color='red')
plt.loglog(delta_E_values, utils.WAVENUMS_TO_INVERSE_PS*np.array(rates3), label='3')
plt.legend()
plt.show()


