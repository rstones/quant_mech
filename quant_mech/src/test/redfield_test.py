'''
Created on 7 Jun 2016

Test Redfield tensor code using a dimer system

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils
import quant_mech.open_systems as os

delta_E_values = np.logspace(0, 3, 100)
coupling_values = np.array([20., 100., 500.])


def hamiltonian(delta_E, coupling):
    return np.array([[delta_E/2., coupling],[coupling, -delta_E/2.]])

reorg_energy = 100.
cutoff_freq = 53.08
temperature = 300.

'''
Compute the time evolution of density matrix under influence of the Redfield tensor
'''
H = hamiltonian(100, 100.)
evalues,evectors = np.linalg.eig(H)
evectors = evectors.T
evalues = evalues * utils.WAVENUMS_TO_INVERSE_PS
redfield_tensor = os.redfield_tensor_identical_drude(H, reorg_energy, cutoff_freq, temperature) * utils.WAVENUMS_TO_INVERSE_PS
print os.redfield_tensor_identical_drude(H, reorg_energy, cutoff_freq, temperature)

def drho_by_dt(t, rho):
    system_dimension = int(np.sqrt(rho.size))
    omega_matrix = np.zeros((system_dimension, system_dimension))
    for i in range(system_dimension):
        for j in range(system_dimension):
            if i != j:
                omega_matrix[i,j] = evalues[i] - evalues[j]
    return -1.j*omega_matrix.flatten()*rho - np.dot(redfield_tensor, rho)
     
#from scipy.integrate import odeint
#dynamics = odeint(drho_by_dt, np.array([1., 0, 0, 0], dtype='complex128'), np.linspace(0,5,1000, dtype='complex128'), args=(np.complex128(evalues)*utils.WAVENUMS_TO_INVERSE_PS, np.complex128(redfield_tensor)*utils.WAVENUMS_TO_INVERSE_PS))
 
time_step = 0.0001
duration = 1.
 
print np.dot(evectors, np.dot(np.array([[1.,0],[0,0]]), evectors.T))
 
dynamics = os.RK4(drho_by_dt, np.array([0.2763932, -0.4472136, -0.4472136, 0.7236068]), duration, time_step)
print dynamics.shape
 
# transform dynamics to site basis before plotting
site_density_matrices = np.zeros((int(duration/time_step), 2, 2))
for i,dv in enumerate(dynamics):
    dv.shape = 2,2
    site_dm = np.dot(evectors, np.dot(dv, evectors.T))
    site_density_matrices[i] = site_dm
 
plt.plot(np.arange(0, duration, time_step), site_density_matrices.T[1,1,:])
plt.ylim(0.3,1.0)
plt.show()

'''
Compute transfer rates between excitons as function of reorganisation energy
'''
# reorg_energy_values = np.logspace(0, 3, 100)
# rates = np.zeros(reorg_energy_values.size)
#    
# for i,E in enumerate(reorg_energy_values):
#     rates[i] = os.redfield_tensor_identical_drude(hamiltonian(100., 20.), E, cutoff_freq, temperature)[0,0]
#        
# plt.semilogx(reorg_energy_values, rates*utils.WAVENUMS_TO_INVERSE_PS)
# plt.show()


# H = hamiltonian(0, 20.)
# system_dimension = H.shape[0]
# evalues,evectors = utils.sorted_eig(H)
# 
# site_bath_coupling_matrices = np.array([[[1., 0],[0, 0]],[[0, 0],[0, 1.]]])
# 
# def site_correlation_function(freq):
#     if freq != 0 :
#         return os.overdamped_BO_spectral_density(np.abs(freq), reorg_energy, cutoff_freq) * np.abs(utils.planck_distribution(freq, temperature))
#     else:
#         return 0
#   
# print site_correlation_function(-40.)
# 
# print os.redfield_tensor_identical_drude(hamiltonian(0, 20.), reorg_energy, cutoff_freq, temperature)
# 
# print os.exciton_relaxation_rates(hamiltonian(0,20.), reorg_energy, cutoff_freq, os.bo_spectral_density, temperature, None)
  
# def Gamma(a, b, c, d):
#     return np.real(np.sum([np.dot(evectors[a], np.dot(site_bath_coupling_matrices[n], evectors[b])) \
#                             * np.dot(evectors[c], np.dot(site_bath_coupling_matrices[n], evectors[d])) for n in range(system_dimension)]) \
#                             * site_correlation_function(evalues[d]-evalues[c]))
#      
# def tensor_element(a, b, c, d):
#     element = 0
#     if a == c:
#         for e in range(system_dimension):
#             element += Gamma(b, e, e, d)
#     if b == d:
#         for e in range(system_dimension):
#             element += Gamma(a, e, e ,c)
#     element -= Gamma(c, a, b, d) + Gamma(d, b, a, c)
#     return element
# 
# print tensor_element(0,0,0,0)

# evalues,evectors = utils.sorted_eig(hamiltonian(0,20.))
# print evalues
# print evectors
#  
# rates1 = np.zeros((coupling_values.size, delta_E_values.size))
# rates2 = np.zeros((coupling_values.size, delta_E_values.size))
# for i,J in enumerate(coupling_values):
#     for j,E in enumerate(delta_E_values):
#         rates1[i,j] = os.redfield_tensor_identical_drude(hamiltonian(E,J), reorg_energy, cutoff_freq, temperature)[3,3]
#         rates2[i,j] = os.exciton_relaxation_rates(hamiltonian(E,J), reorg_energy, cutoff_freq, os.bo_spectral_density, temperature, None)[1,0]
#      
# for i in range(coupling_values.size):
#     plt.subplot(1,3,i+1)
#     plt.loglog(delta_E_values, rates1[i]*utils.WAVENUMS_TO_INVERSE_PS, label=coupling_values[i])
#     plt.loglog(delta_E_values, rates2[i]*utils.WAVENUMS_TO_INVERSE_PS, label=coupling_values[i], ls='--')
#     plt.xlim(5,1000)
#     plt.ylim(0.01,200)
# plt.show()
