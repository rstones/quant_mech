'''
Created on 21 Nov 2014

Module to test modified Redfield code from open_systems module. Attempts to reproduce parts of figure 3 from Physical Origins and Models of Energy
Transfer in Photosynthetic Light-Harvesting by Novoderezhkin and van Grondelle (2010)

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils
import quant_mech.open_systems as os

def hamiltonian(delta_E, V):
    return np.array([[delta_E/2., V],
                    [V, -delta_E/2.]])
    
def PSII_mode_params(damping):
        # freq and damping constant in wavenumbers
        return np.array([(97.,0.0371,damping),
                         (138.,0.0455,damping),
                         (213.,0.0606,damping),
                         (260.,0.0539,damping),
                         (298.,0.0488,damping),
                         (342.,0.0438,damping),
                         (388.,0.0202,damping),
                         (425.,0.0168,damping),
                         (518.,0.0303,damping),
                         (546.,0.0030,damping),
                         (573.,0.0094,damping),
                         (585.,0.0034,damping),
                         (604.,0.0034,damping),
                         (700.,0.005,damping),
                         (722.,0.0074,damping),
                         (742.,0.0269,damping),
                         (752.,0.0219,damping),
                         (795.,0.0077,damping),
                         (916.,0.0286,damping),
                         (986.,0.0162,damping),
                         (995.,0.0293,damping),
                         (1052.,0.0131,damping),
                         (1069.,0.0064,damping),
                         (1110.,0.0192,damping),
                         (1143.,0.0303,damping),
                         (1181.,0.0179,damping),
                         (1190.,0.0084,damping),
                         (1208.,0.0121,damping),
                         (1216.,0.0111,damping),
                         (1235.,0.0034,damping),
                         (1252.,0.0051,damping),
                         (1260.,0.0064,damping),
                         (1286.,0.0047,damping),
                         (1304.,0.0057,damping),
                         (1322.,0.0202,damping),
                         (1338.,0.0037,damping),
                         (1354.,0.0057,damping),
                         (1382.,0.0067,damping),
                         (1439.,0.0067,damping),
                         (1487.,0.0074,damping),
                         (1524.,0.0067,damping),
                         (1537.,0.0222,damping),
                         (1553.,0.0091,damping),
                         (1573.,0.0044,damping),
                         (1580.,0.0044,damping),
                         (1612.,0.0044,damping),
                         (1645.,0.0034,damping),
                         (1673.,0.001,damping)])
        
delta_E_values = np.linspace(0,2000,200) # wavenumbers
coupling_values = np.array([225., 100., 55.]) # wavenumbers
temperature = 77. # Kelvin
reorg_energy = 35. # wavenumbers
cutoff_freq = 40.
mode_damping = 3.

rates = []

# time, integrand = os.modified_redfield_relaxation_rates(hamiltonian(100., coupling_values[0]), np.array([reorg_energy, reorg_energy]), cutoff_freq, \
#                                                         PSII_mode_params(mode_damping), temperature, 20)
# 
# print time.shape
# print integrand.shape
# plt.plot(np.real(integrand), time)
# plt.show()
                                                        
# x = np.linspace(0,100,100)
# y = x**2
# print x.shape
# print y.shape
# plt.plot(x,y)
# plt.show()


print 'Calculating rates with high energy modes....'
for i,delta_E in enumerate(delta_E_values):
    rates.append(os.modified_redfield_relaxation_rates(hamiltonian(delta_E, coupling_values[0]), np.array([reorg_energy, reorg_energy]), cutoff_freq, \
                                                       PSII_mode_params(mode_damping), temperature, 10)[0,1])
   
np.savez('../../data/modified_redfield_test_high_energy_modes_data.npz', delta_E_values=delta_E_values, rates=rates)
plt.plot(delta_E_values, -utils.WAVENUMS_TO_INVERSE_PS*np.array(rates))
plt.show()

# data = np.load('../../data/modified_redfield_test_high_energy_modes_data.npz')
# rates = data['rates']
# delta_E_values = data['delta_E_values']
# plt.plot(delta_E_values, -utils.WAVENUMS_TO_INVERSE_PS* rates)
# plt.show()