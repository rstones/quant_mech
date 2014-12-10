'''
Created on 10 Dec 2014

@author: rstones
'''
import numpy as np
import quant_mech.utils as utils
import quant_mech.open_systems as os
import matplotlib.pyplot as plt

reorg_energy = 37.
cutoff_freq = 30.
temperature = 77.
mode_damping = 3.

'''
Taken from Table 1 in Energy-Transfer Dynamics in the LHCII Complex of Higher Plants: Modified Redfield
Approach by Novoderezhkin, Palacios, van Amerongen and van Grondelle, J. Phys. Chem. B 2004 
'''
def LHCII_mode_params(damping):
        # freq and damping constant in wavenumbers
        return np.array([(97.,0.02396,damping),
                         (138.,0.02881,damping),
                         (213.,0.03002,damping),
                         (260.,0.02669,damping),
                         (298.,0.02669,damping),
                         (342.,0.06035,damping),
                         (388.,0.02487,damping),
                         (425.,0.01486,damping),
                         (518.,0.03942,damping),
                         (546.,0.00269,damping),
                         (573.,0.00849,damping),
                         (585.,0.00303,damping),
                         (604.,0.00194,damping),
                         (700.,0.00197,damping),
                         (722.,0.00394,damping),
                         (742.,0.03942,damping),
                         (752.,0.02578,damping),
                         (795.,0.00485,damping),
                         (916.,0.02123,damping),
                         (986.,0.01031,damping),
                         (995.,0.02274,damping),
                         (1052.,0.01213,damping),
                         (1069.,0.00636,damping),
                         (1110.,0.01122,damping),
                         (1143.,0.04094,damping),
                         (1181.,0.01759,damping),
                         (1190.,0.00667,damping),
                         (1208.,0.01850,damping),
                         (1216.,0.01759,damping),
                         (1235.,0.00697,damping),
                         (1252.,0.00636,damping),
                         (1260.,0.00636,damping),
                         (1286.,0.00454,damping),
                         (1304.,0.00576,damping),
                         (1322.,0.03032,damping),
                         (1338.,0.00394,damping),
                         (1354.,0.00576,damping),
                         (1382.,0.00667,damping),
                         (1439.,0.00667,damping),
                         (1487.,0.00788,damping),
                         (1524.,0.00636,damping),
                         (1537.,0.02183,damping),
                         (1553.,0.00909,damping),
                         (1573.,0.00454,damping),
                         (1580.,0.00454,damping),
                         (1612.,0.00454,damping),
                         (1645.,0.00363,damping),
                         (1673.,0.00097,damping)])
        

time = np.linspace(0,0.1,1000)
lbf_coeffs = os.lbf_coeffs(reorg_energy, cutoff_freq, temperature, LHCII_mode_params(mode_damping), 20)
lbf = os.site_lbf(time, lbf_coeffs)
plt.plot(time, np.real(lbf), label=r'$\mathcal{R}$e g(t)')
plt.plot(time, np.imag(lbf), label=r'$\mathcal{I}$m g(t)')
plt.xlim(0,0.1)
plt.legend()
plt.show()
        