'''
Created on 9 Dec 2014

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import quant_mech.utils as utils
import quant_mech.open_systems as os


def LHCII_monomer_exciton_hamiltonian():
    return np.diag(np.array([14699., 14751., 14804., 14858., 14918., 14952., 14992., 15022., 15210., 15306., 15363., 15416., 15456., 15512.]))

# amplitudes defined across rows in basis
# {b601, a602, a603, a604, b605, b606, b607, b608, b609, a610, a611, a612, a613, a614}
def LHCII_monomer_eigenvectors():
    return np.sqrt(np.array([[0.0004, 0.1648, 0.0114, 0, 0, 0, 0, 0.0029, 0.0006, 0.5015, 0.1031, 0.1194, 0.0882, 0.0071],
                     [0.0005, 0.1731, 0.0251, 0.0001, 0, 0, 0, 0.0013, 0.0012, 0.2261, 0.2015, 0.2331, 0.1208, 0.0166],
                     [0.0006, 0.2259, 0.0592, 0.0112, 0, 0.0004, 0.0001, 0.0010, 0.0025, 0.1512, 0.1201, 0.1413, 0.2399, 0.0460],
                     [0.0006, 0.1996, 0.1109, 0.0418, 0, 0.0016, 0.0002, 0.0005, 0.0042, 0.0753, 0.0619, 0.0644, 0.2804, 0.1580],
                     [0.0003, 0.0719, 0.3365, 0.2465, 0, 0.0105, 0.0014, 0.0001, 0.0121, 0.0207, 0.0421, 0.0332, 0.0702, 0.1537],
                     [0.0006, 0.0627, 0.1741, 0.1965, 0, 0.0086, 0.0012, 0, 0.0061, 0.0090, 0.1074, 0.1018, 0.0713, 0.2602],
                     [0.0007, 0.0457, 0.1394, 0.2346, 0.0224, 0.0113, 0.0017, 0.0001, 0.0050, 0.0060, 0.1627, 0.1345, 0.0586, 0.1767],
                     [0.0010, 0.0482, 0.0969, 0.2190, 0.0168, 0.0126, 0.0042, 0, 0.0034, 0.0035, 0.1944, 0.1665, 0.0647, 0.1678],
                     [0.1913, 0.0042, 0.0129, 0.0009, 0.5280, 0.0256, 0.1296, 0.0278, 0.0497, 0.0002, 0.0055, 0.0052, 0.0053, 0.0132],
                     [0.3209, 0.0012, 0.0039, 0.0008, 0.1742, 0.0317, 0.2442, 0.1267, 0.0947, 0.0008, 0.0003, 0.0001, 0.0001, 0],
                     [0.2712, 0.0009, 0.0057, 0.0013, 0.1033, 0.0449, 0.2203, 0.1986, 0.1516, 0.0012, 0.0002, 0.0001, 0.0001, 0],
                     [0.0827, 0.0003, 0.0104, 0.0044, 0.0734, 0.0947, 0.1563, 0.2770, 0.2983, 0.0016, 0, 0, 0, 0],
                     [0.0932, 0.0004, 0.0078, 0.0120, 0.0531, 0.2343, 0.1349, 0.2361, 0.2262, 0.0014, 0.0001, 0, 0, 0],
                     [0.0354, 0.0003, 0.0048, 0.0303, 0.0284, 0.5232, 0.1052, 0.1272, 0.1438, 0.0008, 0, 0, 0, 0]]))

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
        
# normalise eigenvectors
evectors = LHCII_monomer_eigenvectors()
for i in range(evectors.shape[0]):
    evectors[i] /= np.sum(evectors[i])
evectors = evectors.T
        
# run check on monomer eigenvectors that they are normalised
for row in evectors:
    row_sum = np.sum(row)
    if 0.99999 > row_sum > 1.00001:
        print "row not equal to 1!"
        print np.sum(row)
        
# define parameters including site reorg energies (there is a scaling for b chlorophylls relative to a chlorophylls)
temperature = 77.
site_reorg_energy = 37.
cutoff_freq = 30.
mode_damping = 3.
chlB_reorg_energy_scaling = 1.15 # can be in range from 1.15 - 1.35
# {b601, a602, a603, a604, b605, b606, b607, b608, b609, a610, a611, a612, a613, a614}
site_reorg_energies = np.array([chlB_reorg_energy_scaling*site_reorg_energy, site_reorg_energy, site_reorg_energy, site_reorg_energy, \
                                chlB_reorg_energy_scaling*site_reorg_energy, chlB_reorg_energy_scaling*site_reorg_energy, chlB_reorg_energy_scaling*site_reorg_energy, \
                                chlB_reorg_energy_scaling*site_reorg_energy, chlB_reorg_energy_scaling*site_reorg_energy, site_reorg_energy,\
                                 site_reorg_energy, site_reorg_energy, site_reorg_energy, site_reorg_energy])
        
# calculate site Hamiltonian
site_hamiltonian = np.dot(evectors, np.dot(LHCII_monomer_exciton_hamiltonian(), np.linalg.inv(evectors)))

# check evecs calculated from this site hamiltonian match those returned from LHCII_monomer_eigenvectors and that hamiltonian is Hermitian etc....
evals, evecs = np.linalg.eig(site_hamiltonian)
print evectors
print evecs

print np.dot(np.linalg.inv(evecs), np.dot(site_hamiltonian, evecs))

# calculate modified Redfield rates between excitons
#rates = os.MRT_rate_ed(site_hamiltonian, site_reorg_energies, cutoff_freq, temperature, LHCII_mode_params(mode_damping), 10, 5)

# construct Liouvillian


# calculate transient dynamics starting from some initial state to see if it agrees with dynamics in J. Phys. Chem. B, Vol. 109, No. 20, 2005
