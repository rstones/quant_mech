'''
Created on 28 Oct 2015

@author: rstones
'''
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def standard_deviation(fwhm):
    return fwhm / (2.*np.sqrt(2.*np.log(2.)))

mean = 1.
FWHM = 0.5
num_realisations = 1000

random.seed()
data_pts_1 = random.normal(mean, standard_deviation(FWHM), num_realisations)
random.seed()
data_pts_2 = random.normal(mean, standard_deviation(FWHM), num_realisations)

plt.plot(np.linspace(0, 1, num_realisations), data_pts_1)
plt.plot(np.linspace(0, 1, num_realisations), data_pts_2)
plt.show()