'''
Script to plot data generated from modified_redfield_test module
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.interpolate as interp
import quant_mech.utils as utils

font = {'size':18}
matplotlib.rc('font', **font)

data = np.load('../../data/modified_redfield_test_simps_data.npz')
rates = data['rates']
delta_E_values = data['delta_E_values']
coupling_values = data['coupling_values']

rates = np.array([rates[0], rates[2], rates[3]])
coupling_values = np.array([20., 100., 500.])

for i,V in enumerate(coupling_values):
    plt.subplot(1, coupling_values.size, i+1)
    plt.loglog(delta_E_values, utils.WAVENUMS_TO_INVERSE_PS*rates[i], label='V = ' + str(V), linewidth=2)
    
    # plot extracted data from Ed's thesis
    xdata, ydata = np.loadtxt('../../data/thieved_data'+str(i)+'.txt', delimiter=', ', unpack=True)
    plt.loglog(xdata, ydata, color='red')
    #s = interp.UnivariateSpline(xdata, ydata, k=2, s=None)
    #plt.loglog(xdata, s(xdata), color='red')

    plt.xlim(5,1000)
    plt.ylim(0.01,200)
    plt.xlabel(r'$\Delta E$ (cm$^{-1}$)')
    plt.ylabel(r'rate (ps$^{-1}$)') if i == 0 else None

plt.show()