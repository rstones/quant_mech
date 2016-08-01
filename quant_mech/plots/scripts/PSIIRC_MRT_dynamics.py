import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

data = np.load('../../data/PSIIRC_MRT_dynamics_vary_cutoff_freq.npz')
dynamics = data['dynamics']
cutoff_freqs = data['cutoff_freqs']
time = data['dynamics_time']

for i,v in enumerate(cutoff_freqs):
    plt.subplot(1,3,i+1)
    for j in range(6):
        plt.plot(time, dynamics[i].T[j], linewidth=2, label=j)
    plt.text(0.2, 0.92, r'$\Omega_c = '+str(int(v))+'cm^{-1}$')
    plt.xlabel('time (ps)')
    plt.xlim(0,3.5)

plt.subplot(131)
plt.ylabel('population')

plt.legend().draggable()
plt.show()