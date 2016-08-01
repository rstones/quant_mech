import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, linewidth=100, suppress=True)

data = np.load('../../data/PSIIRC_ChlD1_pathway_dynamics_77K__incorrect_ordering_data.npz')
population_dynamics_realisations = data['population_dynamics_realisations']
dynamics_time = data['dynamics_time']
num_realisations = data['num_realisations']
liouvillians = data['liouvillians']

print num_realisations

site_init_state = np.array([0.07, 0.14, 0.79, 0.4, 0.58, 0.56, 0, 0])
normalisation = np.sum(site_init_state)

for i in range(num_realisations):
    max = np.amax(population_dynamics_realisations[i])
    if max > 2.5 or np.isnan(max):
        print i
        print max
        print liouvillians[i]
        
for col in liouvillians[1935].T:
    print np.sum(col)
        

population_dynamics_realisations = np.delete(population_dynamics_realisations, 1935, 0)
num_realisations -= 1

# population_dynamics_realisations = np.delete(population_dynamics_realisations, 415, 0)
# population_dynamics_realisations = np.delete(population_dynamics_realisations, 888, 0)
# population_dynamics_realisations = np.delete(population_dynamics_realisations, 4615, 0)
# num_realisations -= 3

# average realisations
population_dynamics_averaged = np.sum(population_dynamics_realisations/normalisation, axis=0) / (num_realisations)
  
labels = ["P_D1", "P_D2", "Chl_D1", "Chl_D2", "Phe_D1", "Phe_D2", "Chl_D1+Phe_D1-", "P_D1+Phe_D1-"]
  
for i in range(population_dynamics_averaged.shape[0]):
    plt.plot(dynamics_time, population_dynamics_averaged[i], label=labels[i])
#plt.ylim(0, 3)
plt.legend().draggable()
plt.show()

# dynamics_index = 2617
#   
# for i in range(population_dynamics_realisations[dynamics_index].shape[0]):
#     plt.plot(dynamics_time, population_dynamics_realisations[dynamics_index][i], label=labels[i])
# #plt.ylim(0, 3)
# plt.legend().draggable()
# plt.show()


