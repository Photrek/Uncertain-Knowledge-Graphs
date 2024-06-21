import matplotlib.pyplot as plt
import numpy as np

# Probability
probabilities = [0.75, 0.42, 0.83, 0.57, 0.69, 0.88, 0.75, 0.66, 0.45]

# Dof=1000
lower_bounds_1 = [0.7504, 0.4198, 0.8304, 0.5701, 0.6903, 0.8804, 0.7504, 0.6603, 0.4499]
upper_bounds_1 = [0.7496, 0.4202, 0.8296, 0.5699, 0.6897, 0.8796, 0.7496, 0.6597, 0.4501]

# Dof=100
lower_bounds_2 = [0.7504, 0.4198, 0.8304, 0.5701, 0.6903, 0.8804, 0.7504, 0.6603, 0.4499]
upper_bounds_2 = [0.7496, 0.4202, 0.8296, 0.5699, 0.6897, 0.8796, 0.7496, 0.6597, 0.4501]

# Dof=10
lower_bounds_3 = [0.7889, 0.4044, 0.8702, 0.5838, 0.7232, 0.9161, 0.7889, 0.6891, 0.4401]
upper_bounds_3 = [0.7015, 0.4376, 0.7744, 0.5546, 0.6507, 0.8249, 0.7015, 0.6262, 0.4611]

# dof=1
lower_bounds_4 = [0.9000, 0.3440, 0.9597, 0.6373, 0.8321, 0.9817, 0.9000, 0.7903, 0.4010]
upper_bounds_4 = [0.2500, 0.5800, 0.1700, 0.4300, 0.3100, 0.1200, 0.2500, 0.3400, 0.5500]

# array 
probabilities = np.array(probabilities)
lower_bounds_1 = np.array(lower_bounds_1)
upper_bounds_1 = np.array(upper_bounds_1)
lower_bounds_2 = np.array(lower_bounds_2)
upper_bounds_2 = np.array(upper_bounds_2)
lower_bounds_3 = np.array(lower_bounds_3)
upper_bounds_3 = np.array(upper_bounds_3)
lower_bounds_4 = np.array(lower_bounds_4)
upper_bounds_4 = np.array(upper_bounds_4)

# Upper and lower bounds
yerr_1 = np.abs([upper_bounds_4 - probabilities, probabilities - lower_bounds_4])
yerr_2 = np.abs([upper_bounds_3 - probabilities, probabilities - lower_bounds_3])
yerr_3 = np.abs([upper_bounds_2 - probabilities, probabilities - lower_bounds_2])
yerr_4 = np.abs([upper_bounds_1 - probabilities, probabilities - lower_bounds_1])

# plot
plt.errorbar(probabilities, probabilities, yerr=yerr_1, fmt='o', label='DoF=1', color='blue', linewidth=3)
plt.errorbar(probabilities, probabilities, yerr=yerr_2, fmt='o', label='DoF=10', color='red', linewidth=3)
plt.errorbar(probabilities, probabilities, yerr=yerr_3, fmt='o', label='DoF=100', color='green', markersize=8)
plt.errorbar(probabilities, probabilities, yerr=yerr_4, fmt='o', label='DoF=1000', color='purple', markersize=5)

plt.xlabel('Probability', fontsize=16)
plt.ylabel('The upper and lower bounds of risks', fontsize=14)

plt.legend(loc='upper left', fontsize=14)

plt.tick_params(axis='both', which='major', labelsize=14)

plt.show()
