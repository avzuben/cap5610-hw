import numpy as np
from scipy.spatial.distance import cdist

a = np.array([(4.7, 3.2), (4.9, 3.1), (5.0, 3.0), (4.6, 2.9)])
b = np.array([(5.9, 3.2), (6.7, 3.1), (6.0, 3.0), (6.2, 2.8)])

distances = cdist(a, b, 'euclidean')

print(distances)
print(np.mean(distances))
