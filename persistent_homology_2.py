import numpy as np
import matplotlib.pyplot as plt
import ripser
from persim import plot_diagrams

# Generate the dataset (e.g., a set of random points)
data = np.random.random((100, 2))

# Compute persistent homology
result = ripser.ripser(data)

# Plot the persistence diagram
plot_diagrams(result['dgms'])
plt.show()
