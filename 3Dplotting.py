# Plug-in your dicovered waveform frequencies to plot a 3D visualization

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Short Time Window for High Frequency Visualizations
duration = 0.01  # 10 milliseconds window
sample_rate = 100000

# Create 2D grid with proper shapes
x = np.linspace(-5, 0.5, 50)  
y = np.linspace(-5, 0.5, 50)
X, Y = np.meshgrid(x, y)

# Define frequencies for the waves
f_sine = 1600
f_square = 1
f_triangle = 1550
f_sawtooth = 15900

# Calculate wave components - make sure they are all 2D with same shape as X and Y
sine_component = np.sin(2 * np.pi * f_sine * X)
square_component = np.sign(np.sin(2 * np.pi * f_square * Y))
triangle_component = 2/np.pi * np.arcsin(np.sin(2 * np.pi * f_triangle * X))
sawtooth_component = 2 * (f_sawtooth * Y - np.floor(0.5 + f_sawtooth * Y))

# Combine all components - this should be a 2D array with the same shape as X and Y
Z = sine_component * square_component * triangle_component * sawtooth_component

# Print shapes to debug
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Z shape: {Z.shape}")

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='none', alpha=0.9)

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Set labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Surface with Multiple Waveforms')

plt.tight_layout()
plt.show()
