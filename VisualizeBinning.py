import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
import seaborn as sns
import sys

# The goal of this .py file is to visualize the effect of different values of 'density_strength' on theta binning

xpos_bins = np.linspace(-2.4, 2.4, 25)
theta_raw_bins = np.linspace(-1, 1, 25)
density_values = [0, 0.2, 0.4, 0.6, 0.8, 1]  # Plotting varying density_strength values

# Style plot
plt.style.use("cyberpunk")
plt.figure(figsize=(18, 10), facecolor="#282c44")

# Define a color palette
palette = sns.color_palette("flare", len(density_values))

for i, density_strength in enumerate(density_values):
    if density_strength > 1 or density_strength < 0:
        print(f"ERROR: Density strength ({density_strength}) is out of bounds. Change value to be between zero and 1.")
        sys.exit()

    # Calculating theta bins based on density_strength
    theta_bins = 12 * (density_strength) * (theta_raw_bins)**5 + 12 * (1 - density_strength) * theta_raw_bins

    # Create subplots: 3 rows and 2 columns for 6 plots
    ax = plt.subplot(3, 2, i + 1)
    ax.set_facecolor("#282c44")

    # Scatter plot with specific color from the palette
    scatter = ax.scatter(theta_bins, np.zeros_like(theta_bins), color=palette[i], label=f'Density Strength = {density_strength}')

    # Set titles and labels with light colors for contrast
    ax.set_title(f'Theta Binning with Density Strength = {density_strength}', color='white')
    ax.set_xlabel('Value (degrees)', color='white')
    ax.set_ylabel('Bins', color='white')

    # Customize grid to be visible on dark background
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # Customize tick parameters for better visibility
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Optional: Adding a legend with light text
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color("white")

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the final plot
plt.show()
