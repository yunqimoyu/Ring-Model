import sys
import os

current_script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(current_script_path, '..')
sys.path.append(os.path.abspath(module_path))

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

sigma_EE =  10
sigma_EI =  10
sigma_IE =  10

N = 128
T = 180

# Parameter settings
alpha_EE = np.linspace(1e-5, 0.2, 200)  # Horizontal coordinate range
alpha_EI_IE = np.linspace(1e-5, 0.0102, 200)  # Vertical coordinate range
# Parameters for the two linear boundaries
c1 = np.sqrt(2*np.pi) * sigma_EI * sigma_IE / sigma_EE
c2 = np.sqrt(2*np.pi) * sigma_EI * sigma_IE * (sigma_EI**2 + sigma_IE**2) / sigma_EE ** 3

# Grid the horizontal and vertical coordinates to generate the lattice
X, Y = np.meshgrid(alpha_EE, alpha_EI_IE)

# Define several boundary lines
boundary1 = (T/N) * alpha_EE / c1  # Boundary 1: Linear
boundary2 = (T/N) * alpha_EE / c2  # Boundary 2: Linear

# Define function f, depending on the region of alpha_EE and alpha_EI_IE
def f(alpha_EE, alpha_EI_IE):
    alpha_EE = (N/T) * alpha_EE
    alpha_EI_IE = (N/T)**2 * alpha_EI_IE

    c = np.sqrt(2*np.pi) * alpha_EI_IE * sigma_EI*sigma_IE*(sigma_EI ** 2 + sigma_IE ** 2) / (alpha_EE * sigma_EE ** 3)
    # c = c2 * alpha_EI_IE / alpha_EE
    # c = 10 * alpha_EI_IE * sigma_EI*sigma_IE*(sigma_EI ** 2 + sigma_IE ** 2)/alpha_EE * sigma_EE ** 3
    print('c=', c)
    a = alpha_EE * np.sqrt(2 * np.pi) * sigma_EE * c ** (sigma_EE/(sigma_EE**2 - (sigma_EI**2 + sigma_IE**2)))
    b = alpha_EI_IE * 2 * np.pi * sigma_EI * sigma_IE * c ** (sigma_EE/(sigma_EE**2 - (sigma_EI**2 + sigma_IE**2)))
    return   1 - a + b  # Custom nonlinear function

a_EE = 0.1
a_EI_IE = (T/N) * a_EE / c2
# a_EI_IE = a_EE / c2
f_value = f(a_EE, a_EI_IE)
print(f_value)

# Compute the values of function f
f_values = f(X, Y)

# Define function f2, depending on the region of alpha_EE and alpha_EI_IE
def f2(alpha_EE, alpha_EI_IE):
    alpha_EE = (N/T) * alpha_EE
    alpha_EI_IE = (N/T)**2 * alpha_EI_IE

    a = alpha_EE * np.sqrt(2 * np.pi) * sigma_EE 
    b = alpha_EI_IE * 2 * np.pi * sigma_EI * sigma_IE

    return   1 - a + b  # Custom nonlinear function

# Compute the values of function f2
f2_values = f2(X, Y)

# Plot the phase diagram
plt.figure(figsize=(8, 6))

# Plot each boundary
contour_orange = plt.contour(X, Y, f2_values, levels=[0], colors='orange', linewidths=2, linestyles='dashed')  # f2=0 curve
contour_orange.collections[0].set_visible(False)  # Hide the original orange line
contour = plt.contour(X, Y, f_values, levels=[0], colors='green', linewidths=2, linestyles='dashed')  # f=0 curve
contour.collections[0].set_visible(False)  # Hide the original green line

# Get the orange line path
orange_line_path = contour_orange.collections[0].get_paths()[0]  # Get the orange line path
orange_line_vertices = orange_line_path.vertices
orange_x, orange_y = orange_line_vertices[:, 0], orange_line_vertices[:, 1]

# Get the green line path
green_line_path = contour.collections[0].get_paths()[0]  # Assuming there is only one green line
green_line_vertices = green_line_path.vertices  # Extract green line vertices (x, y) array
green_x, green_y = green_line_vertices[:, 0], green_line_vertices[:, 1]  # Split into x and y

# Use interpolation to compare green and orange lines
interp_green_y = np.interp(orange_x, green_x, green_y)  # Interpolate green line y values
mask_orange = orange_y <= interp_green_y  # Only keep the portion of the orange line below the green line

# Crop the range of the green line, restricting it to the range covered by the orange line
min_orange_x, max_orange_x = np.min(orange_x), np.max(orange_x)
interp_orange_y = np.interp(green_x, orange_x, orange_y, left=np.inf, right=-np.inf)  # Assign invalid values for out-of-range
mask_green = (green_x >= min_orange_x) & (green_x <= max_orange_x) & (green_y <= interp_orange_y)

# Fill each region
plt.fill_between(alpha_EE, 0, boundary1, color='lightblue', alpha=0.5, label='Region 1')  # Bottom-right region
plt.fill_between(alpha_EE, boundary2, 5, color='lightyellow', alpha=0.5, label='Region 4')  # Top-left region

# Fill the unstable region to the right of the green line
# Use fill_betweenx, combining cropped orange and green boundaries
x_max = X.max()  # Maximum y-coordinate on the right of the graph
# plt.fill_betweenx(orange_y[mask_orange], orange_x[mask_orange], x_max, color='white', alpha=1, edgecolor='white', linewidth=2)  # Fill white background
plt.fill_betweenx(orange_y[mask_orange], orange_x[mask_orange], x_max, color="#fcf1f0", alpha=1, edgecolor='#fcf1f0', linewidth=2, label='Invalid region')
# plt.fill_betweenx(green_y[mask_green], green_x[mask_green], x_max, color='white', alpha=1, edgecolor='white', linewidth=2)  # Fill white background
plt.fill_betweenx(green_y[mask_green], green_x[mask_green], x_max, color="#fcf1f0", alpha=1, edgecolor='#fcf1f0', linewidth=2, label='Invalid region')

# # Draw the borders of the filled regions
# plt.plot([green_x[mask_green][0], x_max], [green_y[mask_green][0], green_y[mask_green][0]], 
#          color='#fcf1f0', linewidth=3)  # Bottom border

# Plot the cropped orange boundary
plt.plot(orange_x[mask_orange], orange_y[mask_orange], color='green', linestyle='dashed', linewidth=2, label='Boundary Orange (clipped)')

# Plot the cropped green line
plt.plot(green_x[mask_green], green_y[mask_green], color = 'green', linestyle = 'dashed', linewidth=2, label='Boundary Green (clipped)')

# Crop the red line data
red_x = alpha_EE
red_y = boundary2

# Only keep the portion of the red line to the left of the green line
mask = red_y >= np.interp(red_x, green_x, green_y)  # Interpolate to calculate the y values of the green line
plt.plot(red_x[mask], red_y[mask], 'r-', label='Boundary Red (clipped)')

# Crop the blue line data
blue_x = alpha_EE
blue_y = boundary1

# Only keep the portion of the blue line to the left of the green line
mask = blue_y >= np.interp(blue_x, green_x, green_y)  # Interpolate to calculate the y values of the green line
plt.plot(blue_x[mask], blue_y[mask], 'b-', label='Boundary blue (clipped)')

# Add region labels
plt.text(0.05, 0.0006, 'I', fontsize=12, color='black')  # Place at the center of Region 1
plt.text(0.125, 0.0054, 'II', fontsize=12, color='black')  # Place at the center of Region 2
plt.text(0.025, 0.008, 'III', fontsize=12, color='black')  # Place at the center of Region 3
plt.text(0.135,  0.002, 'Unstable', fontsize=14, color='black')  

# Graph formatting
plt.xlim(0, 0.2)
plt.ylim(0, 0.0102)
plt.xlabel(r'$\alpha_{EE}$', fontsize=12)
plt.ylabel(r'$\alpha_{EI}\alpha_{IE}$', fontsize=12)
# plt.title('Phase Diagram with Additional Non-linear Boundary')
# plt.legend()

# Label the corresponding points
# Coordinates of the points to label
points_x = [0.025, 0.08, 0.16]  # x values of S_EE
points_y = [0.0001, 0.0025, 0.01]  # y values of S_EI*S_IE

# Plot the points
plt.scatter(points_x, points_y, edgecolors='black', facecolors='none', s=24, zorder=5)  # Draw black circles
# for i, (x, y) in enumerate(zip(points_x, points_y)):
#     plt.text(x + 0.1, y, f'P{i+1}', fontsize=12)  # Label the text next to each point, can adjust as needed

# Save the image
save_path = f'{current_script_path}/../../Data/eigenvalue_shape_phase_diagram'
os.makedirs(f'{save_path}', exist_ok=True)
plt.savefig(f'{save_path}/phase_diagram.pdf', dpi=300, bbox_inches='tight')  # File name is 'phase_diagram.png'