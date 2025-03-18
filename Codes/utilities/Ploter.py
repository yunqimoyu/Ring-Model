import sys
import os

current_script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(current_script_path, '..')
sys.path.append(os.path.abspath(module_path))


from utilities.VecOps import NumTrans

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import numpy as np


# Get the default color cycle
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# The second color in the default cycle
orange_color = color_cycle[1]  # Indexing from 0, so the second color is at index 1

def plot_perturb_result_from_img_space(imgs, inputs, outputs, save_path):

    # plot images, including original image; perturbed image; perturbation
    # plot input signals
    # plot outputs
    # save figure
        # Ensure the input lists have appropriate lengths
    # assert len(imgs) == 3, "imgs should contain 3 images."
    # assert len(inputs) == 2, "inputs should contain 2 series."
    # assert len(outputs) == 2, "outputs should contain 2 series."

    # Create a figure with a grid of subplots: 3 rows and 3 columns
    
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(3, 3, figure=fig)

    # Plot images in the first column with colorbars
    for i in range(3):
        ax_img = fig.add_subplot(gs[i, 0])
        im = ax_img.imshow(imgs[i], cmap='viridis', origin='lower')
        fig.colorbar(im, ax=ax_img)
        # ax_img.axis('off')
        # ax_img.set_xticks(range(0, imgs[i].shape[1], 2))
        # ax_img.set_yticks(range(0, imgs[i].shape[0], 2))

    # Plot a single subplot with curves in the second column
    ax_curves_input = fig.add_subplot(gs[:, 1])  # Span all rows in second column
    ax_curves_input.plot(inputs[0], label='Input Curve 1')
    ax_curves_input.plot(inputs[1], label='Input Curve 2')
    ax_curves_input.set_title('Input Curves')
    ax_curves_input.legend()

    # Plot a single subplot with curves in the third column
    ax_curves_output = fig.add_subplot(gs[:, 2])  # Span all rows in third column
    ax_curves_output.plot(outputs[0], label='Output Curve 1')
    ax_curves_output.plot(outputs[1], label='Output Curve 2')
    ax_curves_output.set_title('Output Curves')
    ax_curves_output.legend()

    plt.tight_layout()
    os.makedirs(f'{save_path}', exist_ok=True)
    plt.savefig(f'{save_path}/result.png')
    return

def draw_solutions_comparison(s1, s2, l1, l2, figsize=(12, 4)):
    fig, axs = plt.subplots(ncols=3, figsize = figsize)
    for (i, curve) in enumerate(s1):
        axs[0].plot(curve, label=l1[i])
    for (i, curve) in enumerate(s2):
        axs[1].plot(curve, label=l2[i])
    diff = s2[0]-s1[0]
    axs[2].plot(diff)
    max_value = np.max([np.max(s1), np.max(s2)])
    min_value = np.min([np.min(s1), np.min(s2)])
    maximum = np.ceil(max_value)
    minimum = np.ceil(min_value) - 1
    maximum = NumTrans.max_rise(maximum)
    minimum = NumTrans.min_decrease(minimum)
    axs[0].set_ylim(bottom=minimum, top = maximum)
    axs[1].set_ylim(bottom=minimum, top = maximum)
    formatter = ScalarFormatter(useOffset=False)
    axs[2].yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    return fig

def perturb_influence_visualization(original_image, perturb_image, perturbed_image, input_signal, input_signal_p, RM_output, RM_output_p, figsize=(12, 6)):
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=figsize)

    im0 = axes[0][0].imshow(original_image, cmap='gray', origin='lower')
    colorbar = fig.colorbar(im0)
    im1 = axes[0][1].imshow(perturb_image, cmap='gray', origin='lower')
    colorbar = fig.colorbar(im1)
    im2 = axes[0][2].imshow(perturbed_image, cmap='gray', origin='lower')
    colorbar = fig.colorbar(im2)

    axes[1][0].plot(input_signal, label=f'original input')
    axes[1][0].plot(input_signal_p, label=f'perturbed input')
    axes[1][0].legend()

    axes[1][1].plot(RM_output[0], label=f'original rE')
    axes[1][1].plot(RM_output_p[0], label=f'perturbed rE')
    axes[1][1].plot(RM_output[0] - RM_output_p[0], label=f'rE difference')
    axes[1][1].legend()

    axes[1][2].plot(RM_output[1], label=f'original rI')
    axes[1][2].plot(RM_output_p[1], label=f'perturbed rI')
    axes[1][2].plot(RM_output[1] - RM_output_p[1], label=f'rI difference')
    axes[1][2].legend()
    return fig

def draw_sets_of_curves(x, curves_data, titles, line_labels, figsize=(12, 4), xlabel='ξ', ylabel='value'):
    ncols=len(curves_data)
    fig, axs = plt.subplots(ncols=ncols, figsize=figsize)
    
    # Iterate over each subplot's data, title, and axes
    for i, (data, title, ax) in enumerate(zip(curves_data, titles, axs)):
        # Iterate over each line's data and color for the current subplot
        for (y, linestyle), label in zip(data, line_labels[i]):
            ax.plot(x, y, label=label, linestyle=linestyle)
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel(xlabel)
        if i == 0:  # Only add y label to the first subplot to avoid repetition
            ax.set_ylabel(ylabel)
            
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    return fig

def draw_singular_values(singularvalues, freqs, labels, title):
    fig, ax = plt.subplots()
    for singular_value, freq, label in zip(singularvalues, freqs, labels):
        ax.plot(freq, singular_value, marker='o', label = label)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(f'$\\xi$')
    return fig

def draw_varing_hat_h_inv(hat_h_inv_list):
    fig, ax = plt.subplots(ncols=3, figsize=(12, 3))
    for i, hat_h_inv in enumerate(hat_h_inv_list):
        ax[i].plot(hat_h_inv, marker='o', markersize=4)
        ax[i].set_xlabel(f'$\\xi$')

    plt.tight_layout()
    return fig

def draw_kernel_diagram(kEE, kEIIE, neuron_pref_orients,
                        hat_kEE, hat_kEIIE, freqs,
                        hat_h_inv_with_EE_only, hat_h_inv_with_EIIE_only):
    k = "k"
    k_k = "k*k"
    h = "h"
    subscript='-1'
    fig, axes = plt.subplots(ncols=3, figsize=(14, 3.2))
    axes0_twin = axes[0].twinx()
    axes[0].plot(neuron_pref_orients, kEE, label=f'$k$')
    axes0_twin.plot(neuron_pref_orients, kEIIE, label = f'$k*k$', color=orange_color)
    axes0_twin.tick_params(axis='y', labelcolor=orange_color)
    axes[0].set_title(f'Lateral connection kernels')
    # Create a shared legend
    lines0 = axes[0].get_lines() + axes0_twin.get_lines()  # Combine the lines from both axes
    labels = [line.get_label() for line in lines0]  # Get their labels
    plt.legend(lines0, labels)

    axes[1].plot(freqs, hat_kEE, label=f'$\\widehat{{{k}}}$')
    axes1_twin = axes[1].twinx()
    axes1_twin.plot(freqs, hat_kEIIE, label=f'$\\widehat{{{k_k}}}$', color=orange_color)
    axes1_twin.tick_params(axis='y', labelcolor=orange_color)
    axes[1].set_title(f'Kernels in frequency space')
    # Create a shared legend
    lines1 = axes[1].get_lines() + axes1_twin.get_lines()  # Combine the lines from both axes
    labels = [line.get_label() for line in lines1]  # Get their labels
    plt.legend(lines1, labels)

    axes[2].plot(hat_h_inv_with_EE_only, marker='o', label=f'excitatory')
    axes[2].plot(hat_h_inv_with_EIIE_only, marker='o', label=f'inhibitory')
    axes[2].set_title(f'$\\widehat{{{h}_{{{subscript}}}}}$ when only excitatoy/inhibitory part works')
    axes[2].legend()
    plt.tight_layout()
    axes[0].set_xlabel(f'$\\theta$')
    axes[1].set_xlabel(f'$\\xi$')
    axes[2].set_xlabel(f'$\\xi$')
    return fig

def draw_natural_imgs_pass_gabor(resized_images, signals, module_path):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2, nrows * 3))

    axes = axes.flatten()

    for i in range(5):
        axes[2 * i].imshow(resized_images[i], cmap='gray')
        axes[2 * i].axis('off')

        axes[2 * i + 1].plot(signals[i])
        axes[2 * i + 1].set_title(f'Signal {i+1}')
        i += 1

    plt.tight_layout()
    plt.savefig(f'{module_path}/../Data/Interim/natural_images/natural_images_pass_gabor')
    return


def draw_sinusoids_singularvectors(df, x, f, image_title='singularvectors', num_plots = 8, width_ratios=[1.4, 1, 1], num_tick=7, save_path='./'):
    """
    df: a dataframe contains columns: ut, s, vt, dom_freq, fft_result
    x:  x labels
    f:  x labels for the frequencies
    """
    # Set up the figure layout with constrained_layout
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)

    # Add the 2D plot with a separate colorbar
    ax1 = fig.add_subplot(131)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.2)

    image = np.array(df['vt'].to_list()).T
    im = ax1.imshow(image)  # assuming 'image' is your data
    colorbar = fig.colorbar(im, cax=cax, orientation='vertical')
    
    # Set y-labels
    num_y_ticks = 8
    y_tick_positions = np.linspace(0, image.shape[0]-1, num_y_ticks, endpoint=False)
    y_tick_labels = np.linspace(0, 180, num_y_ticks, endpoint=False)
    ax1.set_ylabel(f'$\\theta$')
    ax1.set_yticks(ticks=y_tick_positions)
    ax1.set_yticklabels(labels=[f'{y:.1f}' for y in y_tick_labels])

    # Increase font size for color bar
    colorbar.ax.tick_params(labelsize=12)  # Adjust font size for ticks

    ax1.set_title(image_title, fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)  # larger tick size

    # Create a grid for the 1D line plots
    gs = gridspec.GridSpec(num_plots, 3, figure=fig, width_ratios=width_ratios, hspace=0.06)
    sorted_df = df.sort_values(by='dom_freq')

    for i in range(num_plots):
        row = sorted_df.iloc[i]
        # Add the line plot for each row
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(x, row['vt'])  # assuming 'x1', 'col1' are defined
        ax2.set_title(f'ξ = {row['dom_freq']}, s = {row['s']:.2e}', loc='left', fontsize=14)  # larger font size
        ax2.set_ylim(sorted_df['vt'].head(8).apply(np.min).min()*1.1, sorted_df['vt'].head(8).apply(np.max).max()*1.1)
        ax2.tick_params(axis='both', labelsize=12)  # larger tick size

        ax3 = fig.add_subplot(gs[i, 2])
        ax3.plot(f, row['fft_result'][:10], marker='o')  # assuming 'x2', 'col2' are defined
        ax3.tick_params(axis='both', labelsize=12)  # larger tick size

        # Remove x-ticks except for the last plot
        if i != num_plots - 1:
            ax2.set_xticks([])
            ax3.set_xticks([])
        else:
            ax2.set_xlabel(f'$\\theta$')
            ax3.set_xlabel(f'$\\xi$')
    fig.savefig(f'{save_path}/singularvectors.pdf')
    return fig

def draw_signals_and_images(U, VT, x_s, nrows= 2, ncols = 3, 
                           y_lim = 0.15, image_size = (64, 64)):
    '''
    U: the matrix of signals. Each column is a signal.
    VT: the matrix of images. Each row is an image, which need to be reshaped.
    '''
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols*2, figsize=(ncols*8, nrows*3), dpi=300)
    exit_flag = False
    for i in range(nrows):
        for j in range(ncols):
            order = i*ncols+j
            if order >= U.shape[1]:
                exit_flag = True
                break
            signal = U[:, order]
            image = VT[order].reshape(image_size)
            ax[i][j*2+1].plot(x_s, signal)
            ax[i][j*2+1].set_xlabel(f'$\\theta$', fontsize=14)
            ax[i][j*2+1].set_ylim([-y_lim, y_lim])
            ax[i][j*2+1].tick_params(axis='x', labelsize=14)
            ax[i][j*2+1].tick_params(axis='y', labelsize=14)
            im = ax[i][j*2].imshow(image, cmap='gray')
            ax[i][j*2].tick_params(axis='x', labelsize=14)
            ax[i][j*2].tick_params(axis='y', labelsize=14)
            cbar = fig.colorbar(im)
            cbar.ax.tick_params(labelsize=14)
        if exit_flag:
            break
    plt.tight_layout()
    return fig

def circulant_matrix_visualization(H, H_inv, save_path):
    fig, ax = plt.subplots(ncols=2)
    im0 = ax[0].imshow(H, cmap='gray')
    im1 = ax[1].imshow(H_inv, cmap='gray')
    colorbar = fig.colorbar(im0, orientation='vertical')
    colorbar = fig.colorbar(im1, orientation='vertical')
    plt.savefig(f'{save_path}/H_and_H_inv')

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0][0].plot(H[0])
    ax[1][0].plot(H_inv[0])
    n = H.shape[1]//2
    ax[0][1].plot(H[n])
    ax[1][1].plot(H_inv[n])
    fig.suptitle(f'left: 0th row, right: {n}th row')
    plt.savefig(f'{save_path}/H_and_H_inv_rows')
    return 

def draw_singularvectors(x, df, save_path, freq_sort=False):
    fig, ax = plt.subplots(nrows=2)
    if freq_sort:
        df = df.sort_values(by='dom_freq')
    def animate(i):
        ax[0].clear()
        ax[1].clear()

        row = df.iloc[i]
        ax[0].plot(x, row['vt'])
        ax[1].plot(row['fft_result'])

        fig.suptitle(f'i={i}  s={row['s']:.3e}  dom_freq={row['dom_freq']}')
        plt.savefig(f'{save_path}/singularvectors_seperate_i={i}')

    ani=animation.FuncAnimation(fig, animate, frames = df.shape[0])
    ani.save(f'{save_path}/singularvector.mp4')
    return fig

def draw_singularvalues(df, plot_freq, save_path):
    filtered_df = df[df['dom_freq'] < plot_freq]
    sorted_df = filtered_df.sort_values(by='dom_freq')
    fig, ax = plt.subplots()
    ax.plot(sorted_df['dom_freq'], sorted_df['s'], marker='o')
    plt.savefig(f'{save_path}/singularvalues')
    print('img is saved!')
    return fig

def draw_sinusoids_amp(sinusoids, plot_freq, save_path):
    df = sinusoids
    filtered_df = df[df['freq'] < plot_freq]
    sorted_df = filtered_df.sort_values(by='freq')
    fig, ax = plt.subplots()
    ax.plot(sorted_df['freq'], sorted_df['amp'], marker='o')
    plt.savefig(f'{save_path}/sinusoids_amp')
    print('img is saved!')
    return fig

def draw_sinusoids(x, sinusoids, save_path):
    fig, ax = plt.subplots()
    def animate(i):
        ax.clear()
        row = sinusoids.iloc[i]
        ax.plot(x, row['sin'], label = 'input')
        ax.plot(x, row['output'], label = 'output')
        ax.legend()
        ax.set_title(f'freq={row['freq']} amp={row['amp']}')
        plt.savefig(f'{save_path}/opt_sinusoid_freq={row['freq']}')
        return
    ani = animation.FuncAnimation(fig, animate, frames = sinusoids.shape[0])
    print(save_path)
    ani.save(f'{save_path}/opt_sinusoids.mp4')
    return fig