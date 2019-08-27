import tensorflow as tf
import numpy as np
import tfplot
import matplotlib.pyplot as plt

# adapted from https://github.com/grishasergei/conviz/
def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(np.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)

def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]

def pca(X, num_observations=64, n_dimensions = 50):
    singular_values, u, _ = tf.svd(X)
    sigma = tf.diag(singular_values)
    print(sigma)
    
    sigma = tf.slice(sigma, [0, 0], [num_observations, n_dimensions])
    
    pca = tf.matmul(u, sigma)
    pca = tf.transpose(pca)
    return pca

@tfplot.wrap
def plot_conv_weights(weights, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param channels_all: boolean, optional

    """
    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    # switched dimensions FIX THIS LATER
    grid_c, grid_r = get_grid_dim(num_filters)
    
    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]), max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.close(fig)
    
    return fig

@tfplot.wrap
def plot_conv_output(conv_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
#     w_min = np.min(conv_img)
#     w_max = np.max(conv_img)

    w_min = 0.0
    w_max = 1.0
    
    print(w_min)
    print(w_max)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns

    grid_r, grid_c = get_grid_dim(num_filters)
    

    if (min([grid_r, grid_c]) == 1 and max([grid_r, grid_c]) == 1):
        
        fig, ax = plt.subplots(figsize=(24,24))   
        fig.suptitle(name, fontsize=25, color='k')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        img = conv_img[0, :, :,  0]
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys_r')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.subplots_adjust(top=0.85)
        plt.close(fig)
        return fig
        
        
    # create figure and axes
    
# VERTICAL MODE UNCOMMENT HERE
#     fig, axes = plt.subplots(min([grid_r, grid_c]),
#                              max([grid_r, grid_c]),
#                             figsize=(24,15))
    
    
    
    fig, axes = plt.subplots(max([grid_r, grid_c]),
                             min([grid_r, grid_c]),
                            figsize=(15,24))
    fig.suptitle(name, fontsize=25, color='k')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='jet')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
       
    plt.close(fig)
    return fig
    
    