import keract
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def display_conv_activations(model,sig):
    if len(sig.shape)==2:
        sig = sig.reshape(1,sig.shape[0],sig.shape[1])
    #get activations
    activations = keract.get_activations(model, sig)
    #get convolutional layers keys
    conv_keys = [key for key in activations.keys() if 'conv' in key]
    #prepare for plotting
    t = np.linspace(0, len(sig[0]),len(sig[0]))
    signal = sig[0].reshape(-1,)
    points = np.array([t, signal]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #create fig and axs
    fig = plt.figure(figsize=(18,5))

    if len(conv_keys)>10:
        print('Warning: only the 10 last layers will be displayed')

    for i in range(1,min([11,len(conv_keys)])):
        ax = fig.add_subplot(2,5,i)
        key = conv_keys[-i]
        act = np.mean(activations[key][0],axis=1)
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(act.min(), act.max())
        lc = LineCollection(segments, cmap='bwr', norm=norm)
        # Set the values used for colormapping
        lc.set_array(act)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)
        ax.set_xlim(t.min(), t.max())
        ax.set_title(key)
    plt.tight_layout()


def display_conv_activations_transplant(model,sig,cols):
    """
    for transplant with 10 channels
    """

    #get activations
    sig = sig.reshape(1,sig.shape[0],sig.shape[1])
    activations = keract.get_activations(model, sig)
    conv_keys = [key for key in activations.keys() if 'conv' in key]
    #prepare for plotting
    t = np.linspace(0, len(sig[0]),len(sig[0]))
    #signal = sig[0].reshape(-1,)
    #create fig and axs
    fig = plt.figure(figsize=(14,5))
    key = conv_keys[-1] #last convolutional layer
    act = np.mean(activations[key][0],axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(act.min(), act.max())
    for i in range(1,11):

        signal = sig[0][:,i-1]
        points = np.array([t, signal]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        ax = fig.add_subplot(2,5,i)
        lc = LineCollection(segments, cmap='bwr', norm=norm)
        # Set the values used for colormapping
        lc.set_array(act)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)
        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(signal.min(),signal.max())
        ax.set_title(cols[i],fontsize=13)

    plt.tight_layout()
