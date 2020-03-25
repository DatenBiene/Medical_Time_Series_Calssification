import keract
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def display_conv_activations(classifier,sig):
    #get activations
    sig = sig.reshape(1,sig.shape[0],sig.shape[1])
    activations = keract.get_activations(classifier.model, sig)
    #prepare for plotting
    t = np.linspace(0, len(sig[0]),len(sig[0]))
    signal = sig[0].reshape(-1,)
    points = np.array([t, signal]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #create fig and axs
    fig = plt.figure(figsize=(18,5))

    for i in range(1,11):
        ax = fig.add_subplot(2,5,i)
        key = 'conv1d_'+str(i)
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
