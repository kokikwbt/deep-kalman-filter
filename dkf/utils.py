import matplotlib.pyplot as plt
import numpy as np


def generate_synthetic(seqlen=500):
    """ Synthetic 4-dimensional data """
    wave1 = 2*np.sin(np.linspace(0, 20*np.pi, seqlen))
    wave2 = 2*np.sin(np.linspace(0, 2*np.pi, seqlen))
    data = np.vstack([wave1, wave1*1.2, wave2, wave2*0.85]).T
    return data + np.random.randn(*data.shape)


def plot_result():
    return