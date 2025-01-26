import edt
import h5py
import numpy as np


def load_data(neuron_file_path, vesicle_file_path):
    with h5py.File(neuron_file_path, 'r') as f:
        neuron_data = f['main'][:]
    with h5py.File(vesicle_file_path, 'r') as f:
        vesicle_data = f['main'][:]
    return neuron_data, vesicle_data


