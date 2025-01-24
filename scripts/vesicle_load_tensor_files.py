import edt
import h5py
import numpy as np


def load_data(neuron_file_path, vesicle_file_path):
    with h5py.File(neuron_file_path, 'r') as f:
        neuron_data = f['main'][:]
    with h5py.File(vesicle_file_path, 'r') as f:
        vesicle_data = f['main'][:]
    return neuron_data, vesicle_data

def calculate_distance_transform(neuron_data):
    print(np.unique(neuron_data))
    return_edt = edt.edt(1 - neuron_data.astype(np.uint32), anisotropy=(8, 8, 30), black_border=True, order='F')
    return_edt = edt.edt(1 - neuron_data, anisotropy=(8, 8, 30))
    return return_edt

