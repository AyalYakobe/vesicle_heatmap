from scripts.vesicle_load_tensor_files import load_data
from scripts.visualize import display_neuron


def vizualization_and_calculations_tensors():
    neuron_data, vesicle_data = load_data('data/vol0_mask.h5', 'data/vol0_vesicle_ins.h5')
    display_neuron(neuron_data, vesicle_data)

if __name__ == "__main__":
    vizualization_and_calculations_tensors()

