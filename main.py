from scripts.visualize import load_calculate_and_visualize_neuron_and_vesicles


def vizualization_and_calculations_tensors():
    load_calculate_and_visualize_neuron_and_vesicles('data/vol0_mask.h5', 'data/vol0_vesicle_ins.h5')


if __name__ == "__main__":
    vizualization_and_calculations_tensors()
