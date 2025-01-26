from skimage.filters.thresholding import threshold_otsu
from skimage.measure import find_contours
from scipy.ndimage import label
import matplotlib.pyplot as plt
import numpy as np

def compute_max_outline(neuron_data):
    neuron_projection = neuron_data.max(axis=0)
    thresh = threshold_otsu(neuron_projection)
    binary_neuron = neuron_projection > thresh
    return binary_neuron

def project_vesicles_to_2d(vesicle_data):
    projected_vesicle_data = vesicle_data.max(axis=0)
    labeled_vesicle_data, num_vesicles = label(projected_vesicle_data)
    print(f"Total vesicles after projection: {num_vesicles}")
    return projected_vesicle_data, labeled_vesicle_data

def compute_vesicle_density(vesicle_data):
    vesicle_density = vesicle_data.sum(axis=0)
    return vesicle_density

def normalize_density(density):
    density_min = np.min(density)
    density_max = np.max(density)
    normalized_density = (density - density_min) / (density_max - density_min)
    return normalized_density

def display_neuron(neuron_data, vesicle_data=None):
    max_neuron_outline = compute_max_outline(neuron_data)
    vesicle_density = compute_vesicle_density(vesicle_data)
    vesicle_density_normalized = normalize_density(vesicle_density)
    heatmap_within_neuron = vesicle_density_normalized * max_neuron_outline
    display_image = np.ones_like(max_neuron_outline, dtype=np.float32)

    neuron_contours = find_contours(max_neuron_outline, level=0.5)
    for contour in neuron_contours:
        for coord in contour:
            y, x = coord.astype(int)
            display_image[y, x] = 0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, neuron_data.shape[2])
    ax.set_ylim(neuron_data.shape[1], 0)
    ax.axis("off")
    ax.imshow(display_image, cmap="gray", interpolation="nearest")
    ax.imshow(
        np.ma.masked_where(max_neuron_outline == 0, heatmap_within_neuron),
        cmap="viridis",
        interpolation="nearest",
        alpha=0.7,
        extent=[0, neuron_data.shape[2], neuron_data.shape[1], 0],
    )
    cbar = plt.colorbar(ax.imshow(
        np.ma.masked_where(max_neuron_outline == 0, heatmap_within_neuron),
        cmap="viridis",
        interpolation="nearest",
        alpha=0.7,
        extent=[0, neuron_data.shape[2], neuron_data.shape[1], 0],
    ), ax=ax, shrink=0.7)
    cbar.set_label("Normalized Vesicle Density", fontsize=12)
    plt.title("Neuron with Black Outline and Heatmap Restricted Inside")
    plt.show()
