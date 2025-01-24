import numpy as np
import pyvista as pv
from skimage.measure import label
from scripts.vesicle_load_tensor_files import load_data, calculate_distance_transform

scaling_factors = np.array([30, 8, 8])


def create_surface_mesh(positions):
    point_cloud = pv.PolyData(positions.astype(np.float32) * scaling_factors)
    if not point_cloud.points.any():
        return None
    return point_cloud.delaunay_2d()


def create_density_heatmap(vesicle_data, scaling_factors):
    vesicle_coords = np.column_stack(np.nonzero(vesicle_data))
    scaled_positions = vesicle_coords.astype(np.float32) * scaling_factors
    density, edges = np.histogramdd(scaled_positions, bins=(100, 100, 100))
    grid_x, grid_y, grid_z = np.meshgrid(
        edges[0][:-1], edges[1][:-1], edges[2][:-1], indexing='ij'
    )
    points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    density_values = density.ravel()
    nonzero_mask = density_values > 0
    points = points[nonzero_mask]
    density_values = density_values[nonzero_mask]
    point_cloud = pv.PolyData(points)
    point_cloud['Density'] = density_values
    return point_cloud


def visualize_data(neuron_mesh, vesicle_density_mesh):
    plotter = pv.Plotter()
    plotter.add_mesh(neuron_mesh, color='white', style='wireframe', opacity=0.001)
    plotter.add_mesh(vesicle_density_mesh, scalars='Density', cmap='hot', style='wireframe', opacity=1.0)
    plotter.add_axes()
    plotter.show_grid()
    plotter.set_scale(zscale=0.001, xscale=0.001, yscale=0.001)
    plotter.show_grid(xlabel='X axis (10 microns)', ylabel='Y axis (10 microns)', zlabel='Z axis (10 microns)')
    plotter.show()


def load_calculate_and_visualize_neuron_and_vesicles(neuron_file_path, vesicle_file_path):
    neuron_data, vesicle_data = load_data(neuron_file_path, vesicle_file_path)
    labeled_vesicles, num_vesicles = label(vesicle_data, return_num=True)
    neuron_positions = np.column_stack(np.nonzero(neuron_data))
    neuron_mesh = create_surface_mesh(neuron_positions)
    vesicle_density_mesh = create_density_heatmap(vesicle_data, scaling_factors)
    visualize_data(neuron_mesh, vesicle_density_mesh)
    print("Number of vesicle objects:", num_vesicles)
    print(np.percentile(calculate_distance_transform(neuron_data), [10, 50, 90]))
