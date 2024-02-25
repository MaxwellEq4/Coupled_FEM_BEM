import numpy as np
import matplotlib.pyplot as plt

def loadMeshInfo(N_boundary, n_rings):
    # Construct the directory and filenames based on the given parameters
    subdirectory = f"meshes/circles/BND{N_boundary:02d}"
    base_filename = f"mesh_{N_boundary}_{n_rings}"
    xml_filename = f"{subdirectory}/{base_filename}.xml"

    # Construct paths for the .npy files
    bnd_pts_file = f"{subdirectory}/{base_filename}_bnd_pts.npy"
    bnd_seg_file = f"{subdirectory}/{base_filename}_bnd_seg.npy"
    col_pts_file = f"{subdirectory}/{base_filename}_col_pts.npy"

    # Load the .npy files
    bnd_pts = np.load(bnd_pts_file)
    bnd_seg = np.load(bnd_seg_file)
    col_pts = np.load(col_pts_file)

    # Return the filename for the mesh and the loaded arrays
    return xml_filename, bnd_pts, bnd_seg, col_pts


def generate_meshgrid_with_plot(max_dim, resolution, radius, show_plot=False):
    """
    Generates a 2D meshgrid with given max dimensions and resolution,
    subtracts all points within a given radius from the center,
    and optionally shows a scatter plot of the resulting grid.

    Parameters:
    - max_dim (tuple): The maximum dimensions (x_max, y_max) of the grid.
    - resolution (int): The number of evenly spaced samples within the range.
    - radius (float): The radius of the circle within which points are to be subtracted.
    - show_plot (bool): If True, displays a scatter plot of the meshgrid.

    Returns:
    - coordinates: The coordinates outside the specified radius.
    - X_masked, Y_masked: The meshgrid arrays with points within the circle subtracted.
    - mask_within_radius: A mask indicating points within the specified radius.
    """

    # Create a linear space for x and y dimensions
    x = np.linspace(-max_dim[0] / 2, max_dim[0] / 2, resolution)
    y = np.linspace(-max_dim[1] / 2, max_dim[1] / 2, resolution)

    # Create a 2D meshgrid
    X, Y = np.meshgrid(x, y)

    # Calculate rho for each point
    rho = np.sqrt(X**2 + Y**2)

    # Create a mask where points within the radius are True
    mask_within_radius = rho <= radius

    # Subtract points within the circle (set them to NaN)
    X_masked = np.where(mask_within_radius, np.nan, X)
    Y_masked = np.where(mask_within_radius, np.nan, Y)

    # Extract coordinates where mask is False (outside the radius)
    mask = ~mask_within_radius
    coordinates = np.column_stack((X[mask], Y[mask]))

    if show_plot:
        # Plot the points outside the circle
        plt.figure(figsize=(8, 8))
        plt.scatter(X_masked[mask], Y_masked[mask], color='blue', label='Points outside the circle')

        # Plot the circle for reference
        circle = plt.Circle((0, 0), radius, color='red', fill=False, label='Circle boundary')
        plt.gca().add_artist(circle)

        # Set the plot limits and aspect
        plt.xlim(-max_dim[0] / 2, max_dim[0] / 2)
        plt.ylim(-max_dim[1] / 2, max_dim[1] / 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.title('Scatter plot of points outside a radius')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.show()

    return X,coordinates, X_masked, Y_masked, mask_within_radius