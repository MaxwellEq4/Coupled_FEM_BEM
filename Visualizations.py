import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.tri import Triangulation



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_scattered_field(domain_coordinates, scattered_field, fenics_coordinates, phi, k):
    
    assert domain_coordinates.shape[0] == scattered_field.size

    # Add the incident field to the scattered field
    E_inc_external = np.exp(-1j * k * domain_coordinates[:, 0])
    E_inc_internal = np.exp(-1j * k * fenics_coordinates[:, 0])

    total_field_external = scattered_field + E_inc_external
    total_field_internal = phi 

    # make a inspection plot of e_inc_external
    plt.figure(figsize=(8, 8))
    plt.scatter(domain_coordinates[:, 0], domain_coordinates[:, 1], c=E_inc_external.real, cmap='coolwarm', edgecolor='none')
    plt.colorbar(label='Real part')
    plt.title('Incident field (Real part)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()


    # Combine both internal and external field values
    combined_field_real = np.concatenate((total_field_external.real, total_field_internal.real))
    combined_field_imag = np.concatenate((total_field_external.imag, total_field_internal.imag))
    
    # Normalize real parts to [0, 1]
    min_real = combined_field_real.min()
    max_real = combined_field_real.max()
    normalized_real = (combined_field_real - min_real) / (max_real - min_real)

    # Normalize imaginary parts to [0, 1]
    min_imag = combined_field_imag.min()
    max_imag = combined_field_imag.max()
    normalized_imag = (combined_field_imag - min_imag) / (max_imag - min_imag)

    # Split the normalized fields back into external and internal components
    total_field_external_real_normalized = normalized_real[:scattered_field.size]
    total_field_internal_real_normalized = normalized_real[scattered_field.size:]
    
    total_field_external_imag_normalized = normalized_imag[:scattered_field.size]
    total_field_internal_imag_normalized = normalized_imag[scattered_field.size:]

    # Create a 1x2 subplot figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # Adjust the figure size as needed

    # Plot for real parts on the first subplot
    scatter_plot_real = axs[0].scatter(domain_coordinates[:, 0], domain_coordinates[:, 1], c=total_field_external.real, cmap='coolwarm', edgecolor='none')
    axs[0].scatter(-fenics_coordinates[:, 0], fenics_coordinates[:, 1], c=total_field_internal.real, cmap='coolwarm', edgecolor='none')
    fig.colorbar(scatter_plot_real, ax=axs[0])
    axs[0].set_title('Total field (Real part)')
    axs[0].set_xlabel('X coordinate')
    axs[0].set_ylabel('Y coordinate')

    # Plot for imaginary parts on the second subplot
    scatter_plot_imag = axs[1].scatter(domain_coordinates[:, 0], domain_coordinates[:, 1], c=total_field_external.imag, cmap='coolwarm', edgecolor='none')
    axs[1].scatter(-fenics_coordinates[:, 0], fenics_coordinates[:, 1], c=total_field_internal.imag, cmap='coolwarm', edgecolor='none')
    fig.colorbar(scatter_plot_imag, ax=axs[1])
    axs[1].set_title('Total field (Imaginary part)')
    axs[1].set_xlabel('X coordinate')
    axs[1].set_ylabel('Y coordinate')

    plt.tight_layout()  # Adjust the layout to not overlap
    plt.show()




def plot_exp_kx(k, X_masked):
    """
    Plots the real part of exp(-1j * k * X_masked).

    Parameters:
    - k (float): The wavenumber.
    - X_masked (ndarray): The x-coordinates of the meshgrid, possibly with NaN values.

    The function calculates -exp(-1j * k * X_masked) and plots its real part.
    """
    # Perform the calculation on the X_masked grid
    result_matrix = np.exp(-1j * k * X_masked)

    # Plotting the real part of the result using imshow
    plt.figure(figsize=(8, 8))
    plt.imshow(np.real(result_matrix), extent=(-3, 3, -3, 3), origin='lower', cmap='plasma')
    plt.colorbar(label='Real part')
    plt.title(f'Plane wave with k = {k}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(False)
    plt.show()



def plotBoundaryAndCoordinates(bnd_pts, bnd_seg, coordinates):
    plt.figure(figsize=(8, 8))
    plt.title('Boundary and Coordinates Plot')

    # Plot the boundary points
    plt.scatter(bnd_pts[:, 0], bnd_pts[:, 1], color='red', label='Boundary Points')

    # Draw the segments
    for seg in bnd_seg:
        p1, p2 = bnd_pts[seg[0]], bnd_pts[seg[1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red')

    # Plot the additional coordinates
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='blue', label='Coordinates')

    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # To keep the aspect ratio square
    plt.show()


def plot_dofs_on_mesh(V, node_labels, fenics_coordinates):
    # Create a triangulation from the mesh coordinates
    triangulation = Triangulation(fenics_coordinates[:, 0], fenics_coordinates[:, 1])
    
    # Extract DoF coordinates for plotting
    dof_coordinates = V.tabulate_dof_coordinates()

    # Remove the 3rd coordinate (z) as it is not needed for 2D plotting
    dof_coordinates = dof_coordinates[:, :2]
    
    # Initialize lists to hold boundary and interior DoF coordinates
    boundary_dof_coords = []
    interior_dof_coords = []
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot the mesh
    plt.triplot(triangulation, 'k-', lw=0.5, alpha=0.5)  # Mesh in black, semi-transparent

    # Overlay DoFs on the mesh
    for i, label in enumerate(node_labels):
        if label == 'b':
            plt.plot(dof_coordinates[i, 0], dof_coordinates[i, 1], 'ro')  # Boundary DoFs in red
            boundary_dof_coords.append(dof_coordinates[i])
        else:
            plt.plot(dof_coordinates[i, 0], dof_coordinates[i, 1], 'bo')  # Interior DoFs in blue
            interior_dof_coords.append(dof_coordinates[i])
        # Uncomment the line below if you want to annotate the DoFs
        plt.text(dof_coordinates[i, 0], dof_coordinates[i, 1], str(i), fontsize=8)  # Annotate DoFs with their indices
    
    # Print the number of boundary and interior DoFs
    num_boundary_dofs = len(boundary_dof_coords)
    num_interior_dofs = len(interior_dof_coords)
    print(f"Number of boundary DoFs: {num_boundary_dofs}")
    print(f"Number of interior DoFs: {num_interior_dofs}")

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Mesh with Degrees of Freedom (DoFs) Overlay')
    plt.legend(['Mesh', 'Boundary DoFs', 'Interior DoFs'])
    plt.axis('equal')  # Ensure the aspect ratio is equal to avoid distortion
    plt.show()

    # Convert lists to numpy arrays before returning
    boundary_dof_coords = np.array(boundary_dof_coords)
    interior_dof_coords = np.array(interior_dof_coords)

    # Returning boundary DoF coordinates
    return boundary_dof_coords, interior_dof_coords, dof_coordinates