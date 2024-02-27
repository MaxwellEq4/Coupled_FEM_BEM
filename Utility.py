# DEPENDENCIES: 
import numpy as np
import matplotlib.pyplot as plt
from fenics import Mesh, MeshFunction, SubDomain, cells, facets, vertices
# MESH & DOMAIN INITIALIZATION

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


# MESH PROCESSING
def find_adjacent_boundary_cell_node_coordinates(mesh_filename):
    # Load the mesh from the given file
    mesh = Mesh(mesh_filename)
    
    # Initialize a dictionary to store the node coordinates of boundary-adjacent cells
    boundary_cell_node_coordinates = {}
    
    # Create a mesh function to mark the boundary facets
    boundary_facets = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundary_facets.set_all(0)
    
    class Boundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    
    # Mark the boundary facets
    Boundary().mark(boundary_facets, 1)
    
    # Iterate over all cells in the mesh
    for cell in cells(mesh):
        # Check if any facet of the cell is on the boundary
        for facet in facets(cell):
            if boundary_facets[facet.index()] == 1:
                # Store the node coordinates for this boundary-adjacent cell
                # Only take the x and y coordinates
                node_coords = [vertex.point().array()[:2] for vertex in vertices(cell)]
                boundary_cell_node_coordinates[cell.index()] = node_coords
                break  # Once a boundary facet is found, no need to check other facets of this cell


    return boundary_cell_node_coordinates

def filter_nodes_on_circular_arc(boundary_cell_node_coords, radius, center=(0,0), tolerance=1E-6):
    filtered_node_coords = {}
    
    # Calculate the squared radius for comparison (to avoid square roots)
    radius_squared = radius**2
    
    for cell_index, node_coords_list in boundary_cell_node_coords.items():
        filtered_coords_list = []
        for node_coords in node_coords_list:
            # Calculate squared distance from the node to the circle's center
            dx = node_coords[0] - center[0]
            dy = node_coords[1] - center[1]
            distance_squared = dx**2 + dy**2
            
            # Check if the node lies on the circle (within the tolerance)
            if np.abs(distance_squared - radius_squared) <= tolerance**2:
                filtered_coords_list.append(node_coords)
        
        # If there are nodes on the circle for this cell, add them to the dictionary
        if filtered_coords_list:
            filtered_node_coords[cell_index] = filtered_coords_list
    
    return filtered_node_coords

def distribute_midpoints(cell_vertex_coordinates):
    # Initialize a dictionary to store the midpoints for each cell
    cell_midpoints = {}
    
    # Iterate over each cell and its vertex coordinates
    for cell_index, vertices in cell_vertex_coordinates.items():
        midpoints = []
        
        # Calculate midpoints for each segment in the cell
        num_vertices = len(vertices)
        for i in range(num_vertices):
            # Current vertex
            start = vertices[i]
            # Next vertex (wrapping around to the first for the last segment)
            end = vertices[(i + 1) % num_vertices]
            
            # Calculate the midpoint
            midpoint = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]
            midpoints.append(midpoint)
        
        # Store the midpoints for the cell
        cell_midpoints[cell_index] = midpoints
    
    # remove doubles
    for cell_index, midpoints in cell_midpoints.items():
        cell_midpoints[cell_index] = list(set([tuple(midpoint) for midpoint in midpoints]))
    
    return cell_midpoints

def map_matching_coordinates(cell_midpoints, col_pts):
    # Flatten the cell_midpoints dictionary to a list of tuples (midpoint, cell_index)
    flat_midpoints = [(midpoint, cell_index) for cell_index, midpoints in cell_midpoints.items() for midpoint in midpoints]

    # Initialize the mapping result
    mapping_result = {}

    # Iterate over col_pts and compare with each midpoint
    for col_idx, col_pt in enumerate(col_pts):
        for midpoint, cell_index in flat_midpoints:
            # Check if the coordinates match (considering a tolerance for floating-point comparison)
            if np.allclose(col_pt, midpoint, atol=1e-8):
                # If a match is found, record the mapping
                mapping_result[col_idx] = cell_index
                break  # Stop searching once a match is found for this col_pt

    return mapping_result

def map_dof_to_bnd_pts(bnd_pts, dof, node_labels):
    # Identify indices in dof where node_labels is 'b'¨
    node_labels = np.array(node_labels)
    b_indices = np.where(node_labels == 'b')[0]    
    #print(b_indices)
    
    # identify the indices of i

    # Filter dof points that are labeled as 'b'
    dof_b = dof[b_indices]
    
    # Initialize an empty mapping dictionary: dof index to bnd_pts index
    mapping = {}
    
    # Iterate over each dof point labeled as 'b'
    for i, dof_point in enumerate(dof_b):
        # Find the matching point in bnd_pts
        for j, bnd_point in enumerate(bnd_pts):
            if np.array_equal(dof_point, bnd_point):
                # Map dof index to bnd_pts index
                mapping[b_indices[i]] = j
                break
    
    return mapping
#%% REORDERING OF MATRICES & VECTORS
def reorder_vector_based_on_mapping(col_pts,vector, mapping):
    # Initialize a new vector of the same length as the original vector
    new_vector = np.zeros(len(col_pts))
    
    # Iterate over the mapping and reorder the vector accordingly
    for new_index, original_index in mapping.items():
        new_vector[new_index] = vector[original_index]
    
    return new_vector

def populate_matrix_based_on_labels(input_vector, node_labels):
    # Determine the matrix size
    rows = len(node_labels)
    cols = len(input_vector)
    
    # Initialize the matrix with zeros
    matrix = np.zeros((rows, cols))
    
    # Track the current column to fill in the matrix
    current_col = 0
    
    # Iterate over node_labels to fill the matrix
    for i, label in enumerate(node_labels):
        if label == 'b' and current_col < cols:
            # If the label is 'b', populate the matrix at the current column and row i
            matrix[i, current_col] = input_vector[current_col]
            # Move to the next column for the next 'b'
            current_col += 1
    
    return matrix

def reorder_matrix_based_on_dof(original_matrix, mapping):
    # Determine the size of the matrix
    N = original_matrix.shape[0]
    
    # Initialize a new matrix of the same size
    reordered_matrix = np.zeros_like(original_matrix)
    
    # Create a list of new column indices based on the sorted dof indices (values of the mapping)
    new_order = [mapping[k] for k in sorted(mapping.keys())]
    
    # Rearrange the columns of the original matrix to the new order
    for new_idx, old_idx in enumerate(new_order):
        reordered_matrix[:, new_idx] = original_matrix[:, old_idx]
    
    return reordered_matrix

def insert_zero_columns(input_matrix, node_labels):
    # Determine the total number of columns for the new matrix
    node_labels = np.array(node_labels)

    total_columns = len(node_labels)
    # Initialize the new matrix with zeros
    # Number of rows from the input matrix, number of columns from node_labels
    new_matrix = np.zeros((input_matrix.shape[0], total_columns),dtype=complex)
    
    # Track the current column in the input matrix to be added to the new matrix
    input_matrix_col = 0
    
    # Iterate through node_labels to determine placement of input matrix columns or zeros
    for i, label in enumerate(node_labels):
        if label == 'b':
            new_matrix[:, i] = input_matrix[:, input_matrix_col]
            input_matrix_col += 1  # Move to the next column in the input matrix for the next 'b'
        # Note: 'i' labeled columns are already initialized to zero, so no action needed for 'i'
    
    return new_matrix


def reorder_vector_based_on_dof(original_vector, mapping):
    # Determine the size of the vector
    N = original_vector.shape[0]
    
    # Initialize a new vector of the same size
    reordered_vector = np.zeros_like(original_vector)
    
    # Create a list of new indices based on the sorted dof indices (values of the mapping)
    new_order = [mapping[k] for k in sorted(mapping.keys())]
    
    # Rearrange the elements of the original vector to the new order
    for new_idx, old_idx in enumerate(new_order):
        reordered_vector[new_idx] = original_vector[old_idx]
    
    return reordered_vector

#%% SORTING

def get_boundary_phi(phi, node_labels):
    # We need to map the phi values to the boundary nodes. Psi are only on the boundary nodes
    boundary_phi = []
    interior_phi = []
    for idx, label in enumerate(node_labels):
        if label == 'b':
            # Append only the phi value for boundary nodes
            boundary_phi.append(phi[idx])
        else:
            # Assuming you want to do a similar action for interior nodes
            interior_phi.append(phi[idx])
    boundary_phi = np.array(boundary_phi)
    return boundary_phi
