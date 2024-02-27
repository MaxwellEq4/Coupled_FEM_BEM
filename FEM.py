#%% Dependencies
import time
import numpy as np
from scipy.sparse import coo_matrix
from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri


#%% FEM matrices
def dielectric_homogeneous_object(filename, k, epsilon, mu):
    # Begin timing
    t0 = time.time()

    # Load mesh from file
    mesh = Mesh(filename)

    # Generate mesh and define function space for K
    V = FunctionSpace(mesh, 'P', 1)
    
    # Define another function space for B using DG0
    V_dg = FunctionSpace(mesh, 'DG', 0)

    # Assuming 'mesh' and 'boundary_cells' have been defined earlier in your code

    # Extract mesh coordinates and topology
    coordinates = mesh.coordinates()
    triangles = mesh.cells()

    # Create a matplotlib triangulation object
    triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)

    # Calculate cell centers manually for annotation
    cell_centers = np.mean(coordinates[triangles], axis=1)

    # Identify boundary cells
    boundary_cells = set()
    for f in facets(mesh):
        if f.exterior():  # Check if the facet is on the boundary
            for c in cells(f):
                boundary_cells.add(c.index())


    # Define trial and test functions for K
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # Define trial function for B
    v_dg = TestFunction(V_dg)

    # Extract the coordinates of the mesh nodes
    fenics_coordinates = mesh.coordinates()

    # Define constants
    k_const = Constant(k)
    epsilon_const = Constant(epsilon)
    inv_mu = Constant(1/mu)

    # Formulate the weak problem for K
    a = inv_mu * inner(grad(u), grad(v)) * dx - k_const**2 * epsilon_const * u * v * dx

    # Assemble stiffness matrix K
    K = assemble(a)

    # Formulate the weak problem for B, using the DG space and integrating over the boundary
    m = v_dg * ds  # 
    B = assemble(m)

    # Convert to numpy arrays, if necessary
    K = K.array()
    B_array = B.get_local()  # Use get_local() for vectors    

    # Get boundary nodes
    bc = DirichletBC(V, Constant(0), 'on_boundary')
    
    bc_dofs = bc.get_boundary_values().keys()

    # Extract DoF coordinates for plotting
    dof_coordinates = V.tabulate_dof_coordinates()

    # Remove the 3rd coordinate (z) as it is not needed for 2D plotting
    dof_coordinates = dof_coordinates[:, :2]
    
    # Initialize lists to hold boundary and interior DoF coordinates
    boundary_dof_coords = []
    interior_dof_coords = []

    # List to store boundary ('b') and interior ('i') node labels
    node_labels = ['i'] * V.dim()  # Initialize all nodes as interior
    for dof in bc_dofs:
        node_labels[dof] = 'b'  # Label boundary nodes
    
    # End timing
    t1 = time.time()
    print(f"Time to assemble FEM matrices: {t1 - t0:.2f} seconds")

    # Get the total number of DoFs and the number of boundary DoFs
    total_dofs = V.dim()
    num_boundary_dofs = len(bc_dofs)

    # Create row indices for non-zero entries in the trace matrix
    # This maps each boundary DoF to its own row
    row_indices = np.arange(num_boundary_dofs)

    # Create corresponding column indices using the boundary DoFs
    col_indices = np.array(list(bc_dofs))

    # Create the data for the non-zero entries
    data = np.ones_like(row_indices, dtype=float)

    # Create the sparse trace matrix
    trace_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(num_boundary_dofs, total_dofs))

    return K, B,trace_matrix, node_labels, fenics_coordinates, dof_coordinates, boundary_dof_coords, interior_dof_coords

def epsilon_mu_function(mesh, radii, epsilon_values, mu_values):
    V = FunctionSpace(mesh, 'P', 1)
    epsilon = Function(V)
    mu = Function(V)
    
    x = mesh.coordinates()[:, 0]  # Assuming cylinder axis aligned with x
    y = mesh.coordinates()[:, 1]
    radial_distances = np.sqrt(x**2 + y**2)  # Example for 2D mesh
    
    # Assign epsilon and mu values based on radial_distances
    for i, radius in enumerate(radii):
        within_ring = np.logical_and(radial_distances >= radius[0], radial_distances < radius[1])
        epsilon.vector()[within_ring] = epsilon_values[i]
        mu.vector()[within_ring] = mu_values[i]
    
    return epsilon, mu

def dielectric_inhomogeneous_object(mesh, k, radii, epsilon_values, mu_values):
    # Begin timing
    t0 = time.time()

    # Generate mesh and define function space
    V = FunctionSpace(mesh, 'P', 1)
    
    # Get the fenics coordinates
    fenics_coordinates = mesh.coordinates()

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define variable material properties
    epsilon, mu = epsilon_mu_function(mesh, radii, epsilon_values, mu_values)
    inv_mu = project(1/mu, V)  # Inverse of mu, projected to the same function space for consistency

    # Define constants
    k = Constant(k)

    # Formulate the weak problem with spatially varying epsilon and mu
    a = k**2 * dot(grad(u), grad(v)) * dx + epsilon * inv_mu * u * v * dx

    # Assemble stiffness matrix and mass matrix
    K = assemble(a)
    m = u * v * ds
    B = assemble(m)
    bc = DirichletBC(V, Constant(0), 'on_boundary')
    bc.apply(K)


    # Convert to numpy arrays, if necessary
    B = B.array()
    K = K.array()

    # Get boundary nodes
    bc_dofs = bc.get_boundary_values().keys()

    # get dof coordinates
    def dof_coordinates(V):
        # Get the DoF map
        dofmap = V.dofmap()
        # Get the coordinates of all the DoFs
        coordinates = V.tabulate_dof_coordinates()
        return coordinates

    dof_coordinates = dof_coordinates(V)

    # List to store boundary ('b') and interior ('i') node labels
    node_labels = ['i'] * V.dim()  # Initialize all nodes as interior
    for dof in bc_dofs:
        node_labels[dof] = 'b'  # Label boundary nodes



    
    # End timing
    t1 = time.time()
    print(f"Time to assemble FEM matrices: {t1 - t0:.2f} seconds")

    # Get the total number of DoFs and the number of boundary DoFs
    total_dofs = V.dim()
    num_boundary_dofs = len(bc_dofs)

    # Create row indices for non-zero entries in the trace matrix
    # This maps each boundary DoF to its own row
    row_indices = np.arange(num_boundary_dofs)

    # Create corresponding column indices using the boundary DoFs
    col_indices = np.array(list(bc_dofs))

    # Create the data for the non-zero entries
    data = np.ones_like(row_indices, dtype=float)
    # Assuming `bc_dofs` needs to be sorted
    col_indices = np.sort(np.array(list(bc_dofs)))

    # Now create the trace matrix with sorted boundary DoF column indices
    trace_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(num_boundary_dofs, total_dofs))


    return K, B,trace_matrix, node_labels, fenics_coordinates, dof_coordinates(V)
