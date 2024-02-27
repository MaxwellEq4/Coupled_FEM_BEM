#%% DEPENDENCIES
import FEM as fem
import MoM as mom
import Solvers as sol
import Utility as util
import Visualizations as vis
import matplotlib.pyplot as plt


class FEM_BEM_Solver:
    def __init__(self,N_boundary,n_rings, kb, epsi, mui, max_dim, resolution, radius):
        self.N_boundary = N_boundary
        self.n_rings = n_rings
        self.kb = kb
        self.epsi = epsi
        self.mui = mui
        self.max_dim = max_dim
        self.resolution = resolution
        self.radius = radius

    # Load mesh info
    def load_mesh_info(self):
        # Correctly calling util.loadMeshInfo with assumed parameters and setting instance attributes
        self.mesh_file_name, self.bnd_pts, self.bnd_seg, self.col_pts = util.loadMeshInfo(self.N_boundary, self.n_rings)

    # Generate meshgrid and plot
    def generate_meshgrid_with_plot(self):
        # Assuming util.generate_meshgrid_with_plot is correctly defined and returns expected values
        self.X, self.coordinates, self.X_masked, self.Y_masked, self.mask_within_radius = util.generate_meshgrid_with_plot(self.max_dim, self.resolution, self.radius, show_plot=False)

    # Process the mesh and instigate mappings
    def process_mesh_and_map(self):
        boundary_cell_node_coords = util.find_adjacent_boundary_cell_node_coordinates(self.mesh_file_name)
        filtered_coords = util.filter_nodes_on_circular_arc(boundary_cell_node_coords, self.radius)
        cell_midpoints = util.distribute_midpoints(filtered_coords)
        self.mapping_result = util.map_matching_coordinates(cell_midpoints,self.col_pts)

    # Get the FEniCS matrices
    def get_fenics_matrices(self):
        # Correctly access instance attributes with self and assign results to instance attributes
        self.K, self.B, self.T, self.node_labels, self.fenics_coordinates, self.dof_coordinates, self.boundary_dof, self.interior_dof = fem.dielectric_homogeneous_object(self.mesh_file_name, self.kb, self.epsi, self.mui)

    # Get the MoM matrices
    def get_mom_matrices(self):
        self.Q = mom.weak_mom(self.bnd_pts, self.bnd_seg, self.col_pts, self.kb)
        self.P = mom.strong_mom(self.bnd_pts, self.bnd_seg, self.col_pts, self.kb)

    # Reorder B to fit the N x N_s coupling matrix structure
    def reorder_B(self):
        self.reorder_B = util.reorder_vector_based_on_mapping(self.col_pts, self.B, self.mapping_result)
        self.B_DG0_test = util.populate_matrix_based_on_labels(self.reorder_B, self.node_labels)

    # Reorder P to fit the N_s x N coupling matrix structure
    def reorder_P(self):
        self.mapping = util.map_dof_to_bnd_pts(self.bnd_pts, self.dof_coordinates, self.node_labels)
        self.P_reordered = util.reorder_matrix_based_on_dof(self.P, self.mapping)
        self.P_reordered_zeroed = util.insert_zero_columns(self.P_reordered, self.node_labels)

    # Reorder Q based on FEniCS DOF
    def reorder_Q(self):
        self.Q_reordered = util.reorder_matrix_based_on_dof(self.Q, self.mapping)

    # Get the excitation vector & match it to the DOF
    def get_excitation_vector(self):
        self.b_vec = mom.plane_wave_excitation(self.kb, self.col_pts)
        self.b_vec_reordered = util.reorder_vector_based_on_dof(self.b_vec, self.mapping)

    # Solve the block system
    def solve_block_system(self):
        self.phi, self.psi = sol.solve_block_brute_force(self.K, self.B_DG0_test, self.P_reordered_zeroed, self.Q_reordered, self.b_vec)


    def run (self):
        self.load_mesh_info()
        self.generate_meshgrid_with_plot()
        self.process_mesh_and_map()
        self.get_fenics_matrices()
        self.get_mom_matrices()
        self.reorder_B()
        self.reorder_P()
        self.reorder_Q()
        self.get_excitation_vector()
        self.solve_block_system()
        return self.phi, self.psi, self.dof_coordinates