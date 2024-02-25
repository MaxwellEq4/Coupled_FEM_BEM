#%% Dependencies
import time
import numpy as np



#%% Direct solver

def solve_block_brute_force(K, B, P, Q, b_vec):
    """
    Inputs:
    
    """
    # begin timing
    t0 = time.time()
    # T_transposed = T.transpose()
    # Your adjustments
    #I = np.eye(P.shape)  # Assuming square matrices, creates an identity matrix of the same size as P

    # Translated Operation
    #A21 = B.T @ (D - I/2)  # @ is the matrix multiplication operator in Python
    # B_T = B.dot(T_transposed.toarray())
    #P_altered = np.transpose(B) + P

    # count non-zero elements in P_altered
    non_zero_count = np.count_nonzero(P)
    print(f"Number of non-zero elements in P_altered: {non_zero_count}")
    # P_T = P.dot(T.toarray())
    # Print the shapes of the matrices
    print("The shape of K is:", np.shape(K))
    print("The shape of B is:", np.shape(B))
    print("The shape of P is:", np.shape(P))
    print("The shape of Q is:", np.shape(Q))
    AA = np.block([
        [K, B],
        [P+np.transpose(B), Q]
    ])


    zeros = np.zeros(len(K))

    rhs = np.concatenate([zeros, b_vec])

    # Solve the system
    solution = np.linalg.solve(AA, rhs)

    # Splitting the solution back into phi and psi
    phi = solution[:K.shape[0]]
    psi = solution[K.shape[0]:]
    print("The shape of phi is:", np.shape(phi))
    print("The shape of psi is:", np.shape(psi))
    # end timing
    t1 = time.time()
    print(f"Time to solve coupled system: {t1 - t0:.2f} seconds")

    return phi, psi


