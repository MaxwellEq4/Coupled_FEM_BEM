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
    # end timing
    t1 = time.time()
    print(f"Time to solve coupled system: {t1 - t0:.2f} seconds")

    return phi, psi


