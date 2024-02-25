import numpy as np
from numpy.linalg import norm
import scipy.special as sps
import time
import sys
import numpy as np
from numpy.linalg import norm
from numpy.polynomial.legendre import leggauss
#from numba import jit, njit, complex128
from GL_logarithmic import GL_log

from scipy.special import hankel2
from scipy.integrate import quad

import matplotlib.pyplot as plt


#%% STRONG MOM MATRIX


gamma = 0.57721566490153286060651

def compute_normals(bnd_pts, bnd_seg):
    """
    Computes the outward unit normals for each segment in a list of boundary segments.
    
    Parameters:
    - bnd_pts : A list of points [x0, y0], [x1, y1], ...
    - bnd_seg : A list of segments [[0, 1], [1, 2], [2, 3], ...]
    
    Returns:
    - n_hat : An array of outward unit normals for each segment.
    """
    num_seg = len(bnd_seg)
    n_hat = np.zeros((num_seg, 2), dtype=np.double)
    
    for k in range(num_seg):
        # get start and end point indices of the segment k
        idx0, idx1 = bnd_seg[k]
        
        # get start and end point of the segment k
        r0, r1 = bnd_pts[idx0], bnd_pts[idx1]
        
        # tangential vector (reversed to r1 - r0 for outward normal)
        t = np.array(r1) - np.array(r0)
        
        # vector normal to t, thus perpendicular to the boundary, pointing outwards
        n = np.array([-t[1], t[0]])
        
        # unit normal
        hat_n = n / norm(n)
        
        n_hat[k] = hat_n

    return n_hat

def bf1(t):
    return 0.5*(1 - t)

def bf2(t):
    return 0.5*(1 + t)

def quadcpp(f,q,w):

    J = 0.0
    for m in range(len(q)):
        J += w[m]*f(q[m])

    return J

def H20(x):
    return sps.hankel2(0,x)

def H21(x):
    return sps.hankel2(1,x)

def J0(x):
    return sps.jv(0,x)

def J1(x):
    return sps.jv(1,x)

def g(x):
    return 0.25*1j * H20(x)

def gsing(x):
    return 0.25*1j * (-1j * 2/np.pi * np.log(x/2))

def dg(x):
    return -0.25*1j * H21(x)

def dgsing(x):
    return -0.25*1j * (1j * 2/np.pi * 1/x)

@np.vectorize
def greg(x):

    if (abs(x) != 0.0):
        return g(x) - gsing(x)
    
    else:
        return 0.25*1j * (J0(x) - 1j * 2/np.pi * gamma)
    
@np.vectorize
def dgreg(x):

    if (abs(x) != 0.0):
        return dg(x) - dgsing(x)
    
    else:
        return complex(0.0)

def D(s,sp):
    return norm(s-sp)

def grad_D(s, sp):
    d = s-sp
    n = norm(d)

    if (n>0) : 
        ud = d/n
        return ud
    else:
        return d


from numpy.polynomial.legendre import leggauss

def strong_mom(bnd_pts, bnd_seg,col_pts,kb):
    # begin timing
    t0 = time.time()

    num_seg = len(bnd_seg)
    rk, wreg = leggauss(5)
    # Initialize the matrix
    M = np.zeros((num_seg, num_seg), dtype=complex)

    # Compute the normals
    n_hat = compute_normals(bnd_pts, bnd_seg)

    # Fill the matrix

    for l in range(num_seg):
        rc = col_pts[l]

        for k in range(num_seg):
            idx0, idx1 = bnd_seg[k]

            r0, r1 = bnd_pts[idx0], bnd_pts[idx1]

            # tangential vector
            dr = 0.5*(r1 - r0)

            # center of the segment
            r2 = 0.5*(r1 + r0)

            # jacobian
            jac = np.linalg.norm(dr)

            # normal unit vector
            n = n_hat[k]

            # parameterized boundary: -1 <= t <= 1
            r  = lambda t: r2 + t*dr

            if l != k:

                # integrand
                f1 = lambda t : bf1(t) * n.dot(dg(kb*D(rc,r(t))) * grad_D(rc,r(t)))
                f2 = lambda t : bf2(t) * n.dot(dg(kb*D(rc,r(t))) * grad_D(rc,r(t)))

                mom_int1 = kb * quadcpp(f1, rk, wreg)
                mom_int2 = kb * quadcpp(f2, rk, wreg)

                M[l][idx0] = M[l][idx0] + mom_int1 * jac
                M[l][idx1] = M[l][idx1] + mom_int2 * jac

            else:

                mom_int_sing1 = bf1(0.0) * 0.5
                mom_int_sing2 = bf2(0.0) * 0.5

                # regular part of the integrand
                freg1 = lambda t : kb * bf1(t) * n.dot( dgreg(kb*D(rc,r(t))) * grad_D(rc,r(t)) )
                freg2 = lambda t : kb * bf2(t) * n.dot( dgreg(kb*D(rc,r(t))) * grad_D(rc,r(t)) )

                mom_int_reg1 = kb * quadcpp(freg1, rk, wreg)
                mom_int_reg2 = kb * quadcpp(freg2, rk, wreg)

                # singular part of the integrand¨
                mom_int1 = mom_int_sing1 + mom_int_reg1 * jac
                mom_int2 = mom_int_sing2 + mom_int_reg2 * jac

                M[l][idx0] = M[l][idx0] + mom_int1
                M[l][idx1] = M[l][idx1] + mom_int2

    # end timing
    t1 = time.time()
    print(f"Time to assemble strong MoM matrix: {t1 - t0:.2f} seconds")
    return M



#%% WEAK MOM MATRIX

pi = np.pi
pi2 = pi**2

def H20 (x):
    return hankel2(0, x)

def H21 (x):
    return hankel2(1, x)

def g(x):
    return 0.25j * H20(x)

def dgdn(kb, x):
    return -0.25j * kb * H21(x)


def weak_mom(bnd_pts, bnd_seg,col_pts, kb, log_results = False):
    # begin timing
    t0 = time.time()
    # determine the number of segments
    num_seg = len(bnd_seg)

    #initialize the matrix
    A = np.zeros((num_seg, num_seg), dtype = np.complex128)

    # Function to handle logging
    def log(message, file=None):
        if log_results:
            print(message, file=file)


    # Open a file for logging if needed
    log_file = None
    if log_results:
        log_file = open('strong_mom_log.txt', 'w')

    # fill the matrix
    for i in range(0, num_seg):
        #start and end points of the segment
        r0, r1 = bnd_pts[bnd_seg[i]]

        # centre of the segment
        r_mid = 0.5 * (r0 + r1)


        dr = 0.5 * (r1 - r0)

        # jacobian
        jac = norm(dr)

        for j in range(0, num_seg):
            log("\n", file=log_file)
            log("Inspecting segment pair ({}, {})".format(i, j), file=log_file)
            log("Which goes from {} to {}".format(bnd_pts[bnd_seg[j]][0], bnd_pts[bnd_seg[j]][1]), file=log_file)
            log("Segment {} has centre {} and jacobian {}".format(i, r_mid, norm(dr)), file=log_file)
            # get the collocation point
            rc = col_pts[j]

            D = lambda s,sp : kb * norm(s-sp)

            r = lambda t : r_mid + t*dr

            f = lambda t : g(D(rc, r(t))) 

            log("Current colloaction point is {}".format(rc), file=log_file)
            
            if i == j:
                t0 = [0.0]

                mom_int,_ = quad(f, a = -1, b = 1, points = t0, complex_func=True)

                A[i,j] = jac * mom_int
                log(f"Matrix element A[{i}, {j}] = {A[i, j]}", file=log_file)

            else:
                mom_int,_ = quad(f, a = -1, b = 1, complex_func=True)

                A[i,j] = jac * mom_int
                log(f"Matrix element A[{i}, {j}] = {A[i, j]}", file=log_file)
    # Close the log file if it was opened
    if log_file:
        log_file.close()

    # end timing
    t1 = time.time()
    #print(f"Time to complete operation: {t1 - t0:.2f} seconds")
    return A

#%% SCATTERED FIELD

from scipy.integrate import quad

def H20(x):
    return sps.hankel2(0,x)

def H21(x):
    return sps.hankel2(1,x)

def g(x):
    return 0.25*1j * H20(x)

def dgdn(x):
    return -0.25*1j * H21(x)

def grad_D(r,rp):
    d = r-rp
    n = norm(d)
    if (n>0):
        ud = d/n
        return ud
    else:
        return d

#@jit
def calculate_scattered_field(domain_coordinates, boundary_phi, boundary_psi, bnd_pts, bnd_seg, col_pts, k):
    scattered_field = np.zeros(len(domain_coordinates), dtype=complex)
    num_obs = len(domain_coordinates)
    num_seg = len(col_pts)

    for obs in range(num_obs):
        obs_point = domain_coordinates[obs]

        # Progress bar update
        progress = (obs + 1) / num_obs
        sys.stdout.write('\rProgress: [{0:50s}] {1:.1f}%'.format('#' * int(progress * 50), progress * 100))
        sys.stdout.flush()

        for seg in range(num_seg):
            r0, r1 = bnd_pts[bnd_seg[seg]]
            r2 = 0.5 * (r1 + r0)
            dr = 0.5 * (r1 - r0)
            n = np.array([dr[1], -dr[0]])
            hat_n = n / norm(n)
            jac = norm(dr)

            D = lambda r, rp: k * norm(r - rp)
            r = lambda t: r2 + t * dr

            # first integrand
            f1 = lambda t: g(D(obs_point, r(t))) * boundary_psi[seg]
            # second integrand
            f2 = lambda t: k * hat_n.dot(dgdn(k * D(obs_point, r(t))) * grad_D(obs_point, r(t))) * boundary_phi[seg]

            int1, _ = quad(f1, -1, 1, complex_func=True)
            int2, _ = quad(f2, -1, 1, complex_func=True)

            # Accumulate contributions
            scattered_field[obs] += (int1 + int2) * jac

    # Ensure the progress bar goes to the next line after completion
    sys.stdout.write('\n')
    print("Scattered field calculated")
    return scattered_field


def plane_wave_excitation(ka,coordinates):
    b_vec = []
    for x, y in coordinates:
        phi = np.arctan2(y, x)
        b_vec.append(np.exp(-1j * ka * np.cos(phi)))
    return np.array(b_vec)
