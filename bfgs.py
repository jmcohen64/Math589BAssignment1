#----------------------------------------------------------------
# File:     bfgs.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@arizona.edu)
# Date:     Sat Jan 25 20:17:01 2025
# Copying:  (C) Marek Rychlik, 2020. All rights reserved.
# 
#----------------------------------------------------------------
# This is the code from the textbook

import numpy as np

def rosenbrock(x):
    """Compute the Rosenbrock function."""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosenbrock_grad(x):
    """Compute the gradient of the Rosenbrock function."""
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    for i in range(1, len(x) - 1):
        grad[i] = 200 * (x[i] - x[i - 1]**2) - \
                  400 * x[i] * (x[i + 1] - x[i]**2) - 2 * (1 - x[i])
    grad[-1] = 200 * (x[-1] - x[-2]**2)
    return grad

def bfgs(x0, func, grad_func, n_beads, tol=1e-6, max_iter=1000):
    """
    Minimize a function using the BFGS algorithm.

    Parameters:
        x0 (array-like): Initial guess.
        func (callable): Objective function.
        grad_func (callable): Gradient of the objective function.
        tol (float): Tolerance for the norm of the gradient.
        max_iter (int): Maximum number of iterations.

    Returns:
        x (array-like): Estimated position of the minimum.
        f_val (float): Function value at the minimum.
    """
    x = x0.copy()
    n = len(x)
    I = np.eye(n)
    H = I  # Initial inverse Hessian approximation
    g = grad_func(x, n_beads)
    trajectory = [x.copy()]
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            print(f"Converged in {k} iterations.")
            break
        p = -H.dot(g)
        # Line search parameters
        alpha = 1
        c = 1e-4
        rho = 0.9
        # Backtracking line search
        while func(x + alpha * p, n_beads) > func(x, n_beads) + c * alpha * g.dot(p):
            alpha *= rho
        x_new = x + alpha * p
        g_new = grad_func(x_new, n_beads)
        s = x_new - x
        y = g_new - g
        ys = y.dot(s)
        if ys > 1e-10:  # Avoid division by zero
            """
            sos = np.outer(s,s)
            yos = np.outer(y,s)
            soy = np.outer(s,y)
            H = H + (ys + y.dot(H.dot(y)))*(sos)/(ys**2) - (H.dot(yos) + soy.dot(H))/ys
            """
            I = np.eye(n)
            rho_k = 1.0 / ys
            H = (I - rho_k * np.outer(s, y)).dot(H).dot(I - rho_k * np.outer(y, s)) + rho_k * np.outer(s, s)
            
        else:
            H = I  # Reset if ys is too small
        x = x_new
        g = g_new
        trajectory.append(x.copy())  # Append current step to trajectory
    else:
        print(f"Maximum iterations ({max_iter}) reached.")
    f_val = func(x, n_beads)
    return x, f_val, trajectory

"""
if __name__ == "__main__":
    # Initial guess
    x0 = np.array([-1.2, 1.0])

    # Run the BFGS algorithm
    
    _min, f_min, trajectory = bfgs(x0, rosenbrock, rosenbrock_grad)
    print(_min)
    """