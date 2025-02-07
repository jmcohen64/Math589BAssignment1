import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./energy.so')  # Use 'energy.dll' on Windows

# Define function argument and return types
lib.total_energy.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # positions
    ctypes.c_int,                    # n_beads
    ctypes.c_double,                 # epsilon
    ctypes.c_double,                 # sigma
    ctypes.c_double,                 # b
    ctypes.c_double                  # k_b
]
lib.total_energy.restype = ctypes.c_double

lib.compute_gradient.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # positions
    ctypes.POINTER(ctypes.c_double),  # gradients (output)
    ctypes.c_int,                     # n_beads
    ctypes.c_double,                  # epsilon
    ctypes.c_double,                  # sigma
    ctypes.c_double,                  # b
    ctypes.c_double                   # k_b
]

def compute_total_energy(positions, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Wrapper function to compute total energy using the C++ library.
    """
    n_beads = len(positions) // 3
    positions_array = np.array(positions, dtype=np.float64)
    positions_ptr = positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    return lib.total_energy(positions_ptr, n_beads, epsilon, sigma, b, k_b)

def compute_gradient(positions, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Wrapper function to compute the gradient (force) using the C++ library.
    """
    n_beads = len(positions) // 3
    positions_array = np.array(positions, dtype=np.float64)
    
    # Create an empty array for the gradients, same size as positions
    gradients_array = np.zeros_like(positions_array, dtype=np.float64)

    # Get C-style pointers
    positions_ptr = positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    gradients_ptr = gradients_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Call the C++ function (modifies gradients_array in-place)
    lib.compute_gradient(positions_ptr, gradients_ptr, n_beads, epsilon, sigma, b, k_b)

    return gradients_array.flatten()  # Return the computed gradients

# Example usage
if __name__ == "__main__":
    n_beads = 10
    positions = np.random.rand(n_beads * 3)  # Random 3D positions for each bead
    energy = compute_total_energy(positions)
    print(f"Total Energy: {energy}")

    # Compute gradients
    gradients = compute_gradient(positions)
    print("Gradients:")
    print(gradients)