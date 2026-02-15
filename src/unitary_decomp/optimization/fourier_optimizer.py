# Copyright 2025 Vincent Girouard
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
# Fourier Optimizer

This module provides functions to approximate a target unitary matrix as a sequence of phase masks and DFT matrices.
The function `mask_optimizer` solves a global optimization problem to minimize
the infidelity between the target matrix and the approximation. The optimization is performed using the `basinhopping` algorithm
from `scipy.optimize`, which combines a global search with a local optimizer. The module contains the following key functions:

- `fidelity`: Computes the fidelity between two unitary matrices.
- `objective_function`: Objective function to minimize during optimization, which corresponds to the infidelity
    between the target matrix and the matrix constructed from the phase masks.
- `mask_optimizer`: Optimizes the phase masks to approximate a target unitary matrix.

### References
1. Saygin, M. Yu, et al. "Robust architecture for programmable universal unitaries." Physical review letters 124.1 (2020): 010501.
2. Pereira, Luciano, et al. "Minimum optical depth multiport interferometers for approximating arbitrary unitary operations and pure states." Physical Review A 111.6 (2025): 062603.

**Note**: The method used in this module is adapted from Pereira *et al.* and can be found at (https://github.com/akiiop/PUMA/tree/main).
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import basinhopping

from unitary_decomp import matrix_operations as mo
from unitary_decomp.fourier_interferometer import FourierDecomp


def fidelity(U: NDArray, V: NDArray) -> float:
    """Calculate the fidelity between two unitary matrices.

    Args:
        U (np.ndarray): First unitary matrix.
        V (np.ndarray): Second unitary matrix.

    Returns:
        float: Fidelity between the two matrices, between 0 and 1.

    Raises:
        ValueError: If the shapes of U and V do not match.
    """
    if U.shape != V.shape:
        raise ValueError("Both matrices must have the same shape.")

    # Dimension of the unitary matrices
    dim = U.shape[0]

    # Calculate the fidelity between the two unitary matrices
    return abs(np.trace(U.conj().T @ V) / dim) ** 2


def objective_function(
    phases: NDArray[np.floating],
    U: NDArray[np.number],
    length: int,
    mask_shape: NDArray[np.bool],
    mixing_layer: NDArray[np.number] | None = None,
) -> float:
    """Function to minimize during optimization.

    This function returns the infidelity between the target unitary matrix `U` and the matrix constructed from the phases and mixing layer provided.
    If no mixing layer is given, the DFT matrix is used.

    Args:
        phases (NDArray[float]): 1D array of phases to be optimized.
        U (NDArray[complex]): Target unitary matrix.
        length (int): Number of phase masks in the sequence.
        mask_shape (NDArray[bool]): Boolean array of size `U.shape[0]` indicating where phase shifters are located in the masks.
            `True` indicates the presence of a phase shifter.
        mixing_layer (NDArray[float], optional): Mixing layer to be used between phase masks. If None, the DFT matrix is used.

    Returns:
        float: Infidelity between the target unitary matrix `U` and the unitary matrix constructed from the phases and mixing layer.
    """
    # Reshape the phase array into a 2D array where each row corresponds to a mask
    phase_matrix = np.reshape(phases, (length, -1))

    # Insert a phase of zero where the mask shape indicates no phase shifter
    for index in np.where(mask_shape == False)[0]:
        phase_matrix = np.hstack(
            (phase_matrix[:, :index], np.zeros((length, 1)), phase_matrix[:, index:])
        )

    # Convert phases to complex exponentials
    masks = np.exp(1.0j * phase_matrix)

    # Construct the unitary matrix from the masks using DFTs
    U_constructed = masks[0][:, None] * mo.matrix_interleave(masks[1:], mixing_layer=mixing_layer)

    # Calculate and return the infidelity
    return 1 - fidelity(U, U_constructed)


def mask_optimizer(
    U: NDArray[np.number],
    length: int,
    mixing_layer: Optional[NDArray[np.number]] = None,
    mask_shape: Optional[NDArray[np.bool]] = None,
    n_iters: int = 500,
    n_runs: int = 1,
    display: bool = True,
) -> tuple[FourierDecomp, float]:
    """Optimize the mask sequence to approximate a target unitary matrix using phase masks interleaved with a mixing layer.

    Given a target unitary matrix `U` and a mixing layer, this function will optimize the phase masks to minimize the infidelity between `U` and the matrix constructed from the masks.
    The optimization is performed using a basinhopping algorithm along with a Quasi-Newton local optimizer.
    If no mixing layer is provided, the DFT matrix is used by default. The function returns a `FourierDecomp` object containing the optimized mask sequence.

    Args:
        U (NDArray[number]): Target unitary matrix to approximate.
        length (int): Number of phase masks in the sequence.
        mixing_layer (NDArray[number], optional): Mixing layer to be used between phase masks. If None, the DFT matrix is used.
        mask_shape (NDArray[bool], optional): Boolean array of size `U.shape[0]` indicating on which modes phase shifters are located in the masks,
            where `True` indicates the presence of a phase shifter. If None, all modes will have a phase shifter except for the first one.
        n_iters (int, optional): Number of iterations during optimization. Default is 500.
        n_runs (int, optional): Number of runs performed by the optimizer with different initial conditions. Default is 1.
        display (bool, optional): If True, the progress will be displayed in the console. Default is True.

    Returns:
        tuple[FourierDecomp, float]: A tuple containing the optimized mask sequence, given as a `FourierDecomp` object, and the infidelity of the decomposition.

    Raises:
        TypeError: If `U` is not a numpy array or if `mask_shape` is not a numpy array or None or if `mixing_layer` is not a numpy array or None.
        ValueError: If `U` is not a unitary matrix.

    Example:
        >>> import numpy as np
        >>> from scipy.stats import unitary_group
        >>> from unitary_decomp.fourier_interferometer import circuit_reconstruction
        >>> from unitary_decomp.optimization import mask_optimizer

        >>> U = unitary_group(dim=6, seed=137).rvs()  # Generate a random 6x6 unitary matrix
        >>> mask_shape = np.array([True, True, True, True, True, False])  # Define the shape of the phase masks with no phase shifter on the last mode

        >>> # Optimize the mask sequence with N+1 = 7 phase masks (complete parametrization)
        >>> decomposition, infidelity = mask_optimizer(U, length=7, mask_shape=mask_shape, n_iters=300, n_runs=3)
        >>> U_reconstructed = circuit_reconstruction(decomposition)
        >>> print(np.allclose(U, U_reconstructed, rtol=1e-2))
        True

        >>> # Optimize the mask sequence with N = 6 phase masks (under-parametrization)
        >>> decomposition, infidelity = mask_optimizer(U, length=6, n_iters=300, n_runs=3)
        >>> U_reconstructed = circuit_reconstruction(decomposition)
        >>> print(np.allclose(U, U_reconstructed, rtol=1e-2))
        False
    """

    # Check if U is a numpy array
    if not isinstance(U, np.ndarray):
        raise TypeError("U must be a numpy array.")

    # Check if U is a unitary matrix
    if not (np.allclose(U @ U.T.conj(), np.eye(U.shape[0])) and U.shape[0] == U.shape[1]):
        raise ValueError("U must be unitary.")

    # Check if the mixing layer is a numpy array or None
    if not isinstance(mixing_layer, (np.ndarray, type(None))):
        raise TypeError("The mixing layer must be a numpy array or None.")

    # Check if mask_shape is a numpy array or None
    if not isinstance(mask_shape, (np.ndarray, type(None))):
        raise TypeError("mask_shape must be a numpy array or None.")

    # Dimension of the unitary matrix
    dim = U.shape[0]

    # If `mask_shape` is not provided, create a default one with the first mode having no phase shifter
    if mask_shape is None:
        mask_shape = np.array([False] + [True] * (dim - 1))

    # Parameters for the local optimizer
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "args": (U, length, mask_shape, mixing_layer),
    }

    # Run the global optimizer `n_runs` times with different initial conditions
    infidelity = 1
    for _ in range(n_runs):
        # Generate random initial phases
        initial_phases = np.random.uniform(
            0, 2 * np.pi, size=(length * np.count_nonzero(mask_shape),)
        )
        result = basinhopping(
            func=objective_function,
            x0=initial_phases,
            minimizer_kwargs=minimizer_kwargs,
            niter=n_iters,
            niter_success=150,
            T=1.5,
            stepsize=1.5,
            disp=display,
        )

        # Keep the results if the infidelity is lower than the previous best
        if result.fun < infidelity:
            infidelity = result.fun
            phases = result.x

    # Reshape the phase array into a 2D array where each row corresponds to a mask
    phase_matrix = np.reshape(phases, (length, -1)) % (2 * np.pi)
    for index in np.where(mask_shape == False)[0]:
        phase_matrix = np.hstack(
            (phase_matrix[:, :index], np.zeros((length, 1)), phase_matrix[:, index:])
        )

    # Convert phases to complex exponentials
    mask_sequence = np.exp(1.0j * phase_matrix)

    # Adjust the global phase to match the target matrix
    U_reconstructed = mask_sequence[0][:, None] * mo.matrix_interleave(
        mask_sequence[1:], mixing_layer=mixing_layer
    )
    phase_difference = np.mean(np.angle(U / U_reconstructed))
    mask_sequence[0] = mask_sequence[0] * np.exp(1.0j * phase_difference)

    return FourierDecomp(mask_sequence[0], mask_sequence[1:]), infidelity
