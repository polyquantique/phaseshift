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
# LPLM Interferometer Module

This module provides functions to perform the López Pastor, Lundeen and Marquardt (LPLM) decomposition of a unitary matrix into a sequence of
6N+1 phase masks interleaved with discrete Fourier transforms (DFT) matrices.

The module consists of the following main functions:
- `lplm_decomposition`: Decomposes a unitary matrix using the LPLM decomposition.
- `circuit_reconstruction`: Reconstructs the unitary matrix from the decomposition

For more details on the LPLM decomposition, see references [1] and [2].

### References
1. López Pastor, Víctor, Jeff Lundeen, and Florian Marquardt. "Arbitrary optical wave evolution with Fourier transforms and phase masks." Optics Express 29.23 (2021): 38441-38450.
2. Clements, William R., et al. "Optimal design for universal multiport interferometers." Optica 3.12 (2016): 1460-1465.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from . import matrix_operations as mo
from .clements_interferometer import MachZehnder, mzi_decomposition

# Constants
SQRT2 = np.sqrt(2)


def a_mask_sequence(
    dim: int, theta: NDArray[np.floating], phi: NDArray[np.floating]
) -> list[NDArray[np.complex128]]:
    """Generates a sequence of phase masks for the even layers in the LPLM decomposition.

    This function creates a sequence of diagonal matrices that represent the masks used in the LPLM decomposition for even MZI layers.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).
        theta (NDArray[float]): Array of all angles theta for the MZI layer.
        phi (NDArray[float]): Array of all angles phi for the MZI layer.

    Returns:
        list[NDArray[complex]]: List of phase masks.
    """
    mask_sequence = [
        mo.e_matrix(dim),
        mo.g_matrix(dim),
        mo.h_matrix(dim).conj(),
        mo.pi_permutation_matrix(dim) @ (mo.phase_matrix(dim, 2 * theta) * mo.g_matrix(dim)),
        mo.e_matrix(dim),
        mo.phase_matrix(dim, phi) * mo.z_matrix(dim),
    ]

    return mask_sequence


def b_mask_sequence(
    dim: int, theta: NDArray[np.floating], phi: NDArray[np.floating]
) -> list[NDArray[np.complex128]]:
    """Generates a sequence of phase masks for the odd layers in the LPLM decomposition.

    This function creates a sequence of diagonal matrices that represent the masks used in the LPLM decomposition for odd MZI layers.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).
        theta (NDArray[float]): Array of all angles theta for the MZI layer.
        phi (NDArray[float]): Array of all angles phi for the MZI layer.

    Returns:
        list[NDArray[complex]]: List of phase masks.
    """
    mask_sequence = [
        mo.e_matrix(dim),
        mo.pi_permutation_matrix(dim) @ mo.g_matrix(dim),
        mo.h_matrix(dim).conj() * mo.v_matrix(dim),
        mo.phase_matrix_swap(dim, 2 * theta) * mo.g_matrix(dim),
        mo.e_matrix(dim),
        mo.pi_permutation_matrix(dim) @ mo.phase_matrix_swap(dim, phi),
    ]

    return mask_sequence


def extract_layer_parameters(dim: int, circuit: list[MachZehnder]) -> NDArray[np.floating]:
    """Extracts the layer parameters from a Clements decomposition.

    Given a Clements decomposition, this function extracts the parameters `theta` and `phi` for each layers in the form of a 3D array.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).
        circuit (list[MachZehnder]): List of Mach-Zehnder interferometers representing the Clements decomposition.

    Returns:
        NDArray[float]: (layer, unit_cell, parameter) - A 3D array of shape (dim, dim//2, 2) containing the parameters `theta` and `phi` for each layer.
    """
    # Initialize a 3D array to hold the layer parameters
    layers = np.zeros((dim, dim // 2, 2), dtype=np.float64)

    # List to keep track of filled layers
    fill_history: list[tuple] = []

    # Iterate through the circuit in reverse order to fill the layers
    for mzi in circuit[::-1]:

        if mzi.target[0] % 2 == 0:  # Even layer
            target_count = fill_history.count(mzi.target)  # Check in which layer the MZI is located
            fill_history.append(mzi.target)  # Append the target to the history
            layers[2 * target_count, mzi.target[0] // 2] = [
                mzi.theta,
                mzi.phi,
            ]  # Fill the layer with the parameters

        else:  # Odd layer
            target_count = fill_history.count(mzi.target)
            fill_history.append(mzi.target)
            layers[2 * target_count + 1, (mzi.target[0] - 1) // 2] = [mzi.theta, mzi.phi]

    return layers


class LplmDecomp(NamedTuple):
    """A named tuple to store the results of the LPLM decomposition.

    This tuple contains the diagonal phase matrix `D` and the sequence of phase masks used in the decomposition.
    The unitary can be reconstructed as U = D @ matrix_interleave(mask_sequence), where `matrix_interleave` place a DFT matrix `F` between each phase masks,
    i.e. F @ M1 @ F @ M2 @ ... @ Mn.

    Attributes:
        D (NDArray[complex]): Diagonal phase matrix at the output of the interferometer.
        mask_sequence (list[NDArray[complex]]): Sequence of phase masks used in the decomposition.
    """

    D: NDArray[np.complex128]
    """Diagonal phase matrix at the output of the interferometer."""

    mask_sequence: list[NDArray[np.complex128]]
    """Sequence of phase masks used in the decomposition."""


def lplm_decomposition(U: NDArray[np.complex128]) -> LplmDecomp:
    """Performs the LPLM decomposition of a unitary matrix U.

    The LPLM decomposition is a method to decompose a unitary matrix into a sequence of phase masks
    interleaved with discrete Fourier transforms (DFT) matrices. The algorithm generates a sequence of 6N + 1 phase masks
    for a unitary matrix of dimension N. The function returns a `LplmDecomp` object containing the diagonal phase matrix
    `D` and the sequence of phase masks used in the decomposition such that the unitary can be reconstructed as

        U = `D` @ `matrix_interleave(mask_sequence)`.

    For more details on the LPLM decomposition, see references [1] and [2].

    Args:
        U (NDArray[complex]): Unitary matrix to be decomposed. U must have even dimension.

    Returns:
        LplmDecomp: A named tuple containing the diagonal phase matrix `D` and the sequence of phase masks used in the decomposition.
            - `D` is a diagonal matrix that represents the phase shifts applied to the output channels.
            - `mask_sequence` is a list of phase masks that must be interleaved with DFTs.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input matrix is not unitary or if its dimension is not even.

    ### References
        [1]: Víctor López Pastor, Jeff Lundeen, and Florian Marquardt, "Arbitrary optical wave evolution with Fourier transforms and phase masks," Opt. Express 29, 38441-38450 (2021)
        [2]: William R. Clements, Peter C. Humphreys, Benjamin J. Metcalf, W. Steven Kolthammer, and Ian A. Walmsley, "Optimal design for universal multiport interferometers," Optica 3, 1460-1465 (2016)
    """

    # Check if the input is a numpy array
    if not isinstance(U, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    dim = U.shape[0]
    # Check if the input matrix has even dimension
    if dim % 2 != 0:
        raise ValueError("Dimension must be even for LPLM decomposition.")

    # Check if the input matrix is unitary
    if not np.allclose(U @ U.T.conj(), np.eye(dim)):
        raise ValueError("Input matrix must be unitary.")

    # Step 1: Apply the channels permutation
    U = mo.channels_permutation(dim).T @ U @ mo.channels_permutation(dim)

    # Step 2: Decompose the permuted matrix using Clements decomposition
    clements_decomp = mzi_decomposition(U)

    # Step 3: Extract the layer parameters
    parameters = extract_layer_parameters(dim, clements_decomp.circuit)

    # Step 4: Create the mask sequence
    mask_sequence: list[NDArray[np.complex128]] = []
    for i in range(dim):
        if i % 2 == 0:
            mask_sequence[:0] = a_mask_sequence(dim, parameters[i, :, 0], parameters[i, :, 1])
        else:
            mask_sequence[:0] = b_mask_sequence(dim, parameters[i, :, 0], parameters[i, :, 1])

    D = (mo.channels_permutation(dim) @ clements_decomp.D) * mo.g_matrix(dim)
    mask_sequence[-1] = mask_sequence[-1] * mo.g_matrix(dim).conj()

    # Step 5: Return the decomposition
    return LplmDecomp(D, mask_sequence)


def circuit_reconstruction(decomp: LplmDecomp) -> NDArray[np.complex128]:
    """Reconstructs the unitary matrix from it's LPLM decomposition.

    Given a `LplmDecomp` object, this function reconstructs the unitary matrix by applying the diagonal
    phase matrix `D` and interleaving the phase masks with DFT matrices. `decomp` can be obtained from the `lplm_decomposition` function.

    Args:
        decomp (LplmDecomp): A named tuple containing the diagonal phase matrix `D` and the sequence of phase masks used in the decomposition.

    Returns:
        NDArray[complex]: Reconstructed unitary matrix.

    Raises:
        TypeError: If the input is not a `LplmDecomp` object.
    """
    if not isinstance(decomp, LplmDecomp):
        raise TypeError("Input must be a LplmDecomp object.")

    return decomp.D[:, None] * mo.matrix_interleave(decomp.mask_sequence)
