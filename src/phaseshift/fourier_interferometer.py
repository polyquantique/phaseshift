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
# Fourier Interferometer Module

This module implements the Fourier decomposition and the compact Fourier decomposition of a unitary matrix,
allowing it to be expressed exactly as a product of 4N+1 or 2N+5 phase masks interleaved with discrete Fourier transform
(DFT) matrices.

The module consists of the following main functions:
    - `fourier_decomposition`: Decompose a unitary matrix exactly into a sequence of 4N + 1 phase masks interleaved with DFT matrices.
    - `compact_fourier_decomposition`: Decompose a unitary matrix exactly into a sequence of 2N + 5 phase masks interleaved with DFT matrices.
    - `circuit_reconstruction`: Reconstructs a unitary matrix from its Fourier decomposition.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from . import matrix_operations as mo
from .bell_interferometer import BellDecomp, bell_decomposition


class FourierDecomp(NamedTuple):
    """A NamedTuple to store the Fourier decomposition of a unitary matrix.

    Attributes:
        D (NDArray[complex]): Array of single-mode phase shifts at the output of the circuit.
        mask_sequence (list[NDArray[complex]]): List of phase masks in the decomposition.
    """

    D: NDArray[np.complex128]
    """Array of single-mode phase shifts at the output of the circuit."""

    mask_sequence: list[NDArray[np.complex128]]
    """List of phase masks in the decomposition."""


def angle_conversion(delta: float, sigma: float) -> tuple[float, float]:
    """Convert the parameters delta and sigma to theta1 and theta2.

    `delta` and `sigma` are related to the angles `theta_1` and `theta_2` of the symmetric
    Mach-Zehnder interferometer by:

    `sigma` = (`theta_1` + `theta_2`) / 2 and `delta` = (`theta_1` - `theta_2`) / 2.

    Args:
        delta (float): Parameter delta of a symmetric Mach-Zehnder interferometer.
        sigma (float): Parameter sigma of a symmetric Mach-Zehnder interferometer.

    Returns:
        tuple[float, float]: A tuple containing `theta_1` and `theta_2`.
    """
    theta_1 = delta + sigma
    theta_2 = sigma - delta

    return theta_1, theta_2


def distribute_edge_phases(bell_decomposition: BellDecomp) -> BellDecomp:
    """Distributes the edge phases of a Bell decomposition across all modes.

    This function takes a `BellDecomp` instance and distributes the edge phases uniformly across all modes, which
    increases the symmetry of the decomposition and allows for a more compact Fourier decomposition.
    The function adjusts the parameters `sigma` of the sMZIs in the Bell decomposition to account for the
    distributed edge phases.

    Args:
        bell_decomposition (BellDecomp): A Bell decomposition object. Can be obtained from the `bell_decomposition` function in the `bell_interferometer` module.

    Returns:
        BellDecomp: A new `BellDecomp` instance with the edge phases distributed across all modes.
    """
    # Dimension of the system (Number of modes)
    dim = bell_decomposition.dim
    new_phi_edge = {}

    # Iterate over the edge phase shifters in the Bell decomposition.
    for pos, phi in bell_decomposition.phi_edge.items():
        partial_phi = phi / dim
        layer = pos[1]

        # Add the scaled partial phase to the left sMZIs.
        for mode in range(0, dim - 1, 2):
            bell_decomposition.sigma[mode, layer - 1] += (mode + 1) * partial_phi

        # Remove the scaled partial phase from the right sMZIs.
        for mode in range(1, dim - 1, 2):
            bell_decomposition.sigma[mode, layer] -= (mode + 1) * partial_phi

        # Distribute the edge phase across all modes with varying sign.
        for mode in range(dim):
            new_phi_edge[mode, layer] = (-1) ** (mode % 2 + 1) * partial_phi

    # Update the Bell decomposition with the new edge phases.
    bell_decomposition.phi_edge = new_phi_edge

    return bell_decomposition


def extract_bell_parameters(
    bell_decomposition: BellDecomp, distributed_edges: bool = False
) -> NDArray[np.complex128]:
    """Extracts the parameters from a Bell decomposition.

    This function takes a `BellDecomp` object and extracts the parameters needed to
    construct the parametric phase masks in the Fourier interferometers. These phase masks
    are made from the angles `theta_1` and `theta_2` of the sMZIs and from the edge phases in the Bell decomposition.
    The function outputs a 3D array of complex numbers, where the first dimension corresponds
    to the bi-layer index, the second dimension corresponds to the position in the bi-layer (0 for edge layers,
    1 for even MZI layers, and 2 for odd MZI layers), and the third dimension corresponds to the modes.

    Args:
        bell_decomposition (BellDecomp): A Bell decomposition object. Can be obtained from the `bell_decomposition` function in the `bell_interferometer` module.
        distributed_edges (bool): Set to true if the edge phases have been distributed to all modes. Defaults is False.

    Returns:
        NDArray[complex]: A 3D array of complex numbers representing the parametric phase masks in the Fourier decomposition.

    """
    # Dimension of the system (Number of modes)
    dim = bell_decomposition.dim

    # Initialize the parameters array
    parameters = np.ones((dim // 2, 3, dim), np.complex128)

    # Iterate over the sMZI in the Bell decomposition and fill the parameters array.
    for mode, layer in bell_decomposition.delta.keys():

        # Convert delta and sigma to theta_1 (first mode of the MZI) and theta_2 (second mode of the MZI)
        sigma = bell_decomposition.sigma[mode, layer]
        delta = bell_decomposition.delta[mode, layer]
        theta_1, theta_2 = angle_conversion(delta, sigma)

        # Even layers: Interactions between j and j + dim/2
        if layer % 2 == 0:
            parameters[layer // 2, 1, mode // 2] = np.exp(1.0j * theta_1)
            parameters[layer // 2, 1, (mode + dim) // 2] = np.exp(1.0j * theta_2)

        # Odd layers: Interactions between j and j - dim/2 + 1 (mod dim/2)
        else:
            parameters[layer // 2, 2, (mode - 1) // 2] = np.exp(1.0j * theta_1)
            parameters[layer // 2, 2, (mode - 1 + dim) // 2] = np.exp(1.0j * theta_2)

    # Iterate over the edge phase shifters in the Bell decomposition and fill the parameters array.
    for (mode, layer), phi in bell_decomposition.phi_edge.items():

        # If the edge phases are distributed, set the phase of all modes in the layer.
        if distributed_edges:
            parameters[layer // 2, 0, mode] = np.exp(1.0j * phi)

        else:
            parameters[layer // 2 + 1, 0, mode] = np.exp(1.0j * phi)

    return parameters


def layer_mask_sequence(layer_parameters: NDArray[np.complex128]) -> list[NDArray[np.complex128]]:
    """Constructs the mask sequence for a set of bi-layer parameters.

    This function takes the three parametric phase masks of a bi-layer and returns the complete
    sequence of masks that implements the bi-layer as a product of diagonal and DFT matrices.

    Args:
        layer_parameters (NDArray[complex]): A 2D array of complex numbers that contains the 3 phase shift masks of a bi-layer. (edge, even, odd)

    Returns:
        list[NDArray[complex]]: A list of arrays with the parameters of each masks in the bi-layer.
    """

    # Extract parameters of the bi-layer
    dim = layer_parameters.shape[1]
    phase_edge, phase_even, phase_odd = layer_parameters

    # Construct the mask sequence for the bi-layer
    mask_sequence = [
        mo.e_matrix(dim),
        mo.pi_permutation_matrix(dim) @ mo.g_matrix(dim),
        mo.h_matrix(dim).conj(),
        phase_odd,
        mo.h_matrix(dim).conj(),
        mo.pi_permutation_matrix(dim) @ (phase_even * mo.g_matrix(dim)),
        mo.e_matrix(dim),
        phase_edge * mo.z_matrix(dim),
    ]

    return mask_sequence


def compact_mask_sequence(layer_parameters: NDArray[np.complex128]) -> list[NDArray[np.complex128]]:
    """Constructs the mask sequence for a set of bi-layer parameters in a compact form.

    This function takes the three parametric phase masks of a bi-layer and returns the complete
    sequence of masks that implements the bi-layer as a compact product of diagonal and DFT matrices.

    Args:
        layer_parameters (NDArray[complex]): A 2D array of complex numbers that contains the 3 phase shift masks of a bi-layer. (edge, even, odd)

    Returns:
        list[NDArray[complex]]: A list of arrays with the parameters of each masks in the bi-layer.
    """
    # Extract parameters of the bi-layer
    dim = layer_parameters.shape[1]
    phase_edge, phase_even, phase_odd = layer_parameters

    # Construct the mask sequence for the bi-layer
    mask_sequence = [
        mo.h_matrix(dim),
        mo.pi_permutation_matrix(dim) @ phase_odd,
        mo.h_matrix(dim) * phase_edge,
        phase_even,
    ]

    return mask_sequence


def fourier_decomposition(U: NDArray[np.complex128]) -> FourierDecomp:
    """Performs the Fourier decomposition of a unitary matrix.

    This function takes a N x N unitary matrix `U` and performs its Fourier decomposition.
    This decomposition allows to express `U` as a product of 4N + 1 phase masks interleaved
    with discrete Fourier transform (DFT) matrices. The matrix can be reconstructed using
    the `circuit_reconstruction` function.

    Args:
        U (NDArray[complex]): The unitary matrix to be decomposed.

    Returns:
        FourierDecomp: A named tuple containing the following attributes

        * `D`: A 1D array containing the single-mode phase shifts at the output of the circuit.
        * `mask_sequence`: The list of phase masks used in the decomposition.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input matrix is not a unitary matrix of even dimension.
    """

    # Check if the input is a numpy array.
    if not isinstance(U, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    # Dimension of the system (Number of modes)
    dim = U.shape[0]

    # Check if the input is a square matrix of even dimension.
    if (U.shape[0] != U.shape[1]) or (dim % 2 != 0):
        raise ValueError("Input must be a square matrix of even dimension.")

    # Check if the input is a unitary matrix.
    if not np.allclose(U @ U.T.conj(), np.eye(dim)):
        raise ValueError("Input must be a unitary matrix.")

    # Perform the Bell decomposition of the unitary matrix U.
    bell_decomp = bell_decomposition(
        mo.channels_permutation(dim).T @ U @ mo.channels_permutation(dim)
    )

    # Extract the layer parameters from the Bell decomposition and generate the mask sequence.
    mask_sequence: list[NDArray] = []
    params = extract_bell_parameters(bell_decomp)
    for layer in params:
        mask_sequence[:0] = layer_mask_sequence(layer)

    # Construct the diagonal matrix D at the output of the circuit.
    D = np.ones(dim, np.complex128)
    for mode, phi in bell_decomp.phi_output.items():
        D[mode] = np.exp(1.0j * phi)
    D = (mo.channels_permutation(dim) @ D) * mo.g_matrix(dim)

    # Construct the diagonal phase matrix P at the input of the circuit.
    P = np.ones(dim, np.complex128)
    for mode, phi in bell_decomp.phi_input.items():
        P[mode] = np.exp(1.0j * phi)

    # Adjust the last phase mask of the sequence.
    mask_sequence[-1] = (
        mask_sequence[-1] * mo.g_matrix(dim).conj() * (mo.channels_permutation(dim) @ P)
    )

    return FourierDecomp(D, mask_sequence)


def compact_fourier_decomposition(U: NDArray[np.complex128]) -> FourierDecomp:
    """Compact  Fourier decomposition of a unitary matrix.

    This function takes a N x N unitary matrix `U` and performs its Fourier decomposition.
    The decomposition results in a more compact form than the `fourier_decomposition` function,
    allowing to express `U` as a product of 2N + 5 phase masks interleaved with discrete Fourier transform (DFT) matrices.
    The matrix can be reconstructed using the `circuit_reconstruction` function.

    Args:
        U (NDArray[complex]): The unitary matrix to be decomposed.

    Returns:
        FourierDecomp: A named tuple containing the following attributes

        * `D`: A 1D array containing the single-mode phase shifts at the output of the circuit.
        * `mask_sequence`: The list of phase masks used in the decomposition.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input matrix is not a unitary matrix of even dimension.
    """

    # Check if the input is a numpy array.
    if not isinstance(U, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    # Dimension of the system (Number of modes)
    dim = U.shape[0]

    # Check if the input is a square matrix of even dimension.
    if (U.shape[0] != U.shape[1]) or (dim % 2 != 0):
        raise ValueError("Input must be a square matrix of even dimension.")

    # Check if the input is a unitary matrix.
    if not np.allclose(U @ U.T.conj(), np.eye(dim)):
        raise ValueError("Input must be a unitary matrix.")

    # Perform the Bell decomposition of the unitary matrix U.
    bell_decomp = bell_decomposition(
        mo.channels_permutation(dim).T @ mo.pi_transformation(U) @ mo.channels_permutation(dim)
    )

    # Distribute the edge phases across all modes.
    bell_decomp = distribute_edge_phases(bell_decomp)

    # Extract the layer parameters from the Bell decomposition and generate the mask sequence.
    mask_sequence: list[NDArray] = []
    params = extract_bell_parameters(bell_decomp, distributed_edges=True)
    for layer in params:
        mask_sequence[:0] = compact_mask_sequence(layer)

    # Construct the diagonal matrix D at the output of the circuit.
    D = np.ones(dim, np.complex128)
    for mode, phi in bell_decomp.phi_output.items():
        D[mode] = np.exp(1.0j * phi)
    D = mo.pi_permutation_matrix(dim) @ ((mo.channels_permutation(dim) @ D) * mo.g_matrix(dim))

    # Construct the diagonal phase matrix P at the input of the circuit.
    P = np.ones(dim, np.complex128)
    for mode, phi in bell_decomp.phi_input.items():
        P[mode] = np.exp(1.0j * phi)

    # Adjust the phase masks at the edge of the sequence.
    mask_sequence[:0] = [mo.e_matrix(dim), mo.g_matrix(dim)]
    mask_sequence[-1] = mask_sequence[-1] * mo.g_matrix(dim)
    mask_sequence += [
        mo.e_matrix(dim),
        mo.pi_permutation_matrix(dim) @ ((mo.channels_permutation(dim) @ P) * mo.g_matrix(dim)),
    ]

    return FourierDecomp(D, mask_sequence)


def circuit_reconstruction(decomposition: FourierDecomp) -> NDArray[np.complex128]:
    """Reconstructs a unitary matrix from it's Fourier decomposition.

    Given a `FourierDecomp` object, this function reconstructs the unitary matrix by applying the diagonal
    phase matrix `D` and interleaving the phase masks with DFT matrices. `decomposition` can be obtained
    from the `fourier_decomposition` or `compact_fourier_decomposition` functions.

    Args:
        decomposition (FourierDecomp): A named tuple containing the diagonal phase matrix `D` and the sequence of phase masks used in the decomposition.

    Returns:
        NDArray[complex]: Reconstructed unitary matrix.

    Raises:
        TypeError: If the input is not a `FourierDecomp` object.
    """
    if not isinstance(decomposition, FourierDecomp):
        raise TypeError("Input must be a FourierDecomp object.")

    return decomposition.D[:, None] * mo.matrix_interleave(decomposition.mask_sequence)
