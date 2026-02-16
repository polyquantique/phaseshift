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
# Matrix operations

This module provides various matrix definitions and transformations used in the decomposition of unitary matrices
using Fourier transforms and phase masks.

### References
1. López Pastor, Víctor, Jeff Lundeen, and Florian Marquardt. "Arbitrary optical wave evolution with Fourier transforms and phase masks." Optics Express 29.23 (2021): 38441-38450.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import circulant, dft

# Constants
SQRT2 = np.sqrt(2)


def g_matrix(dim: int) -> NDArray[np.complex128]:
    """Generates the G diagonal matrix for a given dimension.

    G is used in the factorization of the X matrix: X = G Y G.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[complex]: G matrix with the specified dimension.
    """

    return np.concatenate((np.ones(dim // 2), 1.0j * np.ones(dim // 2)))


def g_matrix_swap(dim: int) -> NDArray[np.complex128]:
    """Generates the G' diagonal matrix for a given dimension.

    G' is obtained by swapping the first and second half of the channels in the G matrix.
    G' is used in the factorization of the X' matrix: X' = G' Y G'.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[complex]: G matrix with the specified dimension, with swapped blocks.
    """
    return np.concatenate((1.0j * np.ones(dim // 2), np.ones(dim // 2)))


def z_matrix(dim: int) -> NDArray[np.integer]:
    """Generates the Z matrix for a given dimension.

    Z is a diagonal matrix with ones in the first half and negative ones in the second half.
    It can be expressed as Z = G @ G.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[int]: Z matrix with the specified dimension.
    """
    return np.concatenate((np.ones(dim // 2), -np.ones(dim // 2)))


def z_matrix_swap(dim: int) -> NDArray[np.integer]:
    """Generates the Z' matrix for a given dimension.

    Z' is obtained by swapping the first and second half of the channels in the Z matrix.
    Z' can be expressed as Z' = G' @ G'.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[int]: Z matrix swapped with the specified dimension.
    """
    return np.concatenate((-np.ones(dim // 2), np.ones(dim // 2)))


def e_matrix(dim: int) -> NDArray[np.complex128]:
    """E diagonal matrix for a given dimension.

    E is used in the diagonalization of the Y matrix: Y = F.T.conj() @ E @ F.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[complex]: E matrix with the specified dimension.
    """
    return 1 / SQRT2 * np.array([(1 - 1.0j * (-1) ** i) for i in range(dim)])


def h_matrix(dim: int) -> NDArray[np.complex128]:
    """H diagonal matrix for a given dimension.

    H is used in the diagonalization of the CI matrix: CI = F.T.conj() @ H @ F.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[complex]: H matrix with the specified dimension.
    """
    diag = np.zeros(dim, dtype=np.complex128)

    for i in range(dim):
        diag[i] = 1 / 2 * (1 - (-1) ** i) + 1 / 2 * (1 + (-1) ** i) * np.exp(
            1.0j * 2 * np.pi * i / dim
        )
    return diag


def v_matrix(dim: int) -> NDArray[np.integer]:
    """V diagonal matrix for a given dimension.

    V is used in the diagonalization of the Swap matrix: S = F.T.conj() @ V @ F.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[int]: V matrix with the specified dimension.
    """
    return np.array([(-1) ** i for i in range(dim)])


def phase_matrix(dim: int, v: NDArray[np.floating]) -> NDArray[np.complex128]:
    """Phase shift diagonal matrix for a given dimension.

    The phase shift matrix is a parametric diagonal matrix with complex exponentials on the first half of the diagonal and ones on the second half.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).
        v (NDArray[float]): Vector of angles for the first half of the modes.

    Returns:
        NDArray[complex]: Phase shift matrix with the specified dimension.
    """
    diag = np.ones(dim, dtype=np.complex128)

    for i in range(dim // 2):
        diag[i] = np.exp(1.0j * v[i])

    return diag


def phase_matrix_swap(dim: int, v: NDArray[np.floating]) -> NDArray[np.complex128]:
    """Swapped phase shift diagonal matrix for a given dimension.

    The swapped phase shift matrix is a parametric diagonal matrix with complex exponentials on the second half of the diagonal and ones on the first half.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).
        v (NDArray[float]): Vector of angles for the second half of the modes.

    Returns:
        NDArray[complex]: Swapped phase shift matrix with the specified dimension.
    """
    diag = np.ones(dim, dtype=np.complex128)

    for i in range(dim // 2):
        diag[i + dim // 2] = np.exp(1.0j * v[i])

    return diag


def lambda_matrix(dim: int, phi: float) -> NDArray[np.complex128]:
    """Generates the Lambda diagonal matrix for a given dimension and phase.

    The lambda matrix is a parametric diagonal matrix with complex exponentials on the first half of the diagonal and their conjugates on the second half.
    This matrix is used to distribute edge phase shifts across all modes in the Bell decomposition.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).
        phi (float): Phase shift to be applied.

    Returns:
        NDArray[complex]: Lambda matrix with the specified dimension and phase.
    """
    return np.array([np.exp(-1.0j * phi)] * (dim // 2) + [np.exp(1.0j * phi)] * (dim // 2))


def l_matrix(dim: int, phi: int) -> NDArray[np.complex128]:
    """Generates the L diagonal matrix for a given dimension and phase.

    The L matrix is a parametric diagonal matrix. L is used in the diagonalization of the X @ lambda @ X matrix:
    X @ lambda @ X = F.T.conj() @ L @ F.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).
        phi (float): Phase shift to be applied.

    Returns:
        NDArray[complex]: L matrix with the specified dimension and phase.
    """
    return np.array([np.exp(1.0j * phi * (-1) ** (i + 1)) for i in range(dim)])


def y_matrix(dim: int) -> NDArray[np.complex128]:
    """Generates the Y circulant matrix for a given dimension.

    Y is used in the factorization of the X matrix: X = G Y G.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[complex]: Y matrix with the specified dimension.
    """
    column_1 = np.zeros(dim, dtype=np.complex128)
    column_1[0], column_1[dim // 2] = 1, -1.0j
    return 1 / SQRT2 * circulant(column_1)


def x_matrix(dim: int) -> NDArray[np.floating]:
    """Generates the X matrix for a given dimension.

    X represents a layer of 50:50 beam splitter acting between the first
    half and the second half of the modes. It can be factored as X = G Y G.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[float]: X matrix with the specified dimension.
    """
    block_11 = np.eye(dim // 2)
    block_22 = -np.eye(dim // 2)
    return 1 / SQRT2 * np.block([[block_11, block_11], [block_11, block_22]])


def x_matrix_swap(dim: int) -> NDArray[np.floating]:
    """Defines the X' matrix for a given dimension.
    X' represents a 50:50 beam splitter between the first half and the second half of the modes. X' is
    obtained by swapping the first and second half of the channels in the X matrix. X' can be factored as X' = G' Y G'.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[float]: X' matrix with the specified dimension.
    """
    block_11 = -np.eye(dim // 2)
    block_22 = np.eye(dim // 2)
    return 1 / SQRT2 * np.block([[block_11, block_22], [block_22, block_22]])


def swap_matrix(dim: int) -> NDArray[np.integer]:
    """Matrix that swaps the first half and the second half of the channels.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[int]: Swap matrix that rearranges the channels.
    """
    block_11 = np.zeros((dim // 2, dim // 2))
    block_12 = np.eye(dim // 2)
    return np.block([[block_11, block_12], [block_12, block_11]])


def p_permutation_matrix(dim: int) -> NDArray[np.integer]:
    """Matrix that performs a cyclic shift on the first half of the modes.

    P can be factored as P = X Ci X.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[int]: Permutation matrix.
    """
    output_matrix = np.zeros((dim, dim), dtype=np.int64)

    for i in range(dim):
        if i <= dim // 2 - 1:
            output_matrix[i, (i + 1) % (dim // 2)] = 1
        else:
            output_matrix[i, i] = 1

    return output_matrix


def c_matrix(dim: int) -> NDArray[np.integer]:
    """Generates the C circulant matrix for a given dimension.

    C is the cyclic shift matrix used in the definition of the Ci matrix.

    Args:
        dim (int): Dimension of the matrix.

    Returns:
        NDArray[int]: C circulant matrix with the specified dimension.
    """
    column_1 = np.zeros(dim)
    column_1[-1] = 1
    return circulant(column_1)


def ci_matrix(dim: int) -> NDArray[np.floating]:
    """Generates the Ci circulant matrix for a given dimension.

    Ci is the matrix used in the factorization of the P permutation matrix: P = X Ci X.

    Args:
        dim (int): Dimension of the matrix.

    Returns:
        NDArray[float]: Ci circulant matrix with the specified dimension.
    """
    block_11 = c_matrix(dim // 2) + np.eye(dim // 2)
    block_12 = c_matrix(dim // 2) - np.eye(dim // 2)
    return 1 / 2 * np.block([[block_11, block_12], [block_12, block_11]])


def pi_permutation_matrix(dim: int) -> NDArray[np.integer]:
    """Generates the Pi permutation matrix for a given dimension.

    Pi is a permutation matrix that can invert the DFT matrix, i.e., Pi @ DFT = DFT @ Pi = DFT.T.conj().

    Args:
        dim (int): Dimension of the matrix.

    Returns:
        NDArray[int]: Pi permutation matrix with the specified dimension.
    """
    pi_matrix = np.zeros((dim, dim), dtype=np.int64)
    pi_matrix[0, 0] = 1

    for i in range(1, dim):
        pi_matrix[i, dim - i] = 1

    return pi_matrix


def pi_transformation(U: NDArray) -> NDArray:
    """Performs a transformation on the input matrix U using the Pi permutation matrix.

    The output is given by Pi @ U @ Pi.

    Args:
        U (NDArray): Input matrix to be transformed.

    Returns:
        NDArray: Transformed matrix.
    """
    dim = U.shape[0]
    return pi_permutation_matrix(dim) @ U @ pi_permutation_matrix(dim)


def channels_permutation(dim: int) -> NDArray[np.integer]:
    """Matrix that permutes the channels of an interferometer to be decomposed with the LPLM algorithm.

    Args:
        dim (int): Dimension of the interferometer (Number of modes).

    Returns:
        NDArray[int]: Permutation matrix that rearranges the channels.
    """
    matrix = np.zeros((dim, dim), dtype=np.int64)

    for i in range(dim):
        if i % 2 == 0:
            matrix[i // 2, i] = 1
        else:
            matrix[(dim + i - 1) // 2, i] = 1

    return matrix


def matrix_interleave(
    mask_sequence: list[NDArray[np.number]], mixing_layer: NDArray[np.number] | None = None
) -> NDArray[np.number]:
    """Interleave phase masks with a given mixing layer.

    Given a list of phase masks [M1, M2, ..., Mn] and a mixing layer F, this function returns the matrix given by
        F @ M1 @ F @ M2 @ ... @ Mn.

    If no mixing layer is provided, the function uses the normalized DFT matrix as F.

    Args:
        mask_sequence (list[NDArray]): List of phase masks to be interleaved with the mixing layer.
        mixing_layer (NDArray, optional): Mixing layer to be applied between the masks. Default is the normalized DFT matrix.

    Returns:
        NDArray: The resulting matrix.
    """
    dim = len(mask_sequence[0])

    # If no mixing layer is provided, use the normalized DFT matrix
    if mixing_layer is None:
        mixing_layer = dft(dim, scale="sqrtn")

    # Interleave the masks with the mixing layer
    output = np.eye(dim, dtype=np.complex128)
    for mask in mask_sequence:
        output = output @ (mixing_layer * mask)

    return output
