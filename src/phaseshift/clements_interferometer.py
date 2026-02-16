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
# Clements interferometer module.

This module provides functions to perform the Clements *et al.* decomposition of a unitary matrix into a planar mesh of
asymmetric Mach-Zehnder interferometers and phase shifters, as well as to reconstruct the unitary matrix from the decomposition.

The main functions are:
- `clements_decomposition`: Decomposes a unitary matrix into a rectangular network of variable beam splitters (`beam_splitter`).
- `mzi_decomposition`: Decomposes a unitary matrix into a rectangular network of asymmetric Mach-Zehnder interferometers (`mzi_matrix`).
- `circuit_reconstruction`: Reconstructs the unitary matrix from the decomposition obtained with `clements_decomposition`.
- `mzi_circuit_reconstruction`: Reconstructs the unitary matrix from the decomposition obtained with `mzi_decomposition`.

The Clements *et al.* decomposition is described in detail in [1].

### References
1. Clements, William R., et al. "Optimal design for universal multiport interferometers." Optica 3.12 (2016): 1460-1465.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag


def beam_splitter(
    dim: int, target: tuple[int, int], theta: float, phi: float
) -> NDArray[np.complex128]:
    """Matrix form of a beam splitter.

    This function returns the matrix form `u_bs` of a lossless variable beam splitter acting on channels `m` and `n` = `m` + 1 in a N = `dim` modes circuit.
    The beam splitter is parametrized by a reflectivity cos(`theta`) and a phase shift `phi`. `u_bs` is a block diagonal unitary matrix and its 2 x 2 interaction
    block is given by:

        [
            [exp(1.j * phi) * cos(theta), -sin(theta)],
            [exp(1.j * phi) * sin(theta), cos(theta)]
        ].

    `u_bs` is the unit cell used in the paper by Clements *et al.* [1].

    Args:
        dim (int): Number of modes in the circuit.
        target (tuple[int, int]): (m, n) - Indices of the two consecutive modes that the beam splitter acts on. (`n` = `m` + 1).
        theta (float): [0, pi/2] - Reflectivity of the beam splitter.
        phi (float): [0, 2*pi] - Phase shift of the beam splitter.

    Returns:
        NDArray[complex]: `u_bs` - Matrix form of the beam splitter.

    Raises:
        ValueError: If `m` and `n` are not consecutive integers.

    ### References:
        [1]: William R. Clements, Peter C. Humphreys, Benjamin J. Metcalf, W. Steven Kolthammer, and Ian A. Walmsley, "Optimal design for universal multiport interferometers," Optica 3, 1460-1465 (2016)
    """
    # Target modes
    m, n = target

    # Check if m and n are consecutive integers
    if not n - m == 1:
        raise ValueError("m and n must be consecutive integers.")

    # Beam splitter matrix acting on the two target modes
    u_bs_block = np.array(
        [
            [np.exp(1.0j * phi) * np.cos(theta), -np.sin(theta)],
            [np.exp(1.0j * phi) * np.sin(theta), np.cos(theta)],
        ]
    )

    # Construct the block diagonal matrix
    u_bs = block_diag(np.eye(m), u_bs_block, np.eye(dim - m - 2))

    return u_bs


def mzi_matrix(
    dim: int, target: tuple[int, int], theta: float, phi: float
) -> NDArray[np.complex128]:
    """Matrix form of a Mach-Zehnder interferometer with an external phase shifter.

    This function returns the matrix form `u_mzi` of a Mach-Zehnder interferometer + phase shifter acting on channels `m` and `n` = `m` + 1 in a N = `dim` modes circuit.
    The Mach-Zehnder interferometer is parametrized by an internal phase shift `theta` and an external phase shift `phi`. `u_mzi` is a block diagonal unitary matrix and is the unit cell
    of the Clements interferometer. Its 2 x 2 transformation is given by:

        exp(1.j * theta) * [[
            [exp(1.j * phi) * cos(theta), 1.j * sin(theta)],
            [exp(1.j * phi) * 1.j * sin(theta), cos(theta)]
        ]].

    `u_mzi` can be decomposed into a sequence of two 50:50 beam splitters and two phase shifters acting on the first mode: `u_mzi` = X @ P(2 * `theta`) @ X @ P(`phi`),
    where `X` is the matrix form of a 50:50 beam splitter (2 x 2 Hadamard matrix) and `P(x)` is a phase shift of `x` on the first mode, i.e:

        P(x) = [[
            exp(1.j * x), 0],
            [0, 1]
        ]].

    Args:
        dim (int): Number of modes in the circuit.
        target (tuple[int, int]): (m, n) - Indices of the two consecutive modes that the Mach-Zehnder interferometer acts on. (`n` = `m` + 1).
        theta (float): [0, pi/2] - Internal phase shift of the Mach-Zehnder interferometer.
        phi (float): [0, 2*pi] - External phase shift of the Mach-Zehnder interferometer.

    Returns:
        NDArray[complex]: `u_mzi` - Matrix form of the Mach-Zehnder interferometer.

    Raises:
        ValueError: If `m` and `n` are not consecutive integers.
    """

    # Target modes
    m, n = target

    # Check if m and n are consecutive integers
    if not n - m == 1:
        raise ValueError("m and n must be consecutive integers.")

    # Initialize the unitary matrix
    u_mzi = np.eye(dim, dtype=np.complex128)

    # Mach-Zehnder interferometer matrix acting on the two target modes
    global_phase = np.exp(1.0j * theta)
    u_mzi[m, m] = global_phase * np.exp(1.0j * phi) * np.cos(theta)
    u_mzi[m, n] = global_phase * 1.0j * np.sin(theta)
    u_mzi[n, m] = global_phase * np.exp(1.0j * phi) * 1.0j * np.sin(theta)
    u_mzi[n, n] = global_phase * np.cos(theta)

    return u_mzi


class MachZehnder(NamedTuple):
    """Named tuple to store unit cell parameters.

    `MachZehnder` objects can contain the parameters of a `beam_splitter` or `mzi_matrix`.

    Attributes:
        theta (float): Internal phase shift of the MZI.
        phi (float): External phase shift of the MZI.
        target (tuple[int, int]): Indices of the two consecutive modes that the unit cell acts on.
    """

    theta: float
    """Internal phase shift of the Mach-Zehnder interferometer."""

    phi: float
    """External phase shift of the Mach-Zehnder interferometer."""

    target: tuple[int, int]
    """Target modes of the unit cell."""


class Decomposition(NamedTuple):
    """Named tuple to store the decomposition of a unitary matrix.

    `Decomposition` objects can store the output of the `clements_decomposition` or `mzi_decomposition` functions.

    Attributes:
        D (NDArray[complex]): Single-mode phase shifts of the output circuit.
        circuit (list[MachZehnder]): List of `MachZehnder` (theta, phi, (m, n)).
    """

    D: NDArray[np.complex128]
    """Array of single-mode phase shifts of the output circuit."""

    circuit: list[MachZehnder]
    """List of `MachZehnder` (theta, phi, (m, n)) in the circuit."""


def clements_decomposition(U: NDArray[np.complex128]) -> Decomposition:
    """Clements decomposition of a unitary matrix.

    This function decomposes a unitary matrix `U` acting on N modes into a sequence of beam splitters (`beam_splitter`) and a diagonal unitary matrix `D` of single-mode phase-shifts.
    This function implements the algorithm described in [1] in order to create a rectangular mesh of beam splitter and phase shifters. The function returns a named
    tuple `Decomposition` containing the diagonal phase-shifts `D` and the sequence of beam splitter parameters in the circuit.

    Args:
        U (NDArray[complex]): Unitary matrix to be decomposed.

    Returns:
        Decomposition: A named tuple containing the diagonal phase-shifts `D` and the sequence of beam splitter parameters in the circuit.
            - `D` is a diagonal unitary matrix of single-mode phase-shifts.
            - `circuit` is a list of `MachZehnder` objects, each containing the parameters of a beam splitter in the circuit.

    Raises:
        TypeError: If `U` is not a numpy array.
        ValueError: If `U` is not a unitary matrix.

    ### Refenrence:
        [1]: William R. Clements, Peter C. Humphreys, Benjamin J. Metcalf, W. Steven Kolthammer, and Ian A. Walmsley, "Optimal design for universal multiport interferometers," Optica 3, 1460-1465 (2016)
    """

    # Check if U is a numpy array
    if not isinstance(U, np.ndarray):
        raise TypeError("U must be a numpy array.")

    # Check if U is a unitary matrix
    if not np.allclose(U @ U.T.conj(), np.eye(U.shape[0])):
        raise ValueError("U must be a unitary matrix.")

    dim = U.shape[0]
    right_sequence: list[MachZehnder] = []
    left_sequence: list[MachZehnder] = []
    U = np.array(U, dtype=np.complex128)

    # Null the off-diagonals elements of U following the snakes and ladders rule
    for k in range(1, dim):

        # If k is odd, multiply by T^-1 on the right to null the element
        if k % 2 == 1:
            for i, j in zip(range(dim - 1, dim - k - 1, -1), range(k - 1, -1, -1)):
                if U[i, j] == 0:
                    # Skip if the element is already null
                    continue
                elif U[i, j + 1] == 0:
                    # Avoid division by zero
                    phi = 0.0
                else:
                    # Compute phi to null the element
                    phi = np.angle(U[i, j] / U[i, j + 1]) % (2 * np.pi)

                # Compute theta to null the element
                theta = np.arctan2(np.abs(U[i, j]), np.abs(U[i, j + 1]))
                right_sequence.append(MachZehnder(theta, phi, (j, j + 1)))
                U = U @ beam_splitter(dim, (j, j + 1), theta, phi).conj().T

        # If k is even, multiply by T on the left to null the element
        else:
            for i, j in zip(range(dim - k, dim), range(0, k)):
                if U[i, j] == 0:
                    # Skip if the element is already null
                    continue
                elif U[i - 1, j] == 0:
                    # Avoid division by zero
                    phi = 0.0
                else:
                    # Compute phi to null the element
                    phi = np.angle(-U[i, j] / U[i - 1, j]) % (2 * np.pi)

                # Compute theta to null the element
                theta = np.arctan2(np.abs(U[i, j]), np.abs(U[i - 1, j]))
                left_sequence.insert(0, MachZehnder(theta, phi, (i - 1, i)))
                U = beam_splitter(dim, (i - 1, i), theta, phi) @ U

    # Find a new diagonal unitary matrix D and new beam splitter parameters to change the order of the circuit so that
    # the single-mode phase shifts are at the beginning of the circuit.
    D = np.diag(U).copy()
    new_left_sequence: list[MachZehnder] = []

    for theta, phi, (m, n) in left_sequence:
        new_phi = (np.pi + np.angle(D[m]) - np.angle(D[n])) % (2 * np.pi)
        new_left_sequence.insert(0, MachZehnder(theta, new_phi, (m, n)))
        new_alpha = np.angle(D[n]) - np.pi - phi
        D[m] = np.exp(1.0j * new_alpha)

    # Return the new diagonal unitary matrix D and the entire sequence of beam splitter parameters
    return Decomposition(D=D, circuit=new_left_sequence + right_sequence[::-1])


def mzi_decomposition(U: NDArray[np.complex128]) -> Decomposition:
    """Clements decomposition of a unitary matrix using a mesh of Mach-Zehnder interferometers.

    This function decomposes a unitary matrix `U` acting on N modes into a sequence of Mach-Zehnder interferometers and a diagonal unitary matrix `D` of single-mode phase-shifts.
    This function adapts the Clements decomposition algorithm described in [1] to use the exact transfer matrix of a Mach-Zehnder interferometer with an external phase shifter as defined in
    the `mzi_matrix` function. The algorithm creates a rectangular mesh of Mach-Zehnder interferometers and phase shifters.

    Args:
        U (NDArray[complex]): Unitary matrix to be decomposed.

    Returns:
        Decomposition: A named tuple containing the diagonal phase-shifts `D` and the sequence of Mach-Zehnder parameters in the circuit.
            - `D` is a diagonal unitary matrix of single-mode phase-shifts.
            - `circuit` is a list of `MachZehnder` named tuples, each containing the parameters of a Mach-Zehnder interferometer in the circuit.

    Raises:
        TypeError: If `U` is not a numpy array.
        ValueError: If `U` is not a unitary matrix.

    Reference:
        [1]: William R. Clements, Peter C. Humphreys, Benjamin J. Metcalf, W. Steven Kolthammer, and Ian A. Walmsley, "Optimal design for universal multiport interferometers," Optica 3, 1460-1465 (2016)
    """

    # Check if U is a numpy array
    if not isinstance(U, np.ndarray):
        raise TypeError("U must be a numpy array.")

    # Check if U is a unitary matrix
    if not np.allclose(U @ U.T.conj(), np.eye(U.shape[0])):
        raise ValueError("U must be a unitary matrix.")

    dim = U.shape[0]
    right_sequence: list[MachZehnder] = []
    left_sequence: list[MachZehnder] = []
    U = np.array(U, dtype=np.complex128)

    # Null the off-diagonals elements of U following the snakes and ladders rule
    for k in range(1, dim):

        # If k is odd, multiply by T^-1 on the right to null the element
        if k % 2 == 1:
            for i, j in zip(range(dim - 1, dim - k - 1, -1), range(k - 1, -1, -1)):
                if U[i, j] == 0:
                    # Skip if the element is already null
                    continue
                elif U[i, j + 1] == 0:
                    # Avoid division by zero
                    phi = 0.0
                else:
                    # Compute phi to null the element
                    phi = (np.angle(U[i, j] / U[i, j + 1]) - np.pi / 2) % (2 * np.pi)

                # Compute theta to null the element
                theta = np.arctan2(np.abs(U[i, j]), np.abs(U[i, j + 1]))
                right_sequence.append(MachZehnder(theta, phi, (j, j + 1)))
                U = U @ mzi_matrix(dim, (j, j + 1), theta, phi).conj().T

        # If k is even, multiply by T on the left to null the element
        else:
            for i, j in zip(range(dim - k, dim), range(0, k)):
                if U[i, j] == 0:
                    # Skip if the element is already null
                    continue
                elif U[i - 1, j] == 0:
                    # Avoid division by zero
                    phi = 0.0
                else:
                    # Compute phi to null the element
                    phi = (np.angle(U[i, j] / U[i - 1, j]) + np.pi / 2) % (2 * np.pi)

                # Compute theta to null the element
                theta = np.arctan2(np.abs(U[i, j]), np.abs(U[i - 1, j]))
                left_sequence.insert(0, MachZehnder(theta, phi, (i - 1, i)))
                U = mzi_matrix(dim, (i - 1, i), theta, phi) @ U

    # Find a new diagonal unitary matrix D and new beam splitter parameters to change the order of the circuit so that
    # the single-mode phase shifts are at the beginning of the circuit.
    D = np.diag(U).copy()
    new_left_sequence: list[MachZehnder] = []

    for theta, phi, (m, n) in left_sequence:
        new_phi = (np.pi + np.angle(D[m]) - np.angle(D[n])) % (2 * np.pi)
        new_left_sequence.insert(0, MachZehnder(theta, new_phi, (m, n)))
        new_beta = np.angle(D[n]) - 2 * theta
        new_alpha = np.angle(D[n]) - np.pi - phi - 2 * theta
        D[m] = np.exp(1.0j * new_alpha)
        D[n] = np.exp(1.0j * new_beta)

    # Return the new diagonal unitary matrix D and the entire sequence of beam splitter parameters
    return Decomposition(D=D, circuit=new_left_sequence + right_sequence[::-1])


def circuit_reconstruction(
    decomposition: Decomposition,
    loss_db: float = 0.0,
) -> NDArray[np.complex128]:
    """Reconstruct a unitary matrix from its decomposition using variable beam splitters.

    Given the decomposition of a unitary matrix into a mesh of beam splitters and phase shifters, this function reconstructs the original unitary matrix.
    This function uses the matrix form defined in the `beam_splitter` function as a unit cell.
    Optionally, applies loss (in dB) to each unit cell.

    Args:
        decomposition (Decomposition): Decomposition of the unitary matrix. This decomposition can be obtained using the `clements_decomposition` function.
        loss_db (float, optional): Loss in decibels (dB) to apply to each unit cell. Default is 0 (no loss).

    Returns:
        NDArray[complex]: Reconstructed unitary matrix with optional loss.

    Raises:
        TypeError: If `decomposition` is not a Decomposition object.
    """
    # Check if decomposition is a Decomposition object
    if not isinstance(decomposition, Decomposition):
        raise TypeError("decomposition must be a Decomposition object.")

    # Initialize the reconstructed matrix as a diagonal matrix with the phase shifts from decomposition.D
    reconstructed_matrix = np.diag(decomposition.D)

    # Iterate over the beam splitters in the decomposition circuit
    # and apply the beam splitter matrices to reconstruct the unitary.
    for theta, phi, (m, n) in decomposition.circuit:

        bs_matrix = beam_splitter(reconstructed_matrix.shape[0], (m, n), theta, phi)

        if loss_db != 0.0:
            # Convert dB loss to linear loss
            loss_linear = 10 ** (-loss_db / 20)

            # Apply loss only to the two modes affected by the beamsplitter
            loss_mat = np.eye(reconstructed_matrix.shape[0], dtype=np.complex128)
            loss_mat[m, m] = loss_linear
            loss_mat[n, n] = loss_linear
            bs_matrix = loss_mat @ bs_matrix

        reconstructed_matrix = reconstructed_matrix @ bs_matrix

    return reconstructed_matrix


def mzi_circuit_reconstruction(decomposition: Decomposition) -> NDArray[np.complex128]:
    """Reconstruct a unitary matrix from its decomposition using Mach-Zehnder interferometers.

    Given the decomposition of a unitary matrix into a mesh of Mach-Zehnder interferometers and phase shifters, this function reconstructs the original unitary matrix.
    The function uses the exact transfer matrix of a Mach-Zehnder interferometer with an external phase shifter as defined in the `mzi_matrix` function.

    Args:
        decomposition (Decomposition): Decomposition of the unitary matrix. This decomposition can be obtained using the `mzi_decomposition` function.

    Returns:
        NDArray[complex]: Reconstructed unitary matrix.

    Raises:
        TypeError: If `decomposition` is not a Decomposition object.
    """

    # Check if decomposition is a Decomposition object
    if not isinstance(decomposition, Decomposition):
        raise TypeError("decomposition must be a Decomposition object.")

    # Initialize the reconstructed matrix as a diagonal matrix with the phase shifts from decomposition.D
    reconstructed_matrix = np.diag(decomposition.D)

    # Iterate over the Mach-Zehnder interferometers in the decomposition circuit
    # and apply the Mach-Zehnder matrices to reconstruct the unitary.
    for theta, phi, (m, n) in decomposition.circuit:

        mzi_matrix_ = mzi_matrix(reconstructed_matrix.shape[0], (m, n), theta, phi)
        reconstructed_matrix = reconstructed_matrix @ mzi_matrix_

    return reconstructed_matrix
