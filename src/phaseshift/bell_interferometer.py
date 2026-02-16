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
# Bell Interferometer

This module provides functions to perform the Bell *et al.* decomposition of a unitary matrix into a rectangular mesh
of symmetric Mach-Zehnder interferometers and phase-shifters. The Bell decomposition is an improved version of the Clements decomposition,
which allows for a more compact representation of linear optical unitaries. The module includes the following functions:

- `bell_decomposition`: Decomposes a unitary matrix using the Bell *et al.* decomposition.
- `circuit_reconstruction`: Reconstructs the unitary matrix from the Bell decomposition.

The Bell *et al.* decomposition is described in [1].

### References
1. Bell, Bryn A., and Ian A. Walmsley. "Further compactifying linear optical unitaries." Apl Photonics 6.7 (2021).

**Note**: The code used in this module is adapted from [*strawberryfields*](https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/decompositions.py).
It was introduced in [this PR by Jake Bulmer and Yuan Yao](https://github.com/XanaduAI/strawberryfields/pull/584#issue-894649549).
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


def symmetric_mzi(
    dim: int, targets: tuple[int, int], delta: float, sigma: float
) -> NDArray[np.complex128]:
    """Matrix form of the symmetric Mach-Zehnder interferometer (sMZI).

    The symmetric Mach-Zehnder interferometer is a symmetric block unitary matrix that
    acts on two consecutive modes, defined by the target tuple (m, n).
    It consists of two 50:50 beam splitters and two internal phase shifters.
    The 2x2 interaction can be decomposed as:

    T = X @ diag(exp(1.j * `theta_1`), exp(1.j * `theta_2`)) @ X,

    where X is the Hadamard matrix, and `theta_1` and `theta_2` are the internal phase shifts.

    Args:
        dim (int): The dimension of the unitary matrix. (Number of modes)
        targets (tuple[int, int]): The two consecutive modes (m, n) that the sMZI acts on.
        delta (float): Splitting ratio of the sMZI, defined as (`theta_1` - `theta_2`) / 2, where `theta_1` and `theta_2` are the two internal phase shifts.
        sigma (float): Global phase shift of the sMZI, defined as (`theta_1` + `theta_2`) / 2, where `theta_1` and `theta_2` are the two internal phase shifts.

    Returns:
        NDArray[complex]: The unitary matrix representing the symmetric Mach-Zehnder interferometer.

    Raises:
        ValueError: If the dimension is less than 2.
        ValueError: If the target modes are not consecutive integers.
    """
    # Check if the dimension is at least 2
    if dim < 2:
        raise ValueError("Dimension must be at least 2.")

    # Target modes
    m, n = targets

    # Check if m and n are consecutive integers
    if not n - m == 1:
        raise ValueError("m and n must be consecutive integers.")

    # Construct the symmetric Mach-Zehnder interferometer matrix
    matrix = np.eye(dim, dtype=np.complex128)
    matrix[m, m] = np.exp(1j * sigma) * np.cos(delta)
    matrix[m, n] = 1.0j * np.exp(1j * sigma) * np.sin(delta)
    matrix[n, m] = 1.0j * np.exp(1j * sigma) * np.sin(delta)
    matrix[n, n] = np.exp(1j * sigma) * np.cos(delta)

    return matrix


def phase_shifter(dim: int, target: int, angle: float) -> NDArray[np.complex128]:
    """Phase shifter matrix.

    The phase shifter matrix introduces a phase shift on a target mode. It consists
    of a diagonal matrix where the diagonal element corresponding to the target mode
    is set to exp(1.j * `angle`), and all other diagonal elements are set to 1.

    Args:
        dim (int): The dimension of the unitary matrix.
        target (int): The target mode where the phase shift is applied.
        angle (float): The phase shift angle in radians.

    Returns:
        NDArray[complex]: The unitary matrix representing the phase shifter.
    """
    # Create an identity matrix of the specified dimension
    matrix = np.identity(dim, dtype=np.complex128)

    # Set the diagonal element corresponding to the target mode to exp(1.j * angle)
    matrix[target, target] = np.exp(1j * angle)

    return matrix


@dataclass
class BellDecomp:
    """Dataclass to store the parameters of the Bell decomposition.

    Attributes:
        dim (int): The dimension of the unitary matrix (number of modes).
        phi_input (dict[int, float]): Phase-shifts at the circuit input. Keys are mode indexes and values are angles in rad.
        phi_output (dict[int, float]): Phase-shifts at the circuit output. Keys are mode indexes and values are angles in rad.
        phi_edge (dict[tuple[int, int], float]): Phase-shifts at the edges of the circuit. Keys are tuples (mode, layer) and values are angles in rad.
        sigma (dict[tuple[int, int], float]): Sigma parameters of the symmetric Mach-Zehnder interferometers (sMZIs). Keys are tuples (mode, layer) and values are angles in rad.
        delta (dict[tuple[int, int], float]): Delta parameters of the symmetric Mach-Zehnder interferometers (sMZIs). Keys are tuples (mode, layer) and values are angles in rad.
    """

    dim: int
    """The dimension of the unitary matrix (number of modes)."""

    phi_input: dict[int, float] = field(default_factory=dict)
    """Phase-shifts at the circuit input. `mode` : `angle`"""

    phi_output: dict[int, float] = field(default_factory=dict)
    """Phase-shifts at the circuit output. `mode` : `angle`"""

    phi_edge: dict[tuple[int, int], float] = field(default_factory=dict)
    """Phase-shifts at the edges of the circuit. (`mode`, `layer`) : `angle`"""

    sigma: dict[tuple[int, int], float] = field(default_factory=dict)
    """Sigma parameters of the symmetric Mach-Zehnder interferometers (sMZIs). (`mode`, `layer`) : `angle`"""

    delta: dict[tuple[int, int], float] = field(default_factory=dict)
    """Delta parameters of the symmetric Mach-Zehnder interferometers (sMZIs). (`mode`, `layer`) : `angle`"""


def _bell_decomposition_init(U: NDArray[np.complex128]) -> BellDecomp:
    """Rectangular decomposition of a unitary with sMZIs and phase-shifters.

    First step of the Bell decomposition. This function decompose a N x N unitary matrix `U` into a rectangular mesh
    of symmetric Mach-Zehnder interferometers (sMZIs) and phase-shifters. The decomposition generates a layer of phase
    shifters in the middle of the circuit, and at the beginning and end of the circuit. The middle phase shifters can be
    relocated to the edges of the circuit with the `_absorb_middle_phases` function. For more details, see [1].

    Args:
        U (NDArray[complex]): Unitary matrix to decompose.

    Returns:
        BellDecomp: A dataclass with the following items

        * `dim`: Dimension of the unitary matrix. (Number of modes)
        * `phi_input`: Phase-shifts at the circuit input. (`mode` ; `angle`)
        * `phi_output`: Phase-shifts at the circuit output. (`mode` ; `angle`)
        * `phi_edge`: Phase-shifts at the edges of the circuit. ((`mode`, `layer`) ; `angle`)
        * `sigma`: Sigma parameters of the symmetric Mach-Zehnder interferometers (sMZIs). ((`mode`, `layer`) ; `angle`)
        * `delta`: Delta parameters of the symmetric Mach-Zehnder interferometers (sMZIs). ((`mode`, `layer`) ; `angle`)

    ### References
        [1]: B. A. Bell, I. A. Walmsley; Further compactifying linear optical unitaries. APL Photonics 1 July 2021; 6 (7): 070804. https://doi.org/10.1063/5.0053421
    """

    # Initialize the decomposition object
    dim = U.shape[0]
    decomp = BellDecomp(dim=dim)

    # Set auxillary matrix V
    V = U.conj()

    # Iterate over diagonals j in the mesh
    for j in range(dim - 1):

        # Even diagonal: Apply ladder rule
        if j % 2 == 0:
            row = dim - 1
            column = j

            # Compute phi so that angle(1.j * V[row, column]) = angle(V[row, column + 1])
            phi_j = np.angle(V[row, column + 1]) - np.angle(V[row, column]) - np.pi / 2
            V = V @ phase_shifter(dim, j, phi_j)
            decomp.phi_input[j] = phi_j

            # Iterate over the modes in the diagonal
            for k in range(j + 1):

                # Compute delta to null the matrix element V[row, column]
                if V[row, column] == 0:
                    delta = 0
                else:
                    delta = np.arctan2(abs(V[row, column]), abs(V[row, column + 1]))

                # Null the matrix element
                n = j - k  # First mode of the sMZI
                V_temp = V @ symmetric_mzi(dim, (n, n + 1), delta, 0)

                # Compute sigma so that angle(1.j * V[row - 1, column - 1]) = angle(V[row - 1, column])
                if j != k:
                    sigma = (
                        np.angle(V_temp[row - 1, column - 1])
                        - np.angle(V_temp[row - 1, column])
                        + np.pi / 2
                    )
                else:
                    sigma = 0

                # Compute the new V matrix
                V = V @ symmetric_mzi(dim, (n, n + 1), delta, sigma)
                decomp.delta[n, k] = delta
                decomp.sigma[n, k] = sigma
                row -= 1
                column -= 1

        # Odd diagonal: Apply the snake rule
        else:
            row = dim - j - 1
            column = 0

            # Compute phi so that angle(1.j * V[row, column]) = angle(V[row - 1, column])
            phi_j = np.angle(V[row - 1, column]) - np.angle(V[row, column]) - np.pi / 2
            V = phase_shifter(dim, row, phi_j) @ V
            decomp.phi_output[row] = phi_j

            # Iterate over the modes in the diagonal
            for k in range(j + 1):

                # Compute delta to null the matrix element V[row, column]
                if V[row, column] == 0:
                    delta = 0
                else:
                    delta = np.arctan2(abs(V[row, column]), abs(V[row - 1, column]))

                # Null the matrix element
                V_temp = symmetric_mzi(dim, (row - 1, row), delta, 0) @ V
                n = dim + k - j - 2  # First mode of the sMZI

                # Compute sigma so that angle(1.j * V[row + 1, column + 1]) = angle(V[row, column + 1])
                if j != k:
                    sigma = (
                        np.angle(V_temp[row + 1, column + 1])
                        - np.angle(V_temp[row, column + 1])
                        + np.pi / 2
                    )
                else:
                    sigma = 0

                # Compute the new V matrix
                V = symmetric_mzi(dim, (n, n + 1), delta, sigma) @ V
                decomp.delta[n, dim - k - 1] = delta
                decomp.sigma[n, dim - k - 1] = sigma
                row += 1
                column += 1

    # Multiply V by phase shifters to cancel phases on the diagonal to get the identity matrix.
    phi_middle = {}
    for j in range(dim):
        zeta = -np.angle(V[j, j])
        V = V @ phase_shifter(dim, j, zeta)
        phi_middle[j] = zeta
    setattr(decomp, "_phi_middle", phi_middle)

    # Assert that V is the identity matrix
    assert np.allclose(V, np.eye(dim)), "decomposition failed"

    return decomp


def _absorb_middle_phases(decomp: BellDecomp) -> BellDecomp:
    """Relocate the middle phase-shifters to the edges of the circuit.

    Second step of the Bell decomposition. This function takes the output of `_bell_decomposition_init` and relocates the middle phase-shifters
    to the edges of the circuit, where they do not contribute to circuit depth. For more details, see [1].

    Args:
        decomp (BellDecomp): The output of `_bell_decomposition_init`.

    Returns:
        BellDecomp: A dataclass with the following items

        * `dim`: Dimension of the unitary matrix. (Number of modes)
        * `phi_input`: Phase-shifts at the circuit input. (`mode` ; `angle`)
        * `phi_output`: Phase-shifts at the circuit output. (`mode` ; `angle`)
        * `phi_edge`: Phase-shifts at the edges of the circuit. ((`mode`, `layer`) ; `angle`)
        * `sigma`: Sigma parameters of the symmetric Mach-Zehnder interferometers (sMZIs). ((`mode`, `layer`) ; `angle`)
        * `delta`: Delta parameters of the symmetric Mach-Zehnder interferometers (sMZIs). ((`mode`, `layer`) ; `angle`)

    ### References
        [1]: B. A. Bell, I. A. Walmsley; Further compactifying linear optical unitaries. APL Photonics 1 July 2021; 6 (7): 070804. https://doi.org/10.1063/5.0053421
    """

    dim = decomp.dim
    edge: dict[tuple[int, int], float] = defaultdict(float)

    # Matrix of even dimension
    if dim % 2 == 0:
        # The first middle phase-shifter is already at the output and does not need to be relocated.
        decomp.phi_output[0] = getattr(decomp, "_phi_middle")[0]

        for j in range(1, dim):
            zeta = getattr(decomp, "_phi_middle")[j]
            layer = dim - j

            # Add the phase zeta on the right sMZI
            for mode in range(j, dim - 1, 2):
                decomp.sigma[mode, layer] += zeta

            # Remove the phase zeta from the left sMZI
            for mode in range(j + 1, dim - 1, 2):
                decomp.sigma[mode, layer - 1] -= zeta

            # Locate the phase-shifter at the edge of the circuit
            if layer % 2 == 1:
                edge[dim - 1, layer] += zeta
            else:
                edge[dim - 1, layer - 1] -= zeta

        # The last edge phase-shifter is already at the output.
        decomp.phi_output[dim - 1] = edge[dim - 1, dim - 1]
        del edge[dim - 1, dim - 1]

    # Matrix of odd dimension
    else:
        for j in range(dim):
            zeta = getattr(decomp, "_phi_middle")[j]
            layer = dim - j - 1

            # Add the phase zeta on the right sMZI
            for mode in range(j, dim - 1, 2):
                decomp.sigma[mode, layer] += zeta

            # Remove the phase zeta from the left sMZI
            for mode in range(j + 1, dim - 1, 2):
                decomp.sigma[mode, layer - 1] -= zeta

            # Locate the phase-shifter at the edge of the circuit
            if layer % 2 == 0:
                edge[dim - 1, layer] += zeta
            else:
                edge[dim - 1, layer - 1] -= zeta

        # The last edge phase-shifter is already at the output.
        decomp.phi_output[dim - 1] = edge[dim - 1, dim - 1]
        del edge[dim - 1, dim - 1]

        # The first edge phase-shifter is already at the input.
        decomp.phi_input[dim - 1] = edge[dim - 1, 0]
        del edge[dim - 1, 0]

    delattr(decomp, "_phi_middle")
    decomp.phi_edge = dict(edge)

    return decomp


def bell_decomposition(U: NDArray[np.complex128]) -> BellDecomp:
    """Bell decomposition of a unitary matrix.

    This function decomposes a unitary matrix `U` into a rectangular mesh of symmetric Mach-Zehnder interferometers (sMZIs) and phase-shifters.
    The Bell decomposition is an improved version of the Clements decomposition, which allows for a more compact representation of linear optical unitaries.
    The function returns a `BellDecomp` dataclass containing the parameters of the decomposition, including phase shifts at the input and output, as well as the parameters of the sMZIs.
    For more details, see [1].

    Args:
        U (NDArray[complex]): Unitary matrix to decompose.

    Returns:
        BellDecomp: A dataclass with the following items

        * `dim`: Dimension of the unitary matrix. (Number of modes)
        * `phi_input`: Phase-shifts at the circuit input. (`mode` ; `angle`)
        * `phi_output`: Phase-shifts at the circuit output. (`mode` ; `angle`)
        * `phi_edge`: Phase-shifts at the edges of the circuit. ((`mode`, `layer`) ; `angle`)
        * `sigma`: Sigma parameters of the symmetric Mach-Zehnder interferometers (sMZIs). ((`mode`, `layer`) ; `angle`)
        * `delta`: Delta parameters of the symmetric Mach-Zehnder interferometers (sMZIs). ((`mode`, `layer`) ; `angle`)

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input matrix is not square or not unitary.

    ### References
        [1]: B. A. Bell, I. A. Walmsley; Further compactifying linear optical unitaries. APL Photonics 1 July 2021; 6 (7): 070804. https://doi.org/10.1063/5.0053421
    """
    if not isinstance(U, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if not U.shape[0] == U.shape[1]:
        raise ValueError("Input matrix must be square")

    if not np.allclose(U @ U.conj().T, np.eye(U.shape[0])):
        raise ValueError("The input matrix must be unitary")

    decomp_temp = _bell_decomposition_init(U)
    return _absorb_middle_phases(decomp_temp)


def circuit_reconstruction(decomp: BellDecomp) -> NDArray[np.complex128]:
    """Reconstruct the unitary matrix from the Bell decomposition.

    Given a `BellDecomp` instance, this function reconstructs the unitary matrix by applying
    the phase shifters and symmetric Mach-Zehnder interferometers (sMZIs) in the order specified by the decomposition.
    The `BellDecomp` instance can be obtained from the `bell_decomposition` function.

    Args:
        decomp (BellDecomp): The Bell decomposition containing the parameters of the circuit.

    Returns:
        NDArray[complex]: The reconstructed unitary matrix.

    Raises:
        TypeError: If the input is not a BellDecomp instance.
    """

    if not isinstance(decomp, BellDecomp):
        raise TypeError("Input must be a BellDecomp instance")

    dim = decomp.dim
    U = np.eye(dim, dtype=np.complex128)

    # Apply the phase shifters at the input
    for mode, phi in decomp.phi_input.items():
        U = phase_shifter(dim, mode, phi) @ U

    # Iterate through the layers of the circuit
    for layer in range(dim):
        # Apply the sMZIs in the layer
        for mode in range(layer % 2, dim - 1, 2):
            delta = decomp.delta[mode, layer]
            sigma = decomp.sigma[mode, layer]
            U = symmetric_mzi(dim, (mode, mode + 1), delta, sigma) @ U

        # Apply the phase shifter at the edges
        if (dim - 1, layer) in decomp.phi_edge:
            phi_bottom = decomp.phi_edge[dim - 1, layer]
            U = phase_shifter(dim, dim - 1, phi_bottom) @ U

    # Apply the phase shifters at the output
    for mode, phi in decomp.phi_output.items():
        U = phase_shifter(dim, mode, phi) @ U

    return U
