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

import numpy as np
import pytest
from scipy.linalg import dft
from scipy.stats import unitary_group

from unitary_decomp import bell_interferometer as bi

"""Unit tests for the Bell Interferometer module."""


@pytest.mark.parametrize(
    "theta1, theta2",
    [
        (0, 0),
        (np.pi / 4, np.pi / 4),
        (np.pi / 2, np.pi / 3),
        (np.pi, np.pi / 2),
        (np.pi / 6, np.pi / 6),
        (0.5413, 1.297),
        (1.234, 0.987),
        (2.345, 1.678),
    ],
)
def test_symmetric_mzi_factorisation(theta1: float, theta2: float) -> None:
    """Test the factorisation of the symmetric Mach-Zehnder interferometer."""

    # Calculate delta and sigma from the given theta1 and theta2
    delta = (theta1 - theta2) / 2
    sigma = (theta1 + theta2) / 2

    # Definition of the 50:50 beam splitter matrix X
    X = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

    # Compute the factorisation of the symmetric MZI
    factorisation = X @ np.diag([np.exp(1j * theta1), np.exp(1j * theta2)]) @ X

    assert np.allclose(factorisation, bi.symmetric_mzi(2, (0, 1), delta, sigma))


def test_symmetric_mzi_value_error() -> None:
    """Test the raise of ValueErrors if the dimension is smaller than 2 or the modes are not consecutive."""

    # Test for dimension less than 2
    with pytest.raises(ValueError, match="Dimension must be at least 2"):
        bi.symmetric_mzi(1, (0, 1), 0, 0)

    # Test for non-consecutive modes
    with pytest.raises(ValueError, match="m and n must be consecutive integers"):
        bi.symmetric_mzi(3, (0, 2), 0, 0)


@pytest.mark.parametrize("dim", [2, 7, 10, 31])
def test_phase_shifter(dim: int) -> None:
    """Test the phase_shifter function for a given dimension."""

    # Create a phase shifter for a given dimension
    angle = 0.1234
    target = 5 % dim
    phase_shift = bi.phase_shifter(dim, target, angle)

    # Check the shape of the phase shift matrix
    assert phase_shift.shape == (dim, dim)

    # Check that the diagonal element of the target has the correct phase shift
    assert phase_shift[target, target] == np.exp(1j * angle)


def test_bell_decomposition_type_error() -> None:
    """Test the raise of TypeError if input is not a numpy array."""
    with pytest.raises(TypeError, match="Input must be a numpy array"):
        bi.bell_decomposition([[1, 0], [0, 1]])  # Non-numpy array input


def test_bell_decomposition_value_error() -> None:
    """Test the raise of ValueError if input is not square and unitary."""
    with pytest.raises(ValueError, match="The input matrix must be unitary"):
        bi.bell_decomposition(np.array([[1, 0], [0, 2]]))  # Non-unitary matrix

    with pytest.raises(ValueError, match="Input matrix must be square"):
        bi.bell_decomposition(np.array([[1, 0], [0, 1], [1, 0]]))  # Non-square matrix


@pytest.mark.parametrize(
    "U",
    list(unitary_group(4, seed=7).rvs(5))
    + list(unitary_group(17, seed=7).rvs(5))
    + list(unitary_group(42, seed=7).rvs(5))
    + list(unitary_group(35, seed=7).rvs(5)),
)
def test_bell_decomposition(U: np.ndarray) -> None:
    """Test the Bell decomposition with random unitary matrices."""
    decomp = bi.bell_decomposition(U)
    reconstructed_matrix = bi.circuit_reconstruction(decomp)

    # Check if the reconstructed matrix is close to the original unitary matrix
    assert np.allclose(U, reconstructed_matrix)


@pytest.mark.parametrize("dim", range(2, 20))
def test_bell_decomposition_identity(dim: int) -> None:
    """Test the Bell decomposition for the identity matrix."""
    U = np.eye(dim)
    decomp = bi.bell_decomposition(U)
    reconstructed_matrix = bi.circuit_reconstruction(decomp)

    # Check if the reconstructed matrix is close to the identity matrix
    assert np.allclose(U, reconstructed_matrix)


@pytest.mark.parametrize("dim", range(2, 20))
def test_bell_decomposition_anti_identity(dim: int) -> None:
    """Test the Bell decomposition for the anti-identity matrix."""
    U = np.identity(dim)[::-1]
    decomp = bi.bell_decomposition(U)
    reconstructed_matrix = bi.circuit_reconstruction(decomp)

    # Check if the reconstructed matrix is close to the anti-identity matrix
    assert np.allclose(U, reconstructed_matrix)


@pytest.mark.parametrize("dim", range(2, 20))
def test_bell_decomposition_dft(dim: int) -> None:
    """Test the Bell decomposition for the DFT matrix."""
    U = dft(dim, scale="sqrtn")
    decomp = bi.bell_decomposition(U)
    reconstructed_matrix = bi.circuit_reconstruction(decomp)

    # Check if the reconstructed matrix is close to the DFT matrix
    assert np.allclose(U, reconstructed_matrix)


def test_circuit_reconstruction_type_error() -> None:
    """Test the raise of TypeError if input is not a valid decomposition."""

    with pytest.raises(TypeError, match="Input must be a BellDecomp instance"):
        bi.circuit_reconstruction("Wrong type")  # Non-decomposition input
