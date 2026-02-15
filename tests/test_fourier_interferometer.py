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

import unitary_decomp.fourier_interferometer as fi
from unitary_decomp import matrix_operations as mo

"""Unit tests for the Fourier Interferometer module."""


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_diagonal_invariance_with_x_swap_x(dim: int) -> None:
    """Test the invariance of a diagonal matrix under the X-Swap-X transformation."""

    # Generate a diagonal matrix D
    D = np.diag(np.arange(dim))

    # Create the X-Swap-X matrix of dimension `dim`
    x_swap_x = mo.x_matrix(dim) @ mo.swap_matrix(dim) @ mo.x_matrix(dim)

    # Check that the diagonal matrix D is invariant under the transformation
    assert np.allclose(D, x_swap_x @ D @ x_swap_x)


def test_angle_conversion() -> None:
    """Test the `angle_conversion` function in the Fourier interferometer module."""

    # Define the angles for the test
    theta_1, theta_2 = 0.1234, 1.5678

    # Convert angles to sigma and delta
    sigma = (theta_1 + theta_2) / 2
    delta = (theta_1 - theta_2) / 2

    # Use the angle_conversion function to convert back to theta_1 and theta_2
    theta_1_out, theta_2_out = fi.angle_conversion(delta, sigma)

    assert np.isclose(theta_1_out, theta_1)
    assert np.isclose(theta_2_out, theta_2)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_fourier_decomposition_diagonality(dim: int) -> None:
    """Test that the phase masks are diagonal matrices."""
    U = unitary_group(dim, seed=42).rvs()
    decomp = fi.fourier_decomposition(U)
    for mask in decomp.mask_sequence:
        assert mask.shape == (dim,)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_compact_fourier_decomposition_diagonality(dim: int) -> None:
    """Test that the phase masks in the compact decomposition are diagonal matrices."""
    U = unitary_group(dim, seed=42).rvs()
    decomp = fi.compact_fourier_decomposition(U)
    for mask in decomp.mask_sequence:
        assert mask.shape == (dim,)


@pytest.mark.parametrize(
    "U",
    list(unitary_group(12, 42).rvs(5))
    + list(unitary_group(4, 42).rvs(5))
    + list(unitary_group(26, 42).rvs(5))
    + list(unitary_group(46, 42).rvs(5)),
)
def test_fourier_decomposition(U: np.ndarray) -> None:
    """Test the `fourier_decomposition` function for random unitary matrices."""
    decomp = fi.fourier_decomposition(U)
    reconstructed_matrix = fi.circuit_reconstruction(decomp)

    # Check if the reconstructed matrix is close to the original unitary matrix
    assert np.allclose(U, reconstructed_matrix)

    # Check the number of layers in the decomposition
    assert len(decomp.mask_sequence) == 4 * len(decomp.D)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_fourier_decomposition_identity(dim: int) -> None:
    """Test the `fourier_decomposition` function for the identity matrix."""
    U = np.eye(dim)
    decomp = fi.fourier_decomposition(U)
    reconstructed_matrix = fi.circuit_reconstruction(decomp)
    assert np.allclose(U, reconstructed_matrix)
    assert len(decomp.mask_sequence) == 4 * len(decomp.D)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_fourier_decomposition_anti_identity(dim: int) -> None:
    """Test the `fourier_decomposition` function for the anti-identity matrix."""
    U = np.identity(dim)[::-1]
    decomp = fi.fourier_decomposition(U)
    reconstructed_matrix = fi.circuit_reconstruction(decomp)
    assert np.allclose(U, reconstructed_matrix)
    assert len(decomp.mask_sequence) == 4 * len(decomp.D)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_fourier_decomposition_dft(dim: int) -> None:
    """Test the `fourier_decomposition` function for the DFT matrix."""
    U = dft(dim, scale="sqrtn")
    decomp = fi.fourier_decomposition(U)
    reconstructed_matrix = fi.circuit_reconstruction(decomp)
    assert np.allclose(U, reconstructed_matrix)
    assert len(decomp.mask_sequence) == 4 * len(decomp.D)


@pytest.mark.parametrize(
    "U",
    list(unitary_group(12, 42).rvs(5))
    + list(unitary_group(4, 42).rvs(5))
    + list(unitary_group(26, 42).rvs(5))
    + list(unitary_group(46, 42).rvs(5)),
)
def test_compact_fourier_decomposition(U: np.ndarray) -> None:
    """Test the `compact_fourier_decomposition` function for random unitary matrices."""
    decomp = fi.compact_fourier_decomposition(U)
    reconstructed_matrix = fi.circuit_reconstruction(decomp)

    # Check if the reconstructed matrix is close to the original unitary matrix
    assert np.allclose(U, reconstructed_matrix)

    # Check the number of layers in the decomposition
    assert len(decomp.mask_sequence) == 2 * len(decomp.D) + 4


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_compact_fourier_decomposition_identity(dim: int) -> None:
    """Test the `compact_fourier_decomposition` function for the identity matrix."""
    U = np.eye(dim)
    decomp = fi.compact_fourier_decomposition(U)
    reconstructed_matrix = fi.circuit_reconstruction(decomp)
    assert np.allclose(U, reconstructed_matrix)
    assert len(decomp.mask_sequence) == 2 * len(decomp.D) + 4


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_compact_fourier_decomposition_anti_identity(dim: int) -> None:
    """Test the `compact_fourier_decomposition` function for the anti-identity matrix."""
    U = np.identity(dim)[::-1]
    decomp = fi.compact_fourier_decomposition(U)
    reconstructed_matrix = fi.circuit_reconstruction(decomp)
    assert np.allclose(U, reconstructed_matrix)
    assert len(decomp.mask_sequence) == 2 * len(decomp.D) + 4


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_compact_fourier_decomposition_dft(dim: int) -> None:
    """Test the `compact_fourier_decomposition` function for the DFT matrix."""
    U = dft(dim, scale="sqrtn")
    decomp = fi.compact_fourier_decomposition(U)
    reconstructed_matrix = fi.circuit_reconstruction(decomp)
    assert np.allclose(U, reconstructed_matrix)
    assert len(decomp.mask_sequence) == 2 * len(decomp.D) + 4


def test_fourier_decomposition_errors() -> None:
    """Test the `fourier_decomposition` function with invalid inputs."""

    # Test with a non-numpy array input
    with pytest.raises(TypeError, match="Input must be a numpy array."):
        fi.fourier_decomposition("not a numpy array")

    # Test with a non-square matrix
    with pytest.raises(ValueError, match="Input must be a square matrix of even dimension."):
        fi.fourier_decomposition(np.array([[1, 0], [0, 1], [1, 0], [0, 1]]))

    # Test with an odd-dimensional matrix
    with pytest.raises(ValueError, match="Input must be a square matrix of even dimension."):
        fi.fourier_decomposition(np.eye(3))

    # Test with a non-unitary matrix
    with pytest.raises(ValueError, match="Input must be a unitary matrix."):
        fi.fourier_decomposition(np.array([[1, 0], [0, 2]]))


def test_compact_fourier_decomposition_errors() -> None:
    """Test the `compact_fourier_decomposition` function with invalid inputs."""

    # Test with a non-numpy array input
    with pytest.raises(TypeError, match="Input must be a numpy array."):
        fi.compact_fourier_decomposition("not a numpy array")

    # Test with a non-square matrix
    with pytest.raises(ValueError, match="Input must be a square matrix of even dimension."):
        fi.compact_fourier_decomposition(np.array([[1, 0], [0, 1], [1, 0], [0, 1]]))

    # Test with an odd-dimensional matrix
    with pytest.raises(ValueError, match="Input must be a square matrix of even dimension."):
        fi.compact_fourier_decomposition(np.eye(3))

    # Test with a non-unitary matrix
    with pytest.raises(ValueError, match="Input must be a unitary matrix."):
        fi.compact_fourier_decomposition(np.array([[1, 0], [0, 2]]))


def test_circuit_reconstruction_errors() -> None:
    """Test the `circuit_reconstruction` function with invalid inputs."""

    # Test with a non-FourierDecomp input
    with pytest.raises(TypeError, match="Input must be a FourierDecomp object."):
        fi.circuit_reconstruction("not a FourierDecomp object")
