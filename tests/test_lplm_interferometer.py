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

import unitary_decomp.lplm_interferometer as li

np.random.seed(42)

"""Unit tests for the LPLM Interferometer module."""


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_lplm_decomposition_diagonality(dim: int) -> None:
    """Test that the phase masks are diagonal matrices."""
    U = unitary_group(dim, seed=42).rvs()
    decomp = li.lplm_decomposition(U)
    for mask in decomp.mask_sequence:
        assert mask.shape == (dim,)


@pytest.mark.parametrize(
    "U",
    list(unitary_group(12, 42).rvs(5))
    + list(unitary_group(4, 42).rvs(5))
    + list(unitary_group(26, 42).rvs(5))
    + list(unitary_group(46, 42).rvs(5)),
)
def test_lplm_decomposition(U: np.ndarray) -> None:
    """Test the validity of the lplm_decomposition function for random matrices."""
    decomp = li.lplm_decomposition(U)
    reconstructed_matrix = li.circuit_reconstruction(decomp)
    assert np.allclose(U, reconstructed_matrix)
    assert len(decomp.mask_sequence) == 6 * len(decomp.D)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_lplm_decomposition_identity(dim: int) -> None:
    """Test the lplm_decomposition function for the identity matrix."""
    U = np.eye(dim)
    decomp = li.lplm_decomposition(U)
    reconstructed_matrix = li.circuit_reconstruction(decomp)
    assert np.allclose(U, reconstructed_matrix)
    assert len(decomp.mask_sequence) == 6 * len(decomp.D)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_lplm_decomposition_anti_identity(dim: int) -> None:
    """Test the lplm_decomposition function for the anti-identity matrix."""
    U = np.identity(dim)[::-1]
    decomp = li.lplm_decomposition(U)
    reconstructed_matrix = li.circuit_reconstruction(decomp)
    assert np.allclose(U, reconstructed_matrix)
    assert len(decomp.mask_sequence) == 6 * len(decomp.D)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_lplm_decomposition_dft(dim: int) -> None:
    """Test the lplm_decomposition function for the DFT matrix."""
    U = dft(dim, scale="sqrtn")
    decomp = li.lplm_decomposition(U)
    reconstructed_matrix = li.circuit_reconstruction(decomp)
    assert np.allclose(U, reconstructed_matrix)
    assert len(decomp.mask_sequence) == 6 * len(decomp.D)


def test_lplm_decomposition_errors() -> None:
    """Test the lplm_decomposition function with invalid inputs."""

    # Test with non-numpy array input
    with pytest.raises(TypeError, match="Input must be a numpy array."):
        li.lplm_decomposition([[1, 0], [0, 1]])  # Non-numpy array input

    # Test with non-unitary matrix
    with pytest.raises(ValueError, match="Input matrix must be unitary."):
        li.lplm_decomposition(np.array([[1, 0], [0, 2]]))  # Non-unitary matrix

    # Test with odd dimension
    with pytest.raises(ValueError, match="Dimension must be even for LPLM decomposition."):
        li.lplm_decomposition(np.eye(3))


def test_lplm_reconstruction_errors() -> None:
    """Test the circuit reconstruction function with invalid inputs."""
    # Test with non-decomposition input
    with pytest.raises(TypeError, match="Input must be a LplmDecomp object."):
        li.circuit_reconstruction("Hello")  # Non-Decomposition input
