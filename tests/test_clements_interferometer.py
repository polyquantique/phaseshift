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
from scipy.fft import fft
from scipy.stats import unitary_group

from unitary_decomp import clements_interferometer as ci

"""Unit tests for the Clements interferometer module."""

# Constants
SQRT2 = np.sqrt(2)
X = 1 / SQRT2 * np.array([[1, 1], [1, -1]])  # 50:50 beam splitter


@pytest.mark.parametrize("dim", [5, 10, 22, 51])
@pytest.mark.parametrize("theta", [0, np.pi / 4, np.pi / 2, 0.21416546542, 1.234])
@pytest.mark.parametrize("phi", [0, np.pi / 4, 3 * np.pi / 2, 4.21416546542, 1.234])
def test_beam_splitter(dim: int, theta: float, phi: float) -> None:
    """Test the beam splitter function."""

    # 2 x 2 interaction block
    block = np.array(
        [
            [np.exp(1j * phi) * np.cos(theta), -np.sin(theta)],
            [np.exp(1j * phi) * np.sin(theta), np.cos(theta)],
        ]
    )

    matrix = ci.beam_splitter(dim, (dim // 2, dim // 2 + 1), theta, phi)

    assert matrix.shape == (dim, dim)  # Check shape
    assert np.allclose(matrix @ matrix.T.conj(), np.eye(dim))  # Check unitarity
    assert np.allclose(
        matrix[dim // 2 : dim // 2 + 2, dim // 2 : dim // 2 + 2], block
    )  # Check block structure
    assert matrix[dim // 2, dim // 2 + 2] == 0  # Check zero elements


def test_beam_splitter_invalid_target() -> None:
    """Test the beam splitter function with invalid target indices."""

    # Non-consecutive target indices
    with pytest.raises(ValueError, match="m and n must be consecutive integers."):
        ci.beam_splitter(5, (0, 5), 0.5, 0)


@pytest.mark.parametrize("dim", [5, 10, 22, 51])
@pytest.mark.parametrize("theta", [0, np.pi / 4, np.pi / 2, 0.21416546542, 1.234])
@pytest.mark.parametrize("phi", [0, np.pi / 4, 3 * np.pi / 2, 4.21416546542, 1.234])
def test_mzi_matrix(dim: int, theta: float, phi: float) -> None:
    """Test the mzi_matrix function."""

    # 2 x 2 interaction block
    block = np.exp(1j * theta) * np.array(
        [
            [np.exp(1j * phi) * np.cos(theta), 1.0j * np.sin(theta)],
            [np.exp(1j * phi) * 1.0j * np.sin(theta), np.cos(theta)],
        ]
    )

    matrix = ci.mzi_matrix(dim, (dim // 2, dim // 2 + 1), theta, phi)

    assert matrix.shape == (dim, dim)  # Check shape
    assert np.allclose(matrix @ matrix.T.conj(), np.eye(dim))  # Check unitarity
    assert np.allclose(
        matrix[dim // 2 : dim // 2 + 2, dim // 2 : dim // 2 + 2], block
    )  # Check block structure
    assert matrix[dim // 2, dim // 2 + 2] == 0  # Check zero elements


def test_mzi_matrix_invalid_target() -> None:
    """Test the mzi_matrix function with invalid target indices."""

    # Non-consecutive target indices
    with pytest.raises(ValueError, match="m and n must be consecutive integers."):
        ci.mzi_matrix(5, (0, 5), 0.5, 0)


@pytest.mark.parametrize("theta", [0, np.pi / 4, np.pi / 2, 0.21416546542, 1.234])
@pytest.mark.parametrize("phi", [0, np.pi / 4, 3 * np.pi / 2, 4.21416546542, 1.234])
def test_mzi_matrix_factorization(theta: float, phi: float) -> None:
    """Test the factorization of the MZI matrix. MZI = X @ P(2 * theta) @ X @ P(phi)."""

    mzi = ci.mzi_matrix(2, (0, 1), theta, phi)

    # Phase shift matrix
    phase_shift = lambda phi: np.array([[np.exp(1j * phi), 0], [0, 1]])

    # Factorization using X and phase shifts
    factorization = X @ phase_shift(2 * theta) @ X @ phase_shift(phi)

    assert np.allclose(mzi, factorization)


@pytest.mark.parametrize(
    "U",
    list(unitary_group(13, 42).rvs(10))
    + list(unitary_group(26, 42).rvs(10))
    + list(unitary_group(45, 42).rvs(5)),
)
def test_clements_decomposition(U: np.ndarray) -> None:
    """Test the validity of the clements_decomposition function for random matrices."""
    decomp = ci.clements_decomposition(U)
    assert np.allclose(U, ci.circuit_reconstruction(decomp))


def test_clements_decomposition_errors() -> None:
    """Test the clements_decomposition function with invalid inputs."""

    # Test with a non-numpy array input
    with pytest.raises(TypeError, match="U must be a numpy array."):
        ci.clements_decomposition([[1, 0], [0, 1]])

    # Test with a non-unitary matrix
    with pytest.raises(ValueError, match="U must be a unitary matrix."):
        ci.clements_decomposition(np.array([[1, 0], [0, 2]]))


@pytest.mark.parametrize(
    "U",
    list(unitary_group(13, 42).rvs(10))
    + list(unitary_group(26, 42).rvs(10))
    + list(unitary_group(45, 42).rvs(5)),
)
def test_mzi_decomposition(U: np.ndarray) -> None:
    """Test the validity of the mzi_decomposition function for random matrices."""
    decomp = ci.mzi_decomposition(U)
    assert np.allclose(U, ci.mzi_circuit_reconstruction(decomp))


def test_mzi_decomposition_errors() -> None:
    """Test the mzi_decomposition function with invalid inputs."""

    # Non-numpy array input
    with pytest.raises(TypeError, match="U must be a numpy array."):
        ci.mzi_decomposition([[1, 0], [0, 1]])

    # Non-unitary matrix
    with pytest.raises(ValueError, match="U must be a unitary matrix."):
        ci.mzi_decomposition(np.array([[1, 0], [0, 2]]))


@pytest.mark.parametrize(
    "U",
    [
        np.array(
            [
                [1 / SQRT2, 1j / SQRT2, 0],
                [1j / SQRT2, 1 / SQRT2, 0],
                [0, 0, 1],
            ]
        ),
        np.array([[1j, 0, 0], [0, 0, 1], [0, 1, 0]]),
        np.array(
            [
                [1 / SQRT2, 1j / SQRT2, 0, 0],
                [1j / SQRT2, 1 / SQRT2, 0, 0],
                [0, 0, 1 / SQRT2, 1j / SQRT2],
                [0, 0, 1j / SQRT2, 1 / SQRT2],
            ]
        ),
        np.eye(15),
        np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]),
    ],
)
def test_unitary_with_zeros(U: np.ndarray) -> None:
    """Test decompositions on unitary matrices with zeros."""
    decomp_clements = ci.clements_decomposition(U)
    decomp_mzi = ci.mzi_decomposition(U)
    assert np.allclose(U, ci.circuit_reconstruction(decomp_clements))
    assert np.allclose(U, ci.mzi_circuit_reconstruction(decomp_mzi))


@pytest.mark.parametrize("dim", range(1, 11))
def test_identity_decomp(dim: int) -> None:
    """Test decompositions on the identity matrix."""
    U = np.identity(dim)
    decomp_clements = ci.clements_decomposition(U)
    decomp_mzi = ci.mzi_decomposition(U)
    assert np.allclose(U, ci.circuit_reconstruction(decomp_clements))
    assert np.allclose(U, ci.mzi_circuit_reconstruction(decomp_mzi))


@pytest.mark.parametrize("dim", range(1, 11))
def test_anti_identity_decomp(dim: int) -> None:
    """Test decompositions on the anti-identity matrix."""
    U = np.identity(dim)[::-1]
    decomp_clements = ci.clements_decomposition(U)
    decomp_mzi = ci.mzi_decomposition(U)
    assert np.allclose(U, ci.circuit_reconstruction(decomp_clements))
    assert np.allclose(U, ci.mzi_circuit_reconstruction(decomp_mzi))


@pytest.mark.parametrize("dim", range(1, 11))
def test_fourier_decomp(dim: int) -> None:
    """Test decompositions on the Fourier matrix."""
    U = fft(np.identity(dim), norm="ortho")
    decomp_clements = ci.clements_decomposition(U)
    decomp_mzi = ci.mzi_decomposition(U)
    assert np.allclose(U, ci.circuit_reconstruction(decomp_clements))
    assert np.allclose(U, ci.mzi_circuit_reconstruction(decomp_mzi))


@pytest.mark.parametrize("dim", range(1, 11))
def test_permutation_matrix_decomp(dim: int) -> None:
    """Test decompositions on random permutation matrices."""
    U = np.random.permutation(np.identity(dim))
    decomp_clements = ci.clements_decomposition(U)
    decomp_mzi = ci.mzi_decomposition(U)
    assert np.allclose(U, ci.circuit_reconstruction(decomp_clements))
    assert np.allclose(U, ci.mzi_circuit_reconstruction(decomp_mzi))


def test_circuit_reconstruction_errors() -> None:
    """Test the circuit_reconstruction function with invalid inputs."""

    # Wrong input type
    with pytest.raises(TypeError, match="decomposition must be a Decomposition object."):
        ci.circuit_reconstruction("invalid")


def test_mzi_circuit_reconstruction_errors() -> None:
    """Test the mzi_circuit_reconstruction function with invalid inputs."""

    # Wrong input type
    with pytest.raises(TypeError, match="decomposition must be a Decomposition object."):
        ci.mzi_circuit_reconstruction("invalid")


def test_circuit_reconstruction_loss() -> None:
    """Test the circuit_reconstruction function with a loss applied on the beams splitters."""

    dim = 6
    U = unitary_group(dim, 42).rvs()
    decomp = ci.clements_decomposition(U)
    reconstructed_circuit = ci.circuit_reconstruction(decomp, loss_db=1)

    assert isinstance(reconstructed_circuit, np.ndarray)
