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
from scipy.stats import unitary_group

from unitary_decomp.fourier_interferometer import circuit_reconstruction
from unitary_decomp.matrix_operations import matrix_interleave
from unitary_decomp.optimization.fourier_optimizer import fidelity, mask_optimizer

"""Unit tests for the `fourier_optimizer` module."""


@pytest.mark.parametrize("dim", [3, 6, 9, 12])
def test_fidelity(dim: int) -> None:
    """Test the fidelity function."""

    # Generate two random unitary matrices
    U = unitary_group(dim=dim, seed=42).rvs()
    V = unitary_group(dim=dim, seed=137).rvs()

    # Fidelity should be between 0 and 1
    assert 0 <= fidelity(U, V) <= 1

    # Fidelity of a matrix with itself should be 1
    assert np.isclose(fidelity(U, U), 1.0)
    assert np.isclose(fidelity(V, V), 1.0)


def test_fidelity_shape_error() -> None:
    """Test the raise of ValueError when shapes of matrices do not match."""

    # Generate two random unitary matrices of different dimensions
    U = unitary_group(dim=4, seed=42).rvs()
    V = unitary_group(dim=3, seed=137).rvs()

    with pytest.raises(ValueError, match="Both matrices must have the same shape."):
        fidelity(U, V)


def test_mask_optimizer_type_errors() -> None:
    """Test the raise of a TypeError when input are of the wrong type."""

    # U is not a numpy array
    with pytest.raises(TypeError, match="U must be a numpy array."):
        mask_optimizer(U=[[1, 0], [0, 1]], length=3)

    # mixing_layer is not a numpy array or None
    with pytest.raises(TypeError, match="The mixing layer must be a numpy array or None."):
        mask_optimizer(U=np.eye(2), length=3, mixing_layer=[[1, 0], [0, 1]])

    # mask_shape is not a numpy array or None
    with pytest.raises(TypeError, match="mask_shape must be a numpy array or None."):
        mask_optimizer(U=np.eye(4), length=3, mask_shape=[[True, False, False, True]])


def test_mask_optimizer_value_errors() -> None:
    """Test the raise of a ValueError when the target matrix is not unitary."""

    # U is not unitary
    U = np.array([[1, 0], [0, 2]])
    with pytest.raises(ValueError, match="U must be unitary"):
        mask_optimizer(U=U, length=3)


def test_mask_optimizer_random_unitary() -> None:
    """Test the mask_optimizer with a random unitary matrix and the default mask shape."""

    # Generate a random 5x5 unitary matrix
    U = unitary_group(dim=5, seed=42).rvs()

    # Optimize the mask sequence with N+2 = 7 phase masks (over parametrization, the optimizer will converge)
    decomposition = mask_optimizer(U, length=7, n_iters=100, n_runs=1, display=False)[0]
    U_reconstructed = circuit_reconstruction(decomposition)
    assert np.allclose(U, U_reconstructed, rtol=1e-2)

    # Optimize the mask sequence with N-1 = 4 phase masks (under-parametrization, the optimizer won't converge)
    decomposition = mask_optimizer(U, length=4, n_iters=100, n_runs=1, display=False)[0]
    U_reconstructed = circuit_reconstruction(decomposition)
    assert not np.allclose(U, U_reconstructed, rtol=1e-2)


def test_mask_optimizer_random_unitary_with_mask_shape() -> None:
    """Test the mask_optimizer with a random unitary matrix and a specific mask shape."""

    # Generate a random 5x5 unitary matrix
    U = unitary_group(dim=5, seed=99).rvs()  # Generate a random 5x5 unitary matrix

    # Define a specific mask shape
    mask_shape = np.array([True, False, True, False, True])

    # Optimize the mask sequence with 9 phase masks (complete parametrization, the optimizer will converge)
    decomposition = mask_optimizer(U, length=9, mask_shape=mask_shape, n_iters=100, n_runs=1)[0]
    U_reconstructed = circuit_reconstruction(decomposition)
    assert np.allclose(U, U_reconstructed, rtol=1e-2)

    # Define a different mask shape
    mask_shape = np.array(
        [True, False, True, False, False]
    )  # Change mask shape to under-parametrize

    # Optimize the mask sequence with 14 phase masks (complete parametrization, the optimizer will converge)
    decomposition = mask_optimizer(U, length=14, mask_shape=mask_shape, n_iters=100, n_runs=1)[0]
    U_reconstructed = circuit_reconstruction(decomposition)
    assert np.allclose(U, U_reconstructed, rtol=1e-2)

    # Optimize the mask sequence with 10 phase masks (under-parametrization, the optimizer won't converge)
    decomposition = mask_optimizer(U, length=10, mask_shape=mask_shape, n_iters=100, n_runs=1)[0]
    U_reconstructed = circuit_reconstruction(decomposition)
    assert not np.allclose(U, U_reconstructed, rtol=1e-2)


def test_mask_optimizer_random_unitary_with_mixing_layer() -> None:
    """Test the mask_optimizer with a random unitary matrix and a custom mixing layer."""

    # Generate a random 5x5 unitary matrix
    U = unitary_group(dim=5, seed=42).rvs()

    # Generate a random dense mixing layer
    mixing_layer = unitary_group(dim=5, seed=43).rvs()

    # Optimize the mask sequence with 7 phase masks (over parametrization, the optimizer will converge)
    decomposition = mask_optimizer(
        U, length=7, mixing_layer=mixing_layer, n_iters=100, n_runs=1, display=False
    )[0]
    U_reconstructed = decomposition.D[:, None] * matrix_interleave(
        decomposition.mask_sequence, mixing_layer=mixing_layer
    )
    assert np.allclose(U, U_reconstructed, rtol=1e-2)

    # Optimize the mask sequence with 4 phase masks (under-parametrization, the optimizer won't converge)
    decomposition = mask_optimizer(
        U, length=4, mixing_layer=mixing_layer, n_iters=100, n_runs=1, display=False
    )[0]
    U_reconstructed = decomposition.D[:, None] * matrix_interleave(
        decomposition.mask_sequence, mixing_layer=mixing_layer
    )
    assert not np.allclose(U, U_reconstructed, rtol=1e-2)
