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

import unitary_decomp.optimization.jax_optimizer as jax_opt
from unitary_decomp.fourier_interferometer import FourierDecomp, circuit_reconstruction

np.random.seed(42)

"""Unit tests for the JAX optimizer module."""


@pytest.mark.parametrize("dim", range(2, 10, 2))
def test_forward_product(dim: int) -> None:
    """Test the forward product function."""

    # Generate a random mask sequence
    angles = np.stack([np.random.uniform(0, 2 * np.pi, dim) for _ in range(dim + 1)])
    masks = np.exp(1j * angles)

    # Set up the mixing layer as a DFT matrix
    F = dft(dim, scale="sqrtn")

    # Compare the forward product with the expected result
    circuit = FourierDecomp(masks[0], masks[1:])
    assert np.allclose(jax_opt.forward_product(masks, F), circuit_reconstruction(circuit))

    F = unitary_group(dim, 42).rvs()
    assert np.allclose(
        jax_opt.forward_product(masks, F),
        masks[0][:, None] * np.linalg.multi_dot([F * mask for mask in masks[1:]]),
    )


@pytest.mark.parametrize("dim", range(2, 10, 2))
def test_infidelity_loss_function(dim: int) -> None:
    """Test the infidelity loss function."""

    # Generate a random mask sequence
    angles = np.stack([np.random.uniform(0, 2 * np.pi, dim) for _ in range(dim + 1)])
    masks = np.exp(1j * angles)

    # Set up the mixing layer as a DFT matrix
    F = dft(dim, scale="sqrtn")

    # Place phase shifters everywhere
    ps_indices = np.arange(dim)

    # Compute the forward product
    U = jax_opt.forward_product(masks, F)

    # The infidelity should be zero when the target unitary is the same as the computed one
    infidelity = jax_opt.infidelity_loss_function(angles, F, U, ps_indices)
    assert np.isclose(infidelity, 0.0, rtol=1e-12)

    # The infidelity should be between 0 and 1 for a different target unitary
    infidelity = jax_opt.infidelity_loss_function(angles, F, np.eye(dim), ps_indices)
    assert (infidelity > 0) and (infidelity <= 1.0)


@pytest.mark.parametrize("dim", [2, 5, 8])
def test_jax_mask_optimizer_default(dim: int):
    """Test the JAX mask optimizer with default parameters for random unitary matrices."""

    # Generate a random unitary matrix
    U = unitary_group(dim, 42).rvs()

    # Run the optimizer for default parameters (DFT mixing layer and full phase masks)
    decomp, infidelity = jax_opt.jax_mask_optimizer(U, dim + 1, steps=3000, restarts=25)

    # Check that the infidelity is within a reasonable range
    assert infidelity < 1e-7


def test_jax_mask_optimizer_custom_mixing_layer():
    """Test the JAX mask optimizer with a custom mixing layer."""

    # Set the dimension for the unitary matrix
    dim = 8

    # Generate a random unitary matrix
    U = unitary_group(dim, 42).rvs()

    # Create a custom mixing layer (e.g., a random unitary matrix)
    mixing_layer = unitary_group(dim, 43).rvs()

    # Run the optimizer with the custom mixing layer
    decomp, infidelity = jax_opt.jax_mask_optimizer(
        U, dim + 1, mixing_layer=mixing_layer, steps=3000, restarts=25
    )

    # Check that the infidelity is within a reasonable range
    assert infidelity < 1e-7

    mask_sequence = np.vstack((decomp.D, np.stack(decomp.mask_sequence)))
    U_reconstructed = jax_opt.forward_product(mask_sequence, mixing_layer)

    fidelity = np.abs(np.trace(U.conj().T @ U_reconstructed) / dim) ** 2
    assert np.isclose(infidelity, 1 - fidelity)


def test_jax_mask_optimizer_custom_phase_masks():
    """Test the JAX mask optimizer with custom phase masks."""

    # Set the dimension for the unitary matrix
    dim = 8

    # Generate a random unitary matrix
    U = unitary_group(dim, 42).rvs()

    # Set the DFT as the mixing layer
    mixing_layer = dft(dim, scale="sqrtn")

    # Create custom phase masks (e.g., some phase shifters set to zero)
    zero_indices = np.array([2, 6])
    mask_shape = np.array([True] * dim)
    mask_shape[zero_indices] = False

    # Run the optimizer with the custom phase masks
    decomp, infidelity = jax_opt.jax_mask_optimizer(
        U, 11, mask_shape=mask_shape, steps=3000, restarts=25
    )

    # Check that the infidelity is within a reasonable range
    assert infidelity < 1e-7

    # Check that the returned infidelity is correct.
    mask_sequence = np.vstack((decomp.D, np.stack(decomp.mask_sequence)))
    U_reconstructed = jax_opt.forward_product(mask_sequence, mixing_layer)
    fidelity = np.abs(np.trace(U.conj().T @ U_reconstructed) / dim) ** 2
    assert np.isclose(infidelity, 1 - fidelity)

    # Check that the phase masks are correctly set to 1 at the specified indices
    for i in zero_indices:
        assert np.allclose(mask_sequence[:, i], np.ones(11))


@pytest.mark.parametrize("dim", [2, 5, 8])
def test_scipy_mask_optimizer_default(dim: int):
    """Test the scipy mask optimizer with default parameters for random unitary matrices."""

    # Generate a random unitary matrix
    U = unitary_group(dim, 42).rvs()

    # Run the optimizer for default parameters (DFT mixing layer and full phase masks)
    decomp, infidelity = jax_opt.scipy_mask_optimizer(U, dim + 1, steps=1000, restarts=25)

    # Check that the infidelity is within a reasonable range
    assert infidelity < 1e-7


def test_scipy_mask_optimizer_custom_mixing_layer():
    """Test the scipy mask optimizer with a custom mixing layer."""

    # Set the dimension for the unitary matrix
    dim = 8

    # Generate a random unitary matrix
    U = unitary_group(dim, 42).rvs()

    # Create a custom mixing layer (e.g., a random unitary matrix)
    mixing_layer = unitary_group(dim, 43).rvs()

    # Run the optimizer with the custom mixing layer
    decomp, infidelity = jax_opt.scipy_mask_optimizer(
        U, dim + 1, mixing_layer=mixing_layer, steps=1000, restarts=25
    )

    # Check that the infidelity is within a reasonable range
    assert infidelity < 1e-7

    mask_sequence = np.vstack((decomp.D, np.stack(decomp.mask_sequence)))
    U_reconstructed = jax_opt.forward_product(mask_sequence, mixing_layer)

    fidelity = np.abs(np.trace(U.conj().T @ U_reconstructed) / dim) ** 2
    assert np.isclose(infidelity, 1 - fidelity)


def test_scipy_mask_optimizer_custom_phase_masks():
    """Test the scipy mask optimizer with custom phase masks."""

    # Set the dimension for the unitary matrix
    dim = 8

    # Generate a random unitary matrix
    U = unitary_group(dim, 42).rvs()

    # Set the DFT as the mixing layer
    mixing_layer = dft(dim, scale="sqrtn")

    # Create custom phase masks (e.g., some phase shifters set to zero)
    zero_indices = np.array([2, 6])
    mask_shape = np.array([True] * dim)
    mask_shape[zero_indices] = False

    # Run the optimizer with the custom phase masks
    decomp, infidelity = jax_opt.scipy_mask_optimizer(
        U, 11, mask_shape=mask_shape, steps=1000, restarts=25
    )

    # Check that the infidelity is within a reasonable range
    assert infidelity < 1e-7

    # Check that the returned infidelity is correct.
    mask_sequence = np.vstack((decomp.D, np.stack(decomp.mask_sequence)))
    U_reconstructed = jax_opt.forward_product(mask_sequence, mixing_layer)
    fidelity = np.abs(np.trace(U.conj().T @ U_reconstructed) / dim) ** 2
    assert np.isclose(infidelity, 1 - fidelity)

    # Check that the phase masks are correctly set to 1 at the specified indices
    for i in zero_indices:
        assert np.allclose(mask_sequence[:, i], np.ones(11))
