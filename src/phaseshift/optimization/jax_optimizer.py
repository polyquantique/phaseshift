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
# JAX Optimizer

This module provides functions to optimize a sequence of phase masks to approximate a target unitary matrix using JAX.
The linear interferometer is modeled as a deep neural network which is trained to learn a target unitary matrix by
adjusting the phase masks to minimize a cost function. The optimization is performed using the Adam optimizer with multiple
restarts to find the best set of parameters. The module consists of the following main functions:

- `forward_product`: Computes the forward product of the network given a sequence of phase masks and a mixing layer.
- `infidelity_loss_function`: Computes the infidelity loss function between the target unitary and the approximated unitary.
- `geodesic_distance`: Computes the geodesic distance between the target unitary and the approximated unitary [1].
- `adam_run`: Runs the Adam optimizer for a given number of steps to optimize the angles in the phase masks.
- `jax_mask_optimizer`: Main function to optimize the phase masks to approximate a target unitary matrix.
- `scipy_mask_optimizer`: Similar to `jax_mask_optimizer`, but uses the BFGS algorithm from SciPy for optimization.

### References:

[1] Álvarez-Vizoso, Javier, and David Barral. "Universality and Optimal Architectures for Layered Programmable Unitary Decompositions." arXiv preprint arXiv:2510.19397 (2025).
"""

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import optax
from jax.scipy.optimize import minimize
from scipy.linalg import dft

from phaseshift.fourier_interferometer import FourierDecomp

# Enable 64-bit precision for JAX operations
jax.config.update("jax_enable_x64", True)


def forward_product(mask_sequence: jax.Array, mixing_layer: jax.Array) -> jax.Array:
    """Compute the forward product of the network.

    Given a sequence of phase masks and a mixing layer, this function computes the product of the phase masks
    and mixing layers in an alternating fashion. This function serves as the forward pass of the
    network.

    Args:
        mask_sequence (jax.Array): (L, N) array of complex phases, where L is the number of phase masks
            in the network and N is their dimension.
        mixing_layer (jax.Array): (N, N) array representing the mixing layer.

    Returns:
        jax.Array: (N, N) array representing the product of the phase masks and mixing layers.
    """

    def _single_product(carry, mask):
        """Compute single product step for scan."""
        return carry @ (mixing_layer * mask), None

    # Compute the forward product
    product, _ = jax.lax.scan(
        _single_product, init=jnp.diag(mask_sequence[0]), xs=mask_sequence[1:]
    )
    return product


def infidelity_loss_function(
    angles: jax.Array, mixing_layer: jax.Array, U: jax.Array, ps_indices: jax.Array
) -> jax.Array:
    """Compute the infidelity loss function.

    Given a set of angles and a mixing layer, this function computes the infidelity between the
    target unitary `U` and the unitary obtained from the forward product of the network corresponding to the angles.

    Args:
        angles (jax.Array): (L, N) array of angles, where L is the number of phase masks in the network and N is the dimension of the unitary.
        mixing_layer (jax.Array): (N, N) array representing the mixing layer.
        U (jax.Array): (N, N) array representing the target unitary matrix.
        ps_indices (jax.Array): Array of indices where phase shifters are located in the phase masks. If there are phase shifters on all N channels, `ps_indices = jnp.arange(N)`.

    Returns:
        float: Infidelity between the target unitary and the approximated unitary. (Scalar value between 0 and 1)
    """
    # Dimension of the unitary
    dim = U.shape[0]

    # Construct the phase masks from the angles and the mask shape
    angles = angles.reshape(-1, ps_indices.size)
    mask_sequence = jnp.ones((angles.shape[0], dim), dtype=jnp.complex128)
    mask_sequence = mask_sequence.at[:, ps_indices].set(jnp.exp(1.0j * angles))

    # Compute the forward product for the given mask sequence
    U_approx = forward_product(mask_sequence, mixing_layer)

    # Compute the fidelity between the target unitary and the approximated unitary
    fidelity = jnp.abs(jnp.trace(U.conj().T @ U_approx) / dim) ** 2

    return 1 - fidelity


def geodesic_distance(
    angles: jax.Array, mixing_layer: jax.Array, U: jax.Array, ps_indices: jax.Array
) -> jax.Array:
    """Compute the geodesic distance between two unitary matrices.

    Given a set of angles and a mixing layer, this function computes the geodesic distance between the
    target unitary `U` and the unitary obtained from the forward product of the network corresponding to the angles.

    The geodesic distance is defined in [1]. It is a measure of the true distance in the curved space of SU(N).

    Args:
        angles (jax.Array): (L, N) array of angles, where L is the number of phase masks in the network and N is the dimension of the unitary.
        mixing_layer (jax.Array): (N, N) array representing the mixing layer.
        U (jax.Array): (N, N) array representing the target unitary matrix.
        ps_indices (jax.Array): Array of indices where phase shifters are located in the phase masks. If there are phase shifters on all N channels, `ps_indices = jnp.arange(N)`.

    Returns:
        float: Geodesic distance between the target unitary and the approximated unitary.

    ### References:
        [1] Álvarez-Vizoso, Javier, and David Barral. "Universality and Optimal Architectures for Layered Programmable Unitary Decompositions." arXiv preprint arXiv:2510.19397 (2025).
    """
    # Dimension of the unitary
    dim = U.shape[0]

    # Construct the phase masks from the angles and the mask shape
    angles = angles.reshape(-1, ps_indices.size)
    mask_sequence = jnp.ones((angles.shape[0], dim), dtype=jnp.complex128)
    mask_sequence = mask_sequence.at[:, ps_indices].set(jnp.exp(1.0j * angles))

    # Compute the forward product for the given mask sequence
    U_approx = forward_product(mask_sequence, mixing_layer)

    # Compute the geodesic distance between the target unitary and the approximated unitary
    omega = U_approx.conj().T @ U
    geodesic_distance = jnp.sum(jnp.angle(jnp.linalg.eigvals(omega)) ** 2)

    return geodesic_distance


@partial(jax.jit, static_argnames=["steps", "lr", "cost_function"])
def adam_run(
    init_angles: jax.Array,
    mixing_layer: jax.Array,
    U: jax.Array,
    ps_indices: jax.Array,
    steps: int,
    lr: float,
    cost_function: Callable[..., jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Run the Adam optimizer for a given number of steps.

    Given initial angles, a mixing layer, and a target unitary matrix, this function runs the Adam optimizer
    for a specified number of steps to optimize the angles such that the cost function is minimized.

    Args:
        init_angles (jax.Array): Initial angles in the phase masks.
        mixing_layer (jax.Array): Mixing layer to place between phase masks.
        U (jax.Array): Target unitary matrix to approximate.
        ps_indices (jax.Array): Locations of the phase shifters in the phase masks.
        steps (int): Number of optimization steps to perform.
        lr (float): Learning rate for the optimizer.
        cost_function (Callable): Cost function to be minimized.

    Returns:
        tuple: Optimized parameters and cost function after optimization.
    """
    # Initialize parameters and optimizer
    angles = init_angles
    opt = optax.adam(lr)
    opt_state = opt.init(angles)
    best_config = (angles, jnp.inf)

    def _opt_step(state, _):
        """Single optimization step."""
        angles, opt_state, (best_angle, best_cost) = state

        # Compute the gradient of the cost function with respect to the angles
        cost, grads = jax.value_and_grad(cost_function, argnums=0)(
            angles, mixing_layer, U, ps_indices
        )

        # Check if the new cost is better than the best found so far
        is_better = (cost < best_cost) & (cost > 0.0)
        best_cost = jnp.where(is_better, cost, best_cost)
        best_angle = jnp.where(is_better, angles, best_angle)

        # Update the parameters using the optimizer
        updates, opt_state = opt.update(grads, opt_state, angles)
        angles = optax.apply_updates(angles, updates)

        return (angles, opt_state, (best_angle, best_cost)), None

    # Run the optimization for the specified number of steps
    (_, _, (final_angles, cost)), _ = jax.lax.scan(
        _opt_step, (angles, opt_state, best_config), None, length=steps
    )

    return final_angles, cost


def jax_mask_optimizer(
    U: jax.Array,
    length: int,
    mixing_layer: Optional[jax.Array] = None,
    mask_shape: Optional[jax.Array] = None,
    steps: int = 3000,
    restarts: int = 200,
    learning_rate: float = 1e-2,
    cost_function: str = "infidelity",
) -> tuple[FourierDecomp, float]:
    """Optimize the phase masks to approximate a target unitary matrix.

    Given a target unitary matrix `U`, this function optimizes the sequence of phase masks {Di}, i = 1,..,L,
    such that their product with an interleaved mixing layer M approximates `U`:

    U ≈ D1 @ M @ D2 @ M @ ... @ DL.

    The optimization is performed using the Adam optimizer with multiple restarts to find the best set of parameters.

    Args:
        U (jax.Array): Target unitary matrix to approximate.
        length (int): Number of phase masks in the circuit.
        mixing_layer (jax.Array, optional): Mixing layer to place between phase masks. If None, the DFT matrix is used.
        mask_shape (jax.Array, optional): Boolean array indicating where phase shifters are located in the phase masks.
            True indicates the presence of a phase shifter, False indicates no phase shifter. If None, a full mask (all True) is used.
        steps (int, optional): Number of optimization steps to perform for each restart. Default is 3000.
        restarts (int, optional): Number of restarts for the optimization. Default is 200.
        learning_rate (float, optional): Learning rate for the Adam optimizer. Default is 1e-2.
        cost_function (str, optional): Cost function to minimize during optimization. Default is the `infidelity`.
            Can be set to `geodesic_distance` for an alternative cost function.

    Returns:
        (FourierDecomp, float): A tuple containing the optimized FourierDecomp object and the corresponding cost.
    """

    # Dimension of the unitary
    dim = U.shape[0]

    # If no mixing layer is provided, use the DFT matrix as the default mixing layer
    if mixing_layer is None:
        mixing_layer = jnp.array(dft(dim, scale="sqrtn"))

    # If no mask shape is provided, use a full mask (all True)
    if mask_shape is None:
        mask_shape = jnp.array([True] * dim)

    if cost_function == "geodesic":
        cost_fun = geodesic_distance
    elif cost_function == "infidelity":
        cost_fun = infidelity_loss_function
    else:
        raise ValueError(
            f"Invalid cost function: {cost_function}. Must be 'infidelity' or 'geodesic'."
        )

    # Extract the indices of the phase shifters from the mask shape
    ps_indices = jnp.nonzero(mask_shape)[0]

    # Initialize random keys for each restarts
    key = jax.random.key(137)
    key, *subkeys = jax.random.split(key, restarts + 1)

    # Define the initial parameters for each restart
    init_params = jnp.stack(
        [
            jax.random.uniform(k, (length, ps_indices.size), minval=0, maxval=2 * jnp.pi)
            for k in subkeys
        ]
    )

    # Run the Adam optimizer for each set of initial parameters
    angles, costs = jax.vmap(
        adam_run,
        in_axes=(0, None, None, None, None, None, None),
    )(
        init_params,
        mixing_layer,
        U,
        ps_indices,
        steps,
        learning_rate,
        cost_fun,
    )

    # Find the best parameters and corresponding cost
    best_index = jnp.argmin(costs)
    mask_sequence = jnp.ones((length, dim), dtype=jnp.complex128)
    mask_sequence = mask_sequence.at[:, ps_indices].set(jnp.exp(1.0j * angles[best_index]))
    best_cost = costs[best_index]

    return FourierDecomp(mask_sequence[0], mask_sequence[1:]), best_cost


def scipy_mask_optimizer(
    U: jax.Array,
    length: int,
    mixing_layer: Optional[jax.Array] = None,
    mask_shape: Optional[jax.Array] = None,
    steps: int = 1000,
    restarts: int = 200,
) -> tuple[FourierDecomp, float]:
    """Optimize the phase masks to approximate a target unitary matrix.

    Given a target unitary matrix `U`, this function optimizes the sequence of phase masks {Di}, i = 1,..,L,
    such that their product with an interleaved mixing layer M approximates `U`:

    U ≈ D1 @ M @ D2 @ M @ ... @ DL.

    The optimization is performed using the quasi-Newton "BFGS" algorithm from scipy implemented in the Jax library.
    Multiple restarts are performed to find a global minimum.

    Args:
        U (jax.Array): Target unitary matrix to approximate.
        length (int): Number of phase masks in the circuit.
        mixing_layer (jax.Array, optional): Mixing layer to place between phase masks. If None, the DFT matrix is used.
        mask_shape (jax.Array, optional): Boolean array indicating where phase shifters are located in the phase masks.
            True indicates the presence of a phase shifter, False indicates no phase shifter. If None, a full mask (all True) is used.
        steps (int): Number of optimization steps to perform for each restart. Default is 1000.
        restarts (int): Number of restarts for the optimization. Default is 200.

    Returns:
        (FourierDecomp, float): A tuple containing the optimized FourierDecomp object and the corresponding infidelity.
    """
    # Dimension of the unitary
    dim = U.shape[0]

    # If no mixing layer is provided, use the DFT matrix as the default mixing layer
    if mixing_layer is None:
        mixing_layer = jnp.array(dft(dim, scale="sqrtn"))

    # If no mask shape is provided, use a full mask (all True)
    if mask_shape is None:
        mask_shape = jnp.array([True] * dim)

    # Extract the indices of the phase shifters from the mask shape
    ps_indices = jnp.nonzero(mask_shape)[0]

    # Initialize random keys for each restarts
    key = jax.random.key(137)
    key, *subkeys = jax.random.split(key, restarts + 1)

    # Define the initial parameters for each restart
    init_params = jnp.stack(
        [
            jax.random.uniform(k, (length * ps_indices.size,), minval=0, maxval=2 * jnp.pi)
            for k in subkeys
        ]
    )

    # Run the scipy optimizer for each set of initial parameters
    scipy_minimizer = lambda init_param: minimize(
        infidelity_loss_function,
        init_param,
        (mixing_layer, U, ps_indices),
        method="BFGS",
        options={"maxiter": steps},
    )
    results = jax.vmap(jax.jit(scipy_minimizer))(init_params)

    # Find the best parameters and corresponding infidelity
    best_index = jnp.argmin(results.fun)
    best_infidelity = results.fun[best_index]
    best_angles = results.x[best_index].reshape(length, ps_indices.size)
    best_sequence = jnp.ones((length, dim), dtype=jnp.complex128)
    best_sequence = best_sequence.at[:, ps_indices].set(jnp.exp(1.0j * best_angles))

    return FourierDecomp(best_sequence[0], best_sequence[1:]), best_infidelity
