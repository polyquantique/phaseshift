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
# Phase Plot

This module provides a function to plot the phase masks from a Fourier or LPLM decomposition
as a 2D heatmap. It visualizes the distribution of phases across the masks in the decomposition.

## Example:

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import unitary_group
    >>> from unitary_decomp.plot.phases_plot import plot_phases
    >>> from unitary_decomp import compact_fourier_decomposition

    >>> # Generate a random unitary matrix
    >>> U = unitary_group(10, seed=42).rvs()

    >>> # Decompose the unitary matrix
    >>> decomposition = compact_fourier_decomposition(U)

    >>> # Create a figure and axis for the plot
    >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    >>> # Plot the phases
    >>> plot_phases(decomposition, ax)

    >>> # Show the plot
    >>> plt.show()
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from unitary_decomp.fourier_interferometer import FourierDecomp
from unitary_decomp.lplm_interferometer import LplmDecomp


def _generate_phase_matrix(decomposition: FourierDecomp | LplmDecomp) -> np.ndarray:
    """Generate 2 2D array of phases from the decomposition.

    This function takes a decomposition object and generates a 2D array of phases from the mask sequence.

    Args:
        decomposition (FourierDecomp | LplmDecomp): The decomposition object containing the mask sequence.

    Returns:
        np.ndarray: A 2D array where each row corresponds to a mask in the sequence.
    """
    phases = np.vstack(decomposition.mask_sequence[::-1])
    phases = np.vstack((phases, decomposition.D))

    return np.angle(phases.T)


def plot_phases(
    decomposition: FourierDecomp | LplmDecomp, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot the angles in the masks from a DFT + phase masks decomposition.

    Given a decomposition object (LplmDecomp or FourierDecomp), this function plots the angles
    of all the masks in the sequence as a colored heatmap. This allows to vizualize the distribution
    of the phases across the masks in the decomposition.

    Args:
        decomposition (FourierDecomp | LplmDecomp): The decomposition object containing the mask sequence. Can be obtained from
            `unitary_decomp.fourier_interferometer.compact_fourier_decomposition` or
            `unitary_decomp.lplm_interferometer.lplm_decomposition`.

        ax (plt.Axes, optional): The matplotlib Axes object to plot on. If None, a new figure and axes will be created.

    Raises:
        TypeError: If the decomposition is not a FourierDecomp or LplmDecomp instance.
    """
    if not isinstance(decomposition, (FourierDecomp, LplmDecomp)):
        raise TypeError("decomposition must be an instance of FourierDecomp or LplmDecomp")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Generate the phase matrix from the decomposition
    phases = _generate_phase_matrix(decomposition)

    # Plot the phases as a heatmap
    im = ax.matshow(phases, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)

    # Set colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Phase")

    # Set colorbar ticks in unit of pi
    cbar.set_ticks(
        [
            -np.pi,
            -3 * np.pi / 4,
            -np.pi / 2,
            -np.pi / 4,
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4,
            np.pi,
        ]
    )
    cbar.set_ticklabels(
        [
            r"$-\pi$",
            r"$-\frac{3\pi}{4}$",
            r"$-\frac{\pi}{2}$",
            r"$-\frac{\pi}{4}$",
            r"$0$",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
            r"$\frac{3\pi}{4}$",
            r"$\pi$",
        ]
    )

    # Set x-axis ticks position
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")

    # Set axis labels
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mode")

    return ax
