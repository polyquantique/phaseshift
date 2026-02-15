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

"""# Unitary Decompositions

This package provides functions to decompose and approximate linear optical transformations into various interferometer configurations.

The available decompositions include:

- **Clements Interferometer**: Decomposes a unitary matrix into a rectangular mesh of asymmetric Mach-Zehnder interferometers and phase shifters.

- **Bell Interferometer**: Decomposes a unitary matrix into a rectangular mesh of symmetric Mach-Zehnder interferometers and phase shifters.

- **LPLM Interferometer**: Decomposes a unitary matrix into a sequence of 6N + 1 phase masks interleaved with discrete Fourier transforms.

- **Fourier Interferometer**: Decomposes a unitary matrix into a sequence of 4N + 1 or 2N + 5 phase masks interleaved with discrete Fourier transforms.

For more details on each decomposition, refer to the respective module documentation.

`Unitary-Decomp` also includes optimization routines in the `optimization` subpackage to find approximated decompositions using gradient-based methods.
"""

__version__ = "0.1.0"

from unitary_decomp.bell_interferometer import bell_decomposition
from unitary_decomp.clements_interferometer import clements_decomposition, mzi_decomposition
from unitary_decomp.fourier_interferometer import (
    compact_fourier_decomposition,
    fourier_decomposition,
)
from unitary_decomp.lplm_interferometer import lplm_decomposition
from unitary_decomp.optimization.fourier_optimizer import mask_optimizer
from unitary_decomp.optimization.jax_optimizer import jax_mask_optimizer
