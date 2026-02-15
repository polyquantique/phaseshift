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
# Optimization subpackage

This subpackage provides optimization routines for approximating unitary matrices using phase masks and discrete Fourier transforms.

The `optimization` subpackage contains two modules:
- `fourier_optimizer`: Uses the basin-hopping function from `scipy.optimize` to find the phase masks that minimize the infidelity with respect to a target unitary.
- `jax_optimizer`: Uses the `Jax` and `Optax` libraries to find the phase masks that minimize the infidelity with a target unitary efficiently.
"""

from unitary_decomp.optimization.fourier_optimizer import mask_optimizer
from unitary_decomp.optimization.jax_optimizer import jax_mask_optimizer, scipy_mask_optimizer
