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
Sample code to demonstrate the use of the Clements Interferometer module.
"""

import numpy as np

from unitary_decomp.clements_interferometer import circuit_reconstruction, clements_decomposition

# Example unitary matrix (U = H \otimes H)
U = 1 / 2 * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
print("Initial Unitary:\n", U)

# Apply Clements decomposition
decomposition = clements_decomposition(U)

# Print the decomposition parameters
print("\nCircuit:\n")
for theta, phi, target in decomposition.circuit:
    print(f"theta: {theta:.3f}, phi: {phi:.2f}, target: {target}")

# Reconstruct the unitary from the decomposition
reconstructed_unitary = circuit_reconstruction(decomposition)

# Print the reconstructed unitary and assert initial matrix
print("\nReconstructed Unitary:\n", reconstructed_unitary.round(4))
assert np.allclose(U, reconstructed_unitary), "Reconstructed unitary does not match original."
