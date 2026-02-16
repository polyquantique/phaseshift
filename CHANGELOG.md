## [0.2.0] - 2026-02-16
### Implementation of the geodesic distance

The geodesic distance provides a more accurate description of the real distance between unitary matrices in the SU(N) sub-manifold [1]. It was implemented as an alternative cost function to the infidelity in the `jax_optimizer` module for better and faster convergence during gradient descent optimization.

### Renaming of the package

The name of the package was changed to `PhaseShift` in preparation for the release.

#### References
- [1] Álvarez-Vizoso, Javier, and David Barral. "Universality and Optimal Architectures for Layered Programmable Unitary Decompositions." arXiv preprint arXiv:2510.19397 (2025).