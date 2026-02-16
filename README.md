[![Paper](https://img.shields.io/badge/Paper-Read%20Article-blue)](https://doi.org/10.1364/JOSAB.577579)

# PhaseShift

Decomposition and approximation tools for linear optical unitaries.

## Table of Contents

- [About this project](#about-this-project)
- [Package contents](#package-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Citing this work](#citing-this-work)
- [License](#license)
- [References](#references)

## About This Project

`PhaseShift` is a **Python** package for performing various **decompositions** and **approximations** of unitary matrices into planar arrangements of simple optical components. These tools can be used to design and program **universal multiport interferometers** (UMIs), devices capable of implementing arbitrary linear transformations on multiple optical modes. Such devices have numerous applications in communication, imaging, and information processing.

The algorithms implemented in this package cover two main classes of planar UMI architectures: 
- [Networks of two-mode components](#two-mode-component-networks) 
- [Sequences of multichannel components](#multichannel-component-sequences)

### Two-mode Component Networks
This class of decompositions seeks to express an $N \times N$ unitary matrix as a planar mesh of configurable two-mode unit cells, typically realized using Mach–Zehnder interferometers (MZIs). The first design of this kind was proposed by [Reck *et al.*, 1994](https://doi.org/10.1103/PhysRevLett.73.58), who used a triangular mesh of asymmetric Mach–Zehnder interferometers to implement arbitrary unitary transformations. This design was later improved by [Clements *et al.*, 2016](https://doi.org/10.1364/OPTICA.3.001460), who introduced a more compact rectangular mesh using the same unit cells. [Bell *et al.*, 2021](https://doi.org/10.1063/5.0053421) further compactified the Clements *et al.* design by using a symmetric Mach–Zehnder interferometer as unit cell, which helped reduce the optical depth of the interferometer. 

<div align="center">
Rectangular mesh of Mach–Zehnder interferometers based on the Clements architecture for 6 modes 
</div>

![Clements design](figures/Clements.svg)


### Multichannel Component Sequences

This second class of decompositions aims to express an $N \times N$ unitary matrix as a sequence of configurable phase masks interleaved with a multichannel mixing layer, such as the discrete Fourier transform (DFT). Numerical evidence suggests that using $N+1$ layers of phase masks with any dense mixing layer is enough to result in a universal design [(Saygin *et al.*, 2020)](https://doi.org/10.1103/PhysRevLett.124.010501) [(Zelaya *et al.*, 2024)](https://doi.org/10.1038/s41598-024-60700-8). The first constructive design based on this approach was proposed by [López Pastor *et al.*, 2021](https://doi.org/10.1364/OE.432787) and generates a sequence of $6N + 1$ phase masks to implement a $N \times N$ unitary. We improved this design to reach $4N+1$ and $2N+5$ phase masks.

<div align="center">
Sequence of phase masks interleaved with the discrete Fourier transform mixing layer for 6 modes
</div>

![Fourier design](figures/Fourier.svg)

## Package Contents
`PhaseShift` provides tools to perform **exact decompositions** and **numerical approximations** of unitary matrices.

### Exact Decompositions
`PhaseShift` includes four main modules to perform exact decompositions of unitary matrices:

- [`clements_interferometer`](src/phaseshift/clements_interferometer.py): Implementation of the algorithm by [Clements *et al.*, 2016](https://doi.org/10.1364/OPTICA.3.001460) to decompose $N \times N$ unitary matrices into a rectangular mesh of $N(N-1)/2$ **asymmetric** Mach–Zehnder interferometers.

- [`bell_interferometer`](src/phaseshift/bell_interferometer.py): Implementation of the algorithm by [Bell *et al.*, 2021](https://doi.org/10.1063/5.0053421) to decompose $N \times N$ unitary matrices into a rectangular mesh of $N(N-1)/2$ **symmetric** Mach–Zehnder interferometers.

- [`lplm_interferometer`](src/phaseshift/lplm_interferometer.py): Implementation of the algorithm by [López Pastor *et al.*, 2021](https://doi.org/10.1364/OE.432787) to decompose $N \times N$ unitary matrices into a sequence of $6N+1$ phase masks interleaved with the **DFT** matrix.

- [`fourier_interferometer`](src/phaseshift/fourier_interferometer.py): Implementation of the **Fourier decomposition** and the **compact Fourier decomposition** to decompose $N \times N$ unitary matrices into sequences of $4N+1$ and $2N+5$ phase masks interleaved with **DFT** respectively.

### Optimization Tools

In addition to exact decompositions, `PhaseShift` also has an `optimization` subpackage, which contains tools to approximate unitary matrices into a sequence of phase masks interleaved with a chosen mixing layer. The `optimization` subpackage has two modules:

- [`fourier_optimizer`](src/phaseshift/optimization/fourier_optimizer.py): Uses the basin-hopping algorithm from `scipy.optimize` to solve a global minimization problem, yielding the sequence of phase masks that minimizes the infidelity with respect to a target unitary.

- [`jax_optimizer`](src/phaseshift/optimization/jax_optimizer.py): Uses `Jax` and `Optax` to perform gradient-based optimization of the phase masks with multiple restarts to minimize the infidelity with respect to a target unitary. This algorithm can run efficiently on CPU or GPU and is significantly faster than the SciPy-based implementation.

---
**Note:** For more detailed descriptions and usage examples, see the documentation of the individual modules.

## Installation

You can install `PhaseShift` from source as follows:

1. Clone the repository

```bash
git clone https://github.com/polyquantique/Unitary-Decomp.git
cd Unitary-Decomp
```

2. (Optional) Create and activate a virtual environment

- Linux / macOS:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
- Windows (Command Prompt):

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. Install the package and dependencies

- Standard installation:

    ```bash
    pip install .
    ```
- Editable (developer) installation:

    ```bash
    pip install -e .[dev]
    ```
    **Notes:**

    For GPU support with JAX on Linux, install the appropriate CUDA-enabled version:

```bash
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

4. (Optional) Run the tests

```bash
pip install pytest
pytest tests
```

## Usage

### 1. `clements_interferometer` module
This first example shows how to use the `clements_interferometer` module to decompose a random unitary matrix.

```python
>>> from phaseshift import clements_interferometer as ci
>>> from scipy.stats import unitary_group
>>> import numpy as np
```

The function `clements_decomposition` performs the decomposition described in [Clements *et al.*, 2016](https://doi.org/10.1364/OPTICA.3.001460) on a random unitary matrix `U`.
```python
>>> # Generate a random unitary matrix
>>> U = unitary_group(dim = 8, seed = 137).rvs() 

>>> # Compute the Clements decomposition
>>> decomposition = ci.clements_decomposition(U)
```

The output of this function is a `Decomposition` object, which has a `circuit` attribute. `Decomposition.circuit` is a list of `MachZehnder` objects that contain the parameters $\theta$ and $\phi$ of each unit cell in the mesh.

```python
>>> # Extract the circuit from the decomposition
>>> circuit = decomposition.circuit

>>> # Print the parameters of the first unit cell in the circuit
>>> print(circuit[0])
MachZehnder(theta=np.float64(0.7697802543915319), phi=np.float64(3.8400842306814207), target=(5, 6))
```

The function `circuit_reconstruction` allows to compute the matrix that corresponds to a `Decomposition` object. This matrix can be compared to the original matrix.

```python
>>> # Reconstruct the unitary matrix from the decomposition
>>> reconstructed_matrix = ci.circuit_reconstruction(decomposition)

>>> # Compare with the initial matrix
>>> print(np.allclose(U, reconstructed_matrix))
True
```

### 2. `fourier_interferometer` module

This example shows how to use the `fourier_interferometer` module to decompose a random unitary matrix.

```python
>>> from phaseshift import fourier_interferometer as fi
>>> from scipy.stats import unitary_group
>>> import numpy as np
```

The `compact_fourier_decomposition` function decomposes a random $N \times N$ unitary matrix into a sequence of $2N+5$ phase masks and $2N+4$ DFT layers.

```python
>>> # Generate a random unitary matrix
>>> U = unitary_group(dim = 8, seed = 137).rvs() 

>>> # Compute the Compact Fourier decomposition
>>> decomposition = fi.compact_fourier_decomposition(U)
```

The output is a `FourierDecomp` object, which contains the `mask_sequence` attribute. `FourierDecomp.mask_sequence` stores the sequence of phase masks to be interleaved with DFT matrices.

```python
>>> # Extract the mask sequence from the decomposition
>>> mask_sequence = decomposition.mask_sequence

>>> # Print the first mask in the sequence
>>> print(mask_sequence[0].round(3))
[0.707-0.707j 0.707+0.707j 0.707-0.707j 0.707+0.707j 0.707-0.707j
 0.707+0.707j 0.707-0.707j 0.707+0.707j]
```

The function `circuit_reconstruction` computes the matrix given by a `FourierDecomp` object by inserting a DFT matrix between each phase mask. The matrix can then be compared to the initial matrix.

```python
>>> # Reconstruct the unitary matrix from the decomposition
>>> reconstructed_matrix = fi.circuit_reconstruction(decomposition)

>>> # Compare with the initial matrix
>>> print(np.allclose(U, reconstructed_matrix))
True
```


### 3. `optimization.jax_optimizer` module

This example shows how to use the `jax_optimizer` module found in the `optimization` subpackage to find numerically a sequence of phase masks that approximate a given unitary matrix.

```python
>>> from phaseshift.optimize import jax_optimizer as jo
>>> from scipy.stats import unitary_group
>>> import numpy as np
```

The function `jax_mask_optimizer` uses an Adam optimizer with multiple restarts to optimize a sequence of phase masks of a given length. In this case, we optimize 9 phase masks to ensure a complete parametrization of an $8 \times 8$ random unitary `U`. 

```python
>>> # Generate a random unitary matrix
>>> U = unitary_group(dim = 8, seed = 137).rvs() 

>>> # Compute the phase masks that minimize infidelity
>>> decomp, infidelity = jo.jax_mask_optimizer(U, length=9, steps=3500, restarts=50)
```

The function returns the decomposition, given as a `FourierDecomp` object, as well as the final infidelity obtained by the optimizer. 

```python
>>> # Print the final infidelity
>>> print(infidelity)
2.220446049250313e-16

>>> # Print the type of decomp
>>> print(type(decomp).__name__)
FourierDecomp
```

The final matrix can be reconstructed using the `circuit_reconstruction` function from the `fourier_interferometer` module.


## Documentation

The **LPLM algorithm** found in the [`lplm_interferometer`](src/phaseshift/lplm_interferometer.py) module was adapted from [López Pastor *et al.*, 2021](https://doi.org/10.1364/OE.432787) and uses a slightly different sequence of phase masks than the original paper. A comprehensive derivation of this new sequence can be found in the following document:

- [Decomposition of Unitary Matrices Using Fourier
Transforms and Phase Masks](papers/LPLM_algorithm_derivation.pdf)

## Citing This Work
If you find our research useful in your work, please cite it as
```
@article{girouard2025,
author = {Vincent Girouard and Nicol\'{a}s Quesada},
journal = {J. Opt. Soc. Am. B},
number = {3},
pages = {A66--A73},
title = {Near-optimal decomposition of unitary matrices using phase masks and the discrete Fourier transform},
volume = {43},
year = {2026},
url = {https://opg.optica.org/josab/abstract.cfm?URI=josab-43-3-A66},
doi = {10.1364/JOSAB.577579}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## References

- Girouard, Vincent, Nicolas, Quesada. "Near-optimal decomposition of unitary matrices using phase masks and the discrete Fourier transform". JOSA B Vol. 43, Issue 3, (2026): A66-A73 

- Clements, William R., et al. "Optimal design for universal multiport interferometers." Optica 3.12 (2016): 1460-1465.

- Bell, Bryn A., and Ian A. Walmsley. "Further compactifying linear optical unitaries." Apl Photonics 6.7 (2021).

- López Pastor, Víctor, Jeff Lundeen, and Florian Marquardt. "Arbitrary optical wave evolution with Fourier transforms and phase masks." Optics Express 29.23 (2021): 38441-38450.

- Saygin, M. Yu, et al. "Robust architecture for programmable universal unitaries." Physical review letters 124.1 (2020): 010501.

- Pereira, Luciano, et al. "Minimum optical depth multiport interferometers for approximating arbitrary unitary operations and pure states." Physical Review A 111.6 (2025): 062603.
