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

from unitary_decomp import matrix_operations as mo

"""Unit tests for the matrix operations module."""


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_p_matrix_factorization(dim: int) -> None:
    """Test the matrix factorization of the P permutation matrix: P = X Ci X."""
    factorization = mo.x_matrix(dim) @ mo.ci_matrix(dim) @ mo.x_matrix(dim)
    assert np.allclose(factorization, mo.p_permutation_matrix(dim))


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_p_matrix_inverse(dim: int) -> None:
    """Test the inverse of the P permutation matrix: P.T @ P = I."""
    p_matrix = mo.p_permutation_matrix(dim)
    assert np.allclose(p_matrix @ p_matrix.T, np.eye(dim))
    assert np.allclose(p_matrix.T @ p_matrix, np.eye(dim))


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_x_matrix_factorization(dim: int) -> None:
    """Test the matrix factorization of the X matrix: X = G Y G"""
    factorization = np.diag(mo.g_matrix(dim)) @ mo.y_matrix(dim) @ np.diag(mo.g_matrix(dim))
    assert np.allclose(factorization, mo.x_matrix(dim))


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_xx_identity(dim: int) -> None:
    """Test that the X matrix squared is the identity matrix."""
    assert np.allclose(np.eye(dim), mo.x_matrix(dim) @ mo.x_matrix(dim))


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_x_matrix_swap_factorization(dim: int) -> None:
    """Test the matrix factorization of the X' matrix: X' = G' Y G'."""
    factorization = (
        np.diag(mo.g_matrix_swap(dim)) @ mo.y_matrix(dim) @ np.diag(mo.g_matrix_swap(dim))
    )
    assert np.allclose(factorization, mo.x_matrix_swap(dim))


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_g_matrix_swap(dim: int) -> None:
    """Test that G' = swap @ G @ swap."""
    g_matrix_swap = mo.swap_matrix(dim) @ np.diag(mo.g_matrix(dim)) @ mo.swap_matrix(dim)
    assert np.allclose(g_matrix_swap, np.diag(mo.g_matrix_swap(dim)))


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_z_matrix_factorization(dim: int) -> None:
    """Test the matrix factorization of the Z matrix: Z = G @ G."""
    factorization = np.diag(mo.g_matrix(dim)) @ np.diag(mo.g_matrix(dim))
    assert np.allclose(factorization, np.diag(mo.z_matrix(dim)))


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_z_matrix_swap_factorization(dim: int) -> None:
    """Test the matrix factorization of the Z' matrix: Z' = G' @ G'."""
    factorization = np.diag(mo.g_matrix_swap(dim)) @ np.diag(mo.g_matrix_swap(dim))
    assert np.allclose(factorization, np.diag(mo.z_matrix_swap(dim)))


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_x_matrix_swap(dim: int) -> None:
    """Test that X' = swap @ X @ swap."""
    x_matrix_swap = mo.swap_matrix(dim) @ mo.x_matrix(dim) @ mo.swap_matrix(dim)
    assert np.allclose(mo.x_matrix_swap(dim), x_matrix_swap)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_dft_inversion_by_permutation(dim: int) -> None:
    """Test the inversion of the DFT matrix by multiplication with the Pi permutation matrix."""
    assert np.allclose(
        dft(dim, scale="sqrtn").T.conj(), mo.pi_permutation_matrix(dim) @ dft(dim, scale="sqrtn")
    )
    assert np.allclose(
        dft(dim, scale="sqrtn").T.conj(), dft(dim, scale="sqrtn") @ mo.pi_permutation_matrix(dim)
    )


def test_pi_permutation_diagonals() -> None:
    """Test that Pi @ D @ Pi is a diagonal matrix with reversed order of elements."""
    dim = 12
    D = np.arange(1, dim + 1)
    D_permutation = np.diag(mo.pi_transformation(np.diag(D)))
    assert np.isclose(D[0], D_permutation[0]) and np.allclose(D[1:], D_permutation[:0:-1])


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_y_matrix_eigendecomp(dim: int) -> None:
    """Test the eigendecomposition of the Y matrix: Y = F.T.conj() @ E @ F."""
    eigen_decomp = (
        dft(dim, scale="sqrtn").T.conj() @ np.diag(mo.e_matrix(dim)) @ dft(dim, scale="sqrtn")
    )
    assert np.allclose(mo.y_matrix(dim), eigen_decomp)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_ci_matrix_eigendecomp(dim: int) -> None:
    """Test the eigendecomposition of the CI matrix: CI = F.T.conj() @ H @ F."""
    eigen_decomp = (
        dft(dim, scale="sqrtn").T.conj() @ np.diag(mo.h_matrix(dim)) @ dft(dim, scale="sqrtn")
    )
    assert np.allclose(mo.ci_matrix(dim), eigen_decomp)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_ci_matrix_transposed_eigendecomp(dim: int) -> None:
    """Test the eigendecomposition of the transposed CI matrix: CI.T = F.T.conj() @ H.conj() @ F."""
    eigen_decomp = (
        dft(dim, scale="sqrtn").T.conj()
        @ np.diag(mo.h_matrix(dim).conj())
        @ dft(dim, scale="sqrtn")
    )
    eigen_decomp_variation = (
        dft(dim, scale="sqrtn")
        @ np.diag(mo.h_matrix(dim))
        @ dft(dim, scale="sqrtn")
        @ mo.pi_permutation_matrix(dim)
    )
    assert np.allclose(eigen_decomp, mo.ci_matrix(dim).T)
    assert np.allclose(eigen_decomp_variation, mo.ci_matrix(dim).T)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_swap_eigendecomposition(dim: int) -> None:
    """Test the eigendecomposition of the Swap matrix: S = F.T.conj() @ V @ F."""
    eigen_decomp = (
        dft(dim, scale="sqrtn").T.conj() @ np.diag(mo.v_matrix(dim)) @ dft(dim, scale="sqrtn")
    )
    assert np.allclose(mo.swap_matrix(dim), eigen_decomp)


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_pi_permutation_commutation(dim: int) -> None:
    """Test the commutation relations of the Pi permutation matrix with E and H matrices.

    E@Pi = Pi@E and H@Pi = Pi@H.conj()
    """
    assert np.allclose(
        mo.pi_permutation_matrix(dim) @ np.diag(mo.e_matrix(dim)),
        np.diag(mo.e_matrix(dim)) @ mo.pi_permutation_matrix(dim),
    )
    assert np.allclose(
        mo.pi_permutation_matrix(dim) @ np.diag(mo.h_matrix(dim)),
        np.diag(mo.h_matrix(dim).conj()) @ mo.pi_permutation_matrix(dim),
    )


@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_channel_permutation(dim: int) -> None:
    """Test the inverse of the channel permutation matrix, i.e. K @ K.T = I."""
    assert np.allclose(mo.channels_permutation(dim) @ mo.channels_permutation(dim).T, np.eye(dim))


@pytest.mark.parametrize("dim", range(2, 15, 1))
def test_matrix_interleave_dft(dim: int) -> None:
    """Test the `matrix_interleave` function with the default mixing layer (DFT)."""
    masks = [np.exp(1.0j * 2 * np.pi * np.random.rand(dim)) for _ in range(10)]

    out = np.eye(dim)
    for mask in masks:
        out = out @ dft(dim, scale="sqrtn") @ np.diag(mask)

    assert np.allclose(out, mo.matrix_interleave(masks))


@pytest.mark.parametrize(
    "M", [np.eye(8), np.arange(1, 65).reshape(8, 8), unitary_group(8, 42).rvs()]
)
def test_matrix_interleave_mixing_layer(M: np.ndarray) -> None:
    """Test the `matrix_interleave` function with a custom mixing layer."""
    masks = [np.exp(1.0j * 2 * np.pi * np.random.rand(8)) for _ in range(10)]

    out = np.eye(8)
    for mask in masks:
        out = out @ M @ np.diag(mask)

    assert np.allclose(out, mo.matrix_interleave(masks, mixing_layer=M))


@pytest.mark.parametrize("phi", [0, np.pi / 3, np.pi / 2, np.pi, 0.45654, 1.2655])
@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_x_lambda_x_matrix_diagonalization(dim: int, phi: float) -> None:
    """Test the diagonalization of the X @ lambda @ X matrix using the L matrix."""
    x_lambda_x = mo.x_matrix(dim) @ np.diag(mo.lambda_matrix(dim, phi)) @ mo.x_matrix(dim)

    diagonalization = (
        dft(dim, scale="sqrtn").T.conj() @ np.diag(mo.l_matrix(dim, phi)) @ dft(dim, scale="sqrtn")
    )

    assert np.allclose(x_lambda_x, diagonalization)


@pytest.mark.parametrize("phi", [0, np.pi / 3, np.pi / 2, np.pi, 0.45654, 1.2655])
@pytest.mark.parametrize("dim", range(2, 20, 2))
def test_ci_x_lambda_x_diagonalization(dim: int, phi: float) -> None:

    ci_x_lambda_x = (
        mo.ci_matrix(dim)
        @ mo.x_matrix(dim)
        @ np.diag(mo.lambda_matrix(dim, phi))
        @ mo.x_matrix(dim)
    )

    diagonalization = (
        dft(dim, scale="sqrtn").T.conj()
        @ np.diag(mo.l_matrix(dim, phi) * mo.h_matrix(dim))
        @ dft(dim, scale="sqrtn")
    )

    assert np.allclose(ci_x_lambda_x, diagonalization)
