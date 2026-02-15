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
# Clements Interferometer Plotting Module

This module provides functions to characterize the `clements_interferometer` module and to plot the results.
It includes functions to plot the fidelity of reconstructed circuits against circuit size and loss per beam splitter,
as well as to measure and plot the time taken to run the Clements decomposition algorithm.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, unitary_group

from unitary_decomp.clements_interferometer import circuit_reconstruction, clements_decomposition


def fidelity(V: np.ndarray, U: np.ndarray) -> float:
    """
    Calculate the fidelity between two unitary matrices v and U.

    Args:
        V (np.ndarray): First unitary matrix.
        U (np.ndarray): Second unitary matrix.

    Returns:
        float: Fidelity between the two matrices.

    Raises:
        ValueError: If the matrices are not of the same size.
    """
    if V.shape != U.shape:
        raise ValueError("The matrices must be of the same size.")

    return np.abs(np.trace(U.T.conj() @ V) / np.sqrt(U.shape[0] * np.trace(V.T.conj() @ V))) ** 2


def fidelity_vs_circuit_size(max_size: int = 50, num_trials=60) -> tuple:
    """
    Plot the fidelity of reconstructed circuits with respect to circuit size.

    This function generates a plot of the average fidelity of lossy circuits with respect to
    circuit size (number of modes). It averages the fidelity over `num_trials` random unitaries
    per circuit size and displays error bars (std / sqrt(num_trials)).
    Assumes a fixed loss of 0.2 dB per beam splitter.
    Returns the matplotlib figure and axis.

    Args:
        max_size (int): Maximum circuit size to consider.
        num_trials (int): Number of trials for averaging.

    Returns:
        tuple: Matplotlib figure and axis objects.
    """
    # Circuit sizes from 2 to max_size
    sizes = np.arange(2, max_size + 1, 3)
    mean_fidelities = []
    fidelity_uncertainties = []

    # Loss per beam splitter in dB
    loss_db = 0.2

    for n in sizes:
        fidelities = []

        for _ in range(num_trials):
            U = unitary_group.rvs(n)

            # Decompose the unitary matrix
            decomposition = clements_decomposition(U)

            # Reconstruct the circuit with losses
            U_exp = circuit_reconstruction(decomposition, loss_db=loss_db)
            # Calculate fidelity
            fidelities.append(fidelity(U_exp, U))

        mean_fidelities.append(np.mean(fidelities))
        fidelity_uncertainties.append(np.std(fidelities) / np.sqrt(num_trials))

    # Plotting
    fig, ax = plt.subplots()
    ax.errorbar(
        sizes,
        mean_fidelities,
        yerr=fidelity_uncertainties,
        marker="o",
        capsize=3,
        markersize=3,
        color="k",
    )
    ax.set_xlabel("Circuit size (number of modes)")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"Fidelity vs. Circuit Size (0.2 dB loss/beam splitter, {num_trials} trials)")

    return fig, ax


def fidelity_vs_loss(size: int = 20, num_trials: int = 60) -> tuple:
    """
    Plot the fidelity of reconstructed circuits vs. loss (in dB).

    This function generates a plot of the average fidelity of lossy circuits with respect to
    loss per beam splitter (in dB) for a fixed circuit size. It averages the fidelity over `num_trials` random
    unitaries per loss value and displays error bars (std / sqrt(num_trials)).
    Assumes a fixed circuit size of `size`.
    Returns the matplotlib figure and axis.

    Args:
        size (int): Circuit size (Number of modes).
        num_trials (int): Number of trials for averaging.

    Returns:
        tuple: Matplotlib figure and axis objects.
    """
    # Losses from 0 to 1 dB
    losses = np.arange(0, 1.05, 0.1)
    mean_fidelities = []
    fidelity_uncertainties = []

    for loss_db in losses:
        fidelities = []

        for _ in range(num_trials):
            U = unitary_group.rvs(size)

            # Decompose the unitary matrix
            decomposition = clements_decomposition(U)

            # Reconstruct the circuit with loss
            U_exp = circuit_reconstruction(decomposition, loss_db=loss_db)
            # Calculate fidelity
            fidelities.append(fidelity(U_exp, U))

        mean_fidelities.append(np.mean(fidelities))
        fidelity_uncertainties.append(np.std(fidelities) / np.sqrt(num_trials))

    # Plotting
    fig, ax = plt.subplots()
    ax.errorbar(
        losses,
        mean_fidelities,
        yerr=fidelity_uncertainties,
        marker="o",
        capsize=3,
        markersize=3,
        color="k",
    )
    ax.set_xlabel("Loss per beam splitter (dB)")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"Fidelity vs. Loss (size {size}, {num_trials} trials)")

    return fig, ax


def compute_times(
    N_max: int = 200, number_of_points: int = 20, n_trials: int = 20, save: bool = True
) -> tuple:
    """
    Measure the time taken to decompose random unitary matrices of increasing size.

    This function generates random unitary matrices of sizes ranging from 10 to `N_max` and measures the time taken
    to decompose each matrix using the Clements decomposition algorithm. The times are averaged over `n_trials` trials
    for each matrix size. The results are saved to a CSV file.

    Args:
        N_max (int): Maximum size of the unitary matrix to be generated.
        number_of_points (int): Number of points to sample between 10 and `N_max`.
        n_trials (int): Number of trials for averaging the time taken.
        save (bool): Whether to save the results to a CSV file. Default is True.

    Returns:
        tuple: N (array of matrix sizes) and average_times (array of average times taken for decomposition).
    """
    # Matrix sizes
    N = np.logspace(1, np.log10(N_max), number_of_points, dtype=int)

    average_times = []

    for size in N:
        times = []

        for _ in range(n_trials):
            U = unitary_group.rvs(size)

            # Decompose the unitary matrix
            start = time.perf_counter()
            clements_decomposition(U)
            end = time.perf_counter()

            # Measure time taken
            times.append(end - start)
        average_times.append(np.mean(times))

    # Save results to CSV
    df = pd.DataFrame({"N": N, "Time (seconds)": average_times})
    df.to_csv("clements_decomposition_times.csv", index=False)

    return N, average_times


def plot_time_behavior(path: str = "clements_decomposition_times.csv") -> tuple:
    """
    Plots the time behavior of the Clements decomposition algorithm.

    This function reads the time data from a CSV file, fits a line to the log-log plot of time vs. number of modes,
    and generates a plot showing the time taken for decomposition as a function of the number of modes.

    Args:
        path (str): Path to the CSV file containing the time data. Default is "clements_decomposition_times.csv".

    Returns:
        tuple: Matplotlib figure and axis objects.
    """
    # Read the CSV file containing the time data
    df = pd.read_csv(path)

    # Fit a line to the data
    a, b = linregress(np.log(df["N"]), np.log(df["Time (seconds)"]))[:2]
    print(f"Fitted line: y = {np.exp(b):.2f} * x^{a:.2f}")

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(df["N"], df["Time (seconds)"], marker="o", color="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of modes (N)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Clements Decomposition Time vs. Number of Modes")

    # Plot the fitted line
    ax.plot(
        df["N"],
        np.exp(b) * df["N"] ** a,
        linestyle="--",
        linewidth=1,
        label="Slope: {:.2f}".format(a),
    )
    ax.legend()

    return fig, ax
