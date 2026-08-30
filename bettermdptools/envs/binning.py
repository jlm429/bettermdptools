"""
Utility functions for generating bin edges for discretizing continuous variables.
"""

from numbers import Integral, Real

import numpy as np


def generate_bin_edges(range_limit, n_bins, width_ratio, center=True):
    """Generate symmetric bin edges whose widths change exponentially.

    Parameters
    ----------
    range_limit : float
        Finite positive extreme of the range [-range_limit, range_limit].

    n_bins : int
        Number of bins (must be odd).

    width_ratio : float
        Finite positive ratio of the outermost to central bin widths.

    center : bool, default=True
        True: Outer bins are wider than the center bin.
        False: Center bin is wider than the outer bins.

    Returns
    -------

    list[float]
        The edges of the bins. Shape (n_bins + 1)
    """

    # Parameter validation
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    n_bins = int(n_bins)
    if n_bins % 2 == 0:
        raise ValueError("n_bins must be an odd integer.")
    if (
        isinstance(range_limit, bool)
        or not isinstance(range_limit, Real)
        or not np.isfinite(range_limit)
        or range_limit <= 0
    ):
        raise ValueError("range_limit must be a finite positive number.")
    if (
        isinstance(width_ratio, bool)
        or not isinstance(width_ratio, Real)
        or not np.isfinite(width_ratio)
        or width_ratio <= 0
    ):
        raise ValueError("width_ratio must be a finite positive number.")
    if not isinstance(center, (bool, np.bool_)):
        raise ValueError("center must be a boolean.")

    # Normalize NumPy scalars so NumPy 2 promotion rules do not reduce the
    # precision of the geometric calculations or their symmetry.
    range_limit = float(range_limit)
    width_ratio = float(width_ratio)
    if not np.isfinite(range_limit):
        raise ValueError("range_limit must be representable as a finite float.")
    if not np.isfinite(width_ratio):
        raise ValueError("width_ratio must be representable as a finite float.")

    k = (n_bins - 1) // 2  # Number of bins on each side of the center

    if k == 0:
        # Only one bin covering the entire range
        return [-range_limit, range_limit]

    # Calculate the common ratio q for the geometric progression
    if center:
        # Outer bins are wider: q > 1
        q = width_ratio ** (1 / k)
    else:
        # Center bin is wider: q < 1
        q = (1 / width_ratio) ** (1 / k)

    # Calculate the sum of the geometric series for bin widths
    if q != 1.0:
        # Sum = 1 (center) + 2 * (q + q^2 + ... + q^k)
        geometric_sum = 1 + 2 * (q * (q**k - 1) / (q - 1))
    else:
        # If q == 1, all bins have the same width
        geometric_sum = n_bins

    # Calculate the width of the central bin
    w0 = range_limit * (2 / geometric_sum)

    # Generate bin widths: [w_k, ..., w1, w0, w1, ..., w_k]
    bin_widths_left = [w0 * q**i for i in range(k, 0, -1)]
    bin_widths_right = [w0 * q**i for i in range(1, k + 1)]
    bin_widths = bin_widths_left + [w0] + bin_widths_right

    # Construct bin edges starting from -range_limit
    bin_edges = [-range_limit]
    for width in bin_widths:
        bin_edges.append(bin_edges[-1] + width)

    # Due to floating-point arithmetic, ensure the last edge is exactly range_limit
    bin_edges[-1] = range_limit

    if not np.isfinite(bin_edges).all():
        raise ValueError("parameters must produce finite bin edges.")

    return bin_edges


if __name__ == "__main__":
    # Test the generate_bin_edges function
    range_limit = 10
    n_bins = 11
    width_ratio = 3
    center = True

    bin_edges = generate_bin_edges(range_limit, n_bins, width_ratio, center)
    center_bin_width = bin_edges[n_bins // 2 + 1] - bin_edges[n_bins // 2]
    first_bin_width = bin_edges[1] - bin_edges[0]
    last_bin_width = bin_edges[-1] - bin_edges[-2]
    print(f"Center bin width: {center_bin_width}")
    print(f"First bin width: {first_bin_width}")
    print(f"Last bin width: {last_bin_width}")

    # Plot the bin edges as vertical lines
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    for edge in bin_edges:
        plt.axvline(edge, color="k", linestyle="--", linewidth=0.5)
    plt.xlim(-range_limit, range_limit)
    plt.ylim(0, 1)
    plt.show()

    # print("Bin Edges:", bin_edges)
