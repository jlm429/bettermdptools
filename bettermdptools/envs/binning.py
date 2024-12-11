"""
Utility functions for generating bin edges for discretizing continuous variables.
"""

def generate_bin_edges(range_limit, n_bins, width_ratio, center=True):
    """Generates bin edges for a symmetric range with exponentially increasing/decreasing bin widths.

    Parameters
    ----------
    range_limit : float
        The extreme value of the range [-range_limit, range_limit].

    n_bins : int
        Number of bins (must be odd).
    
    width_ratio : float
        Ratio of the outermost bin widths to the central bin width.

    center : bool, default=True
        True: Outer bins are wider than the center bin.
        False: Center bin is wider than the outer bins.

    Returns
    -------

    list[float]
        The edges of the bins. Shape (n_bins + 1)
    """

    # Parameter validation
    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    if n_bins % 2 == 0:
        raise ValueError("n_bins must be an odd integer.")
    if range_limit <= 0:
        raise ValueError("range_limit must be a positive number.")
    if width_ratio <= 0:
        raise ValueError("width_ratio must be a positive number.")

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
        geometric_sum = 1 + 2 * (q * (q ** k - 1) / (q - 1))
    else:
        # If q == 1, all bins have the same width
        geometric_sum = n_bins

    # Calculate the width of the central bin
    w0 = 2 * range_limit / geometric_sum

    # Generate bin widths: [w_k, ..., w1, w0, w1, ..., w_k]
    bin_widths_left = [w0 * q ** i for i in range(k, 0, -1)]
    bin_widths_right = [w0 * q ** i for i in range(1, k + 1)]
    bin_widths = bin_widths_left + [w0] + bin_widths_right

    # Construct bin edges starting from -range_limit
    bin_edges = [-range_limit]
    for width in bin_widths:
        bin_edges.append(bin_edges[-1] + width)

    # Due to floating-point arithmetic, ensure the last edge is exactly range_limit
    bin_edges[-1] = range_limit

    return bin_edges

if __name__ == '__main__':
    # Test the generate_bin_edges function
    range_limit = 10
    n_bins = 11
    width_ratio = 3
    center = True

    bin_edges = generate_bin_edges(range_limit, n_bins, width_ratio, center)
    center_bin_width = bin_edges[n_bins // 2 + 1] - bin_edges[n_bins // 2]
    first_bin_width = bin_edges[1] - bin_edges[0]
    last_bin_width = bin_edges[-1] - bin_edges[-2]
    print(f'Center bin width: {center_bin_width}')
    print(f'First bin width: {first_bin_width}')
    print(f'Last bin width: {last_bin_width}')

    # Plot the bin edges as vertical lines
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    for edge in bin_edges:
        plt.axvline(edge, color='k', linestyle='--', linewidth=0.5)
    plt.xlim(-range_limit, range_limit)
    plt.ylim(0, 1)
    plt.show()
    

    # print("Bin Edges:", bin_edges)
