"""
Diffusion-Limited Aggregation (Witten-Sander 1981, PRL 47, 1400).

2D on-lattice (square lattice) implementation with:
  - Launch circle just outside current cluster radius R_max
  - Kill radius (~2*R_launch) to discard runaway walkers
  - Big-step trick: when |r| > R_launch, jump a distance |r| - R_launch
    in a random direction (cheap approximation to the harmonic
    measure outside the bounding circle; exact form uses the 2D
    Green's function with image charge)
  - Set-based occupancy for O(1) stick check
  - Fractal dimension extracted from N(R_g) ~ R_g^D

Expected: D ~= 1.71 in 2D for large clusters.
Partial sticking (stick_prob < 1) thickens branches; introduces length
scale lambda = l/s (Witten-Sander eq. 13).
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt


def run_dla(n_particles, stick_prob=1.0, seed=None):
    """
    Grow a DLA cluster of n_particles on the 2D square lattice.

    Returns:
        sites: (N, 2) integer array of particle coordinates, in stick order
        Rg_history: radius of gyration vs N (recorded every 50 sticks)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cluster = {(0, 0)}
    order = [(0, 0)]

    R_max = 1.0
    BUFFER = 5
    KILL_FACTOR = 2.0

    # Running sums for R_g:  R_g^2 = (1/N) sum r_i^2  -  (1/N^2)(sum r_i)^2
    sum_x = 0.0
    sum_y = 0.0
    sum_r2 = 0.0
    n = 1

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    Rg_history = [0.0]

    while n < n_particles:
        R_launch = R_max + BUFFER
        R_kill = KILL_FACTOR * R_launch

        theta = random.uniform(0, 2 * math.pi)
        x = R_launch * math.cos(theta)
        y = R_launch * math.sin(theta)

        while True:
            r = math.hypot(x, y)

            if r > R_kill:
                break

            if r > R_launch:
                step = r - R_launch
                phi = random.uniform(0, 2 * math.pi)
                x += step * math.cos(phi)
                y += step * math.sin(phi)
                continue

            ix = int(round(x))
            iy = int(round(y))

            # Check neighbors -- break only on successful stick.
            stuck = False
            for dx, dy in neighbors:
                if (ix + dx, iy + dy) in cluster:
                    if stick_prob >= 1.0 or random.random() < stick_prob:
                        stuck = True
                        break  # only exit loop if particle actually stuck

            if stuck:
                cluster.add((ix, iy))
                order.append((ix, iy))

                sum_x += ix
                sum_y += iy
                sum_r2 += ix * ix + iy * iy
                n += 1

                d = math.hypot(ix, iy)
                if d > R_max:
                    R_max = d

                if n % 50 == 0 or n == n_particles:
                    rg2 = sum_r2 / n - (sum_x / n) ** 2 - (sum_y / n) ** 2
                    Rg_history.append(math.sqrt(max(rg2, 0.0)))
                break

            # No stick -- take random unit step.
            dx, dy = random.choice(neighbors)
            x = ix + dx
            y = iy + dy

    sites = np.array(order, dtype=int)
    return sites, Rg_history


def estimate_D(Rg_history, n_particles):
    """
    Fit log N = D log R_g + const, dropping the small-N transient.
    """
    Ns = np.arange(50, 50 * len(Rg_history), 50)[: len(Rg_history) - 1]
    Rgs = np.array(Rg_history[1:])
    mask = (Ns > n_particles // 10) & (Rgs > 0)
    if mask.sum() < 5:
        return float("nan")
    logN = np.log(Ns[mask])
    logR = np.log(Rgs[mask])
    D, _ = np.polyfit(logR, logN, 1)
    return float(D)


def plot_cluster(sites, Rg_history, n_particles):
    fig, ax1 = plt.subplots(figsize=(6, 5))
    order_idx = np.arange(len(sites))
    sc = ax1.scatter(sites[:, 0], sites[:, 1], c=order_idx,
                     cmap="viridis", s=2)
    ax1.set_aspect("equal")
    ax1.set_title(f"DLA cluster, N = {len(sites)}, s = {stick_prob}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(sc, ax=ax1, label="stick order")

    D = estimate_D(Rg_history, n_particles)
    plt.tight_layout()
    return fig, D


if __name__ == "__main__":
    N = 3000
    stick_prob = 0.3   # set to 1.0 for standard DLA; < 1 thickens branches
    sites, Rg_history = run_dla(N, stick_prob=stick_prob, seed=42)
    fig, D = plot_cluster(sites, Rg_history, N)
    print(f"Particles:  {N}")
    print(f"stick_prob: {stick_prob}")
    print(f"Estimated fractal dimension D = {D:.3f}  (Witten-Sander: ~1.71)")
    plt.show()