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
"""

import math
import random
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


def run_dla(n_particles: int,
            stick_prob: float | Callable[[int], float] = 1.0,
            seed: int | None = None) -> tuple[np.ndarray, list[float]]:
    """
    Grow a DLA cluster of n_particles on the 2D square lattice.

    stick_prob may be either:
      - a float in (0, 1]: constant sticking probability, or
      - a callable f(n) -> probability, evaluated with the current cluster
        size n at each contact. E.g. stick_prob=lambda n: 1.0 / n gives a
        probability that decays as 1/N, so walkers penetrate deeper into the
        cluster as it grows.

    Returns:
        sites: (N, 2) integer array of particle coordinates, in stick order
        Rg_history: radius of gyration vs N (only computed every 50 sticks for speed)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Cluster as a set of (x, y) tuples for O(1) membership.
    cluster: set[tuple[int, int]] = {(0, 0)}
    order: list[tuple[int, int]] = [(0, 0)]

    R_max = 1.0          # current max radius from origin
    BUFFER = 5           # launch a bit outside R_max
    KILL_FACTOR = 2.0    # kill radius / launch radius

    # Running sums for R_g:  R_g^2 = (1/N) sum r_i^2  -  (1/N^2)(sum r_i)^2
    sum_x = 0.0
    sum_y = 0.0
    sum_r2 = 0.0
    n = 1  # seed counted

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    Rg_history: list[float] = [0.0]

    while n < n_particles:
        R_launch = R_max + BUFFER
        R_kill = KILL_FACTOR * R_launch

        # Launch on the launch circle.
        theta = random.uniform(0, 2 * math.pi)
        x = R_launch * math.cos(theta)
        y = R_launch * math.sin(theta)

        # Walk until stuck or killed.
        while True:
            r = math.hypot(x, y)

            if r > R_kill:
                # Far away -- relaunch instead of slowly diffusing back.
                # (Strictly: re-inject using the 2D Green's function;
                # this looser version is the standard cheap trick.)
                break

            if r > R_launch:
                # Big step toward / past launch circle in random direction.
                step = r - R_launch
                phi = random.uniform(0, 2 * math.pi)
                x += step * math.cos(phi)
                y += step * math.sin(phi)
                continue

            # Discretize current position to lattice node, then take a
            # unit lattice step.
            ix = int(round(x))
            iy = int(round(y))

            # If the walker has wandered onto an occupied site (possible when
            # the stick probability is low and it keeps moving), step off
            # rather than attaching a duplicate on top of an existing particle.
            if (ix, iy) in cluster:
                dx, dy = random.choice(neighbors)
                x = ix + dx
                y = iy + dy
                continue

            # Check if any neighbor is occupied -> stick (with prob p).
            # p is either a constant or, for the 1/N variant, a function of
            # the current cluster size n.
            p = float(stick_prob(n) if callable(stick_prob) else stick_prob)
            stuck = False
            for dx, dy in neighbors:
                if (ix + dx, iy + dy) in cluster:
                    if p >= 1.0 or random.random() < p:
                        stuck = True
                        break

            if stuck:
                cluster.add((ix, iy))
                order.append((ix, iy))

                # Update running sums and R_max.
                sum_x += ix
                sum_y += iy
                sum_r2 += ix * ix + iy * iy
                n += 1

                d = math.hypot(ix, iy)
                if d > R_max:
                    R_max = d

                # Record R_g every 50 sticks to keep overhead low.
                if n % 50 == 0 or n == n_particles:
                    rg2 = sum_r2 / n - (sum_x / n) ** 2 - (sum_y / n) ** 2
                    Rg_history.append(math.sqrt(max(rg2, 0.0)))
                break

            # No stick -- take random unit step on lattice.
            dx, dy = random.choice(neighbors)
            x = ix + dx
            y = iy + dy

    sites = np.array(order, dtype=int)
    return sites, Rg_history


def estimate_D(Rg_history: list[float], n_particles: int) -> float:
    """
    Fit log N = D log R_g + const, dropping the small-N transient.
    """
    Ns = np.arange(50, 50 * len(Rg_history), 50)[: len(Rg_history) - 1]
    Rgs = np.array(Rg_history[1:])
    # Drop early points (transient) and any zero R_g.
    mask = (Ns > n_particles // 10) & (Rgs > 0)
    if mask.sum() < 5:
        return float("nan")
    logN = np.log(Ns[mask])
    logR = np.log(Rgs[mask])
    D, _ = np.polyfit(logR, logN, 1)
    return float(D)


def plot_cluster(sites: np.ndarray, Rg_history: list[float], n_particles: int,
                 label: str = ""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if label:
        fig.suptitle(label, fontsize=13)

    # Render as filled lattice cells, colored by attachment order (shows
    # screening). imshow tiles each occupied node as a solid pixel, so the
    # cluster looks like a connected aggregate rather than scattered dots.
    xs, ys = sites[:, 0], sites[:, 1]
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    grid = np.full((ymax - ymin + 1, xmax - xmin + 1), np.nan)
    grid[ys - ymin, xs - xmin] = np.arange(len(sites))

    im = ax1.imshow(grid, origin="lower", cmap="viridis",
                    interpolation="nearest",
                    extent=[xmin - 0.5, xmax + 0.5, ymin - 0.5, ymax + 0.5])
    ax1.set_aspect("equal")
    ax1.set_title(f"DLA cluster, N = {len(sites)}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im, ax=ax1, label="stick order")

    # Log-log N vs R_g.
    Ns = np.arange(50, 50 * len(Rg_history), 50)[: len(Rg_history) - 1]
    Rgs = np.array(Rg_history[1:])
    mask = Rgs > 0
    ax2.loglog(Rgs[mask], Ns[mask], "o", ms=3)

    D = estimate_D(Rg_history, n_particles)
    if not math.isnan(D):
        # Plot fit line.
        Rfit = np.array([Rgs[mask].min(), Rgs[mask].max()])
        # Use last point to anchor the line.
        c = np.log(Ns[mask][-1]) - D * np.log(Rgs[mask][-1])
        Nfit = np.exp(D * np.log(Rfit) + c)
        ax2.loglog(Rfit, Nfit, "r--", label=f"fit: D = {D:.2f}")
    ax2.set_xlabel(r"$R_g$")
    ax2.set_ylabel("N")
    ax2.set_title(r"$N \sim R_g^{D}$")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    return fig, D


if __name__ == "__main__":
    # --- Standard DLA: stick on first contact (p = 1) ---
    N = 3000           # ~3k matches Fig. 4 of Sander & Ziff 1994
    sites, Rg_history = run_dla(N, stick_prob=1, seed=42)
    fig, D = plot_cluster(sites, Rg_history, N,
                          label="Standard DLA  (sticking probability p = 1)")
    print("Standard DLA (p = 1)")
    print(f"  Particles: {N}")
    print(f"  Estimated fractal dimension D = {D:.3f}  (Witten-Sander: ~1.71)")
    fig.savefig("DLA_w_DimPlot.png", dpi=150)

    plt.show()
