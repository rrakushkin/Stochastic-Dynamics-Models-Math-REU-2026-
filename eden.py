"""
Eden growth model simulation (2D square lattice).

Variants:
  A: pick a perimeter site (empty site adjacent to cluster) uniformly.
  B: pick a perimeter bond uniformly -> weight perimeter sites by # occupied neighbors.
  C: pick an occupied site uniformly, then a random neighbor; occupy if empty.

Renders an MP4 showing cluster growth, colored by birth time.

Requires: numpy, matplotlib, ffmpeg on PATH.
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse


# ---------- Eden model core ----------

class EdenModel:
    def __init__(self, grid_size=400, variant="A", seed=None):
        self.L = grid_size
        self.variant = variant
        self.rng = np.random.default_rng(seed)

        # birth_time[i,j] = step at which site (i,j) was occupied, -1 if empty
        self.birth_time = -np.ones((self.L, self.L), dtype=np.int32)

        cx = cy = self.L // 2
        self.cx, self.cy = cx, cy

        # Perimeter: set of empty sites adjacent to cluster
        self._perim = set()
        # For variant B we also need occupied-neighbor counts of perimeter sites
        self._occ_nbrs = {}  # site -> # occupied neighbors (for variant B)
        # For variant C: list of occupied sites
        self._occ_list = []

        self._step_count = 0
        self._seed_site(cx, cy)

    # --- internal helpers ---

    _NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def _in_bounds(self, x, y):
        return 0 <= x < self.L and 0 <= y < self.L

    def _is_occupied(self, x, y):
        return self.birth_time[x, y] >= 0

    def _seed_site(self, x, y):
        self.birth_time[x, y] = self._step_count
        self._step_count += 1
        self._occ_list.append((x, y))
        # add empty neighbors to perimeter
        for dx, dy in self._NBRS:
            nx, ny = x + dx, y + dy
            if self._in_bounds(nx, ny) and not self._is_occupied(nx, ny):
                self._perim.add((nx, ny))
                self._occ_nbrs[(nx, ny)] = self._occ_nbrs.get((nx, ny), 0) + 1

    def _occupy(self, x, y):
        """Mark (x,y) as occupied and update perimeter / weights."""
        self.birth_time[x, y] = self._step_count
        self._step_count += 1
        self._occ_list.append((x, y))
        # remove from perimeter
        self._perim.discard((x, y))
        self._occ_nbrs.pop((x, y), None)
        # update neighbors
        for dx, dy in self._NBRS:
            nx, ny = x + dx, y + dy
            if not self._in_bounds(nx, ny):
                continue
            if self._is_occupied(nx, ny):
                continue
            self._perim.add((nx, ny))
            self._occ_nbrs[(nx, ny)] = self._occ_nbrs.get((nx, ny), 0) + 1

    def _hit_boundary(self, x, y):
        return x == 0 or y == 0 or x == self.L - 1 or y == self.L - 1

    # --- one growth step for each variant ---

    def _step_A(self):
        if not self._perim:
            return False
        # uniform pick over perimeter sites
        site = self.rng.choice(list(self._perim))
        x, y = int(site[0]), int(site[1])
        if self._hit_boundary(x, y):
            return False
        self._occupy(x, y)
        return True

    def _step_B(self):
        if not self._perim:
            return False
        sites = list(self._perim)
        weights = np.array([self._occ_nbrs[s] for s in sites], dtype=np.float64)
        weights /= weights.sum()
        idx = self.rng.choice(len(sites), p=weights)
        x, y = sites[idx]
        if self._hit_boundary(x, y):
            return False
        self._occupy(x, y)
        return True

    def _step_C(self):
        # pick occupied site uniformly, then random neighbor; occupy if empty
        for _ in range(10):  # a few attempts in case of rejection
            i = self.rng.integers(len(self._occ_list))
            x, y = self._occ_list[i]
            dx, dy = self._NBRS[self.rng.integers(4)]
            nx, ny = x + dx, y + dy
            if not self._in_bounds(nx, ny):
                continue
            if self._hit_boundary(nx, ny):
                return False
            if not self._is_occupied(nx, ny):
                self._occupy(nx, ny)
                return True
        return True  # rejected attempt still counts

    def step(self, n=1):
        """Advance n growth steps. Returns number of successful additions."""
        step_fn = {"A": self._step_A, "B": self._step_B, "C": self._step_C}[self.variant]
        added = 0
        for _ in range(n):
            if not step_fn():
                break
            added += 1
        return added

    # --- observables ---

    @property
    def N(self):
        return self._step_count

    @property
    def perimeter_size(self):
        return len(self._perim)

    def radius_of_gyration(self):
        xs, ys = np.where(self.birth_time >= 0)
        if xs.size == 0:
            return 0.0
        mx, my = xs.mean(), ys.mean()
        return float(np.sqrt(((xs - mx) ** 2 + (ys - my) ** 2).mean()))


# ---------- Rendering ----------

def make_video(
    out_path="eden.mp4",
    grid_size=400,
    variant="A",
    total_steps=40_000,
    n_frames=200,
    fps=30,
    seed=0,
    dpi=120,
):
    model = EdenModel(grid_size=grid_size, variant=variant, seed=seed)

    # Quadratic schedule: targets[k] = cumulative # of sites by end of frame k.
    # N grows like frame^2, so radius (~sqrt(N)) grows linearly in time and
    # the first frames show 1, 4, 9, ... sites instead of jumping ahead.
    targets = (np.linspace(0, 1, n_frames + 1) ** 2 * total_steps).astype(int)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")

    # Colormap: viridis for occupied sites, white for empty (masked) cells.
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    def masked_disp():
        return ma.masked_where(model.birth_time < 0, model.birth_time)

    im = ax.imshow(
        masked_disp(),
        cmap=cmap,
        vmin=0,
        vmax=total_steps,
        interpolation="nearest",
        origin="lower",
    )
    title = ax.set_title(f"Eden {variant}  N=1  Rg=0.00")

    def update(frame_idx):
        needed = max(0, int(targets[frame_idx + 1]) - model.N)
        if needed > 0:
            model.step(needed)
        im.set_data(masked_disp())
        title.set_text(
            f"Eden {variant}   step {frame_idx+1}/{n_frames}   "
            f"N={model.N}   |∂C|={model.perimeter_size}   "
            f"Rg={model.radius_of_gyration():.2f}"
        )
        return [im, title]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False
    )

    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=4000,
        extra_args=["-pix_fmt", "yuv420p"],  # broad player compatibility
    )
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(
        f"saved {out_path}  |  variant={variant}  N={model.N}  "
        f"Rg={model.radius_of_gyration():.2f}  perim={model.perimeter_size}"
    )


# ---------- CLI ----------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="eden.mp4")
    p.add_argument("--variant", default="A", choices=["A", "B", "C"])
    p.add_argument("--grid", type=int, default=400)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    make_video(
        out_path=args.out,
        grid_size=args.grid,
        variant=args.variant,
        total_steps=args.steps,
        n_frames=args.frames,
        fps=args.fps,
        seed=args.seed,
    )
