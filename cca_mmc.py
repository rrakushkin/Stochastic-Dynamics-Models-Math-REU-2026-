import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 4 nearest-neighbour displacements on the square lattice
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class CCA:
    def __init__(self, N=150, phi=0.12, Tstar=1.0, J=1.0,
                 periodic=False, seed=0):
        self.N = N
        self.J = J
        self.Tstar = Tstar
        self.periodic = periodic
        self.rng = np.random.default_rng(seed)

        self.M = int(round(phi * N * N))          # number of particles
        self.phi = self.M / (N * N)

        # lattice[i, j] = 0 (empty) or positive cluster id
        self.lattice = np.zeros((N, N), dtype=np.int64)

        # choose M distinct sites
        flat = self.rng.choice(N * N, size=self.M, replace=False)
        rows, cols = np.divmod(flat, N)

        # particle positions; index = particle id, value = (i, j)
        self.pos = np.stack([rows, cols], axis=1).astype(np.int64)

        # cluster registry: id -> set of (i, j) sites
        self.clusters = {}
        for p in range(self.M):
            cid = p + 1                            # ids are positive ints
            i, j = int(rows[p]), int(cols[p])
            self.lattice[i, j] = cid
            self.clusters[cid] = {(i, j)}

        self.sweep = 0

    # ---- geometry helpers --------------------------------------------------
    def _wrap(self, i, j):
        """Return (i, j, valid). valid=False means off-lattice (hard wall)."""
        if self.periodic:
            return i % self.N, j % self.N, True
        if 0 <= i < self.N and 0 <= j < self.N:
            return i, j, True
        return i, j, False

    def _n_same(self, i, j, cid, exclude):
        """Same-cluster occupied neighbours of (i,j), excluding 'exclude'."""
        n = 0
        for di, dj in NEIGHBORS:
            ni, nj, valid = self._wrap(i + di, j + dj)
            if not valid:
                continue
            if (ni, nj) == exclude:
                continue
            if self.lattice[ni, nj] == cid:
                n += 1
        return n

    # ---- cluster merge (weighted union) ------------------------------------
    def _merge(self, a, b):
        """Merge clusters a and b, smaller relabelled into larger. Return survivor id."""
        if a == b:
            return a
        if len(self.clusters[a]) < len(self.clusters[b]):
            a, b = b, a                            # a is now the larger
        for (si, sj) in self.clusters[b]:
            self.lattice[si, sj] = a
        self.clusters[a].update(self.clusters[b])
        del self.clusters[b]
        return a

    # ---- one elementary MMC step -------------------------------------------
    def step(self):
        p = self.rng.integers(self.M)              # uniform random PARTICLE
        i, j = self.pos[p]
        cid = self.lattice[i, j]

        di, dj = NEIGHBORS[self.rng.integers(4)]   # random direction
        ti, tj, valid = self._wrap(i + di, j + dj)
        if not valid:
            return                                 # off-lattice: reject

        target = self.lattice[ti, tj]

        if target != 0:
            if target != cid:                      # contact with foreign cluster -> stick
                self._merge(cid, target)
            return                                 # occupied (same or merged): no hop

        # target empty: Metropolis on the hop
        n_old = self._n_same(i, j, cid, exclude=(ti, tj))
        n_new = self._n_same(ti, tj, cid, exclude=(i, j))
        dH = -self.J * (n_new - n_old)             # = J*(n_old - n_new)

        if dH > 0:
            # accept with prob exp(-dH/T*) ; exponent uses integer (n_new-n_old)
            if self.rng.random() >= np.exp((n_new - n_old) / self.Tstar):
                return                             # rejected

        # ---- accept the hop ----
        self.lattice[i, j] = 0
        self.lattice[ti, tj] = cid
        self.pos[p] = (ti, tj)
        self.clusters[cid].discard((i, j))
        self.clusters[cid].add((ti, tj))

        # post-move merge: any foreign cluster now adjacent to (ti, tj)
        for ddi, ddj in NEIGHBORS:
            ni, nj, ok = self._wrap(ti + ddi, tj + ddj)
            if not ok:
                continue
            fid = self.lattice[ni, nj]
            if fid != 0 and fid != cid:
                cid = self._merge(cid, fid)        # survivor may change id

    def sweep_once(self):
        """One sweep = M attempted moves (1 MCS)."""
        for _ in range(self.M):
            self.step()
        self.sweep += 1

    # ---- observables -------------------------------------------------------
    def n_clusters(self):
        return len(self.clusters)

    def mean_size(self):
        return self.M / len(self.clusters)

    def max_size(self):
        return max(len(s) for s in self.clusters.values())

    def radius_of_gyration(self, sites):
        """R_g = sqrt( (1/s) * sum_i |r_i - r_cm|^2 )."""
        pts = np.array(list(sites), dtype=float)
        cm = pts.mean(axis=0)
        return np.sqrt(((pts - cm) ** 2).sum(axis=1).mean())

    def size_distribution(self):
        sizes = np.array([len(s) for s in self.clusters.values()])
        return sizes

    def fractal_dim_Rg(self, min_size=8):
        """Fit s = a * R_g^Df  ->  log Rg vs log s, slope = 1/Df."""
        s_list, rg_list = [], []
        for sites in self.clusters.values():
            s = len(sites)
            if s >= min_size:
                s_list.append(s)
                rg_list.append(self.radius_of_gyration(sites))
        if len(s_list) < 3:
            return np.nan, (np.array(s_list), np.array(rg_list))
        s_arr = np.array(s_list, float)
        rg_arr = np.array(rg_list, float)
        slope, _ = np.polyfit(np.log(s_arr), np.log(rg_arr), 1)
        Df = 1.0 / slope
        return Df, (s_arr, rg_arr)

    def box_count_Df(self, sites):
        """Box-counting dimension of one cluster's site set."""
        pts = np.array(list(sites))
        if len(pts) < 8:
            return np.nan, None
        mins = pts.min(axis=0)
        span = (pts.max(axis=0) - mins).max() + 1
        eps_vals, counts = [], []
        eps = 1
        while eps <= span:
            boxes = set()
            for (x, y) in pts - mins:
                boxes.add((x // eps, y // eps))
            eps_vals.append(eps)
            counts.append(len(boxes))
            eps *= 2
        eps_vals = np.array(eps_vals, float)
        counts = np.array(counts, float)
        good = counts > 1
        if good.sum() < 2:
            return np.nan, None
        slope, _ = np.polyfit(np.log(1.0 / eps_vals[good]),
                              np.log(counts[good]), 1)
        return slope, (eps_vals, counts)


# ---------------------------------------------------------------------------
def run(N=150, phi=0.12, Tstar=1.0, sweeps=250, seed=0, periodic=False,
        record_every=1, dist_every=10):
    sim = CCA(N=N, phi=phi, Tstar=Tstar, seed=seed, periodic=periodic)

    t, Nc, sbar, smax = [], [], [], []
    snapshots = []          # list of dicts: sweep, lattice, sizes, rgs
    for k in range(sweeps):
        sim.sweep_once()
        if k % record_every == 0:
            t.append(sim.sweep)
            Nc.append(sim.n_clusters())
            sbar.append(sim.mean_size())
            smax.append(sim.max_size())
        if dist_every and k % dist_every == 0:
            sizes, rgs = [], []
            for sites in sim.clusters.values():
                sizes.append(len(sites))
                rgs.append(sim.radius_of_gyration(sites))
            snapshots.append(dict(sweep=sim.sweep,
                                  lattice=sim.lattice.copy(),
                                  sizes=np.array(sizes),
                                  rgs=np.array(rgs)))

    out = dict(sim=sim,
               t=np.array(t), Nc=np.array(Nc),
               sbar=np.array(sbar), smax=np.array(smax),
               snapshots=snapshots)
    return out


def _fit_Df_from_snapshot(sizes, rgs, min_size=6):
    """s ~ R_g^Df  ->  log Rg = (1/Df) log s + c. Returns Df, (s, Rg) used."""
    mask = sizes >= min_size
    s_arr, rg_arr = sizes[mask].astype(float), rgs[mask].astype(float)
    if len(s_arr) < 3:
        return np.nan, (s_arr, rg_arr)
    slope, _ = np.polyfit(np.log(s_arr), np.log(rg_arr), 1)
    return 1.0 / slope, (s_arr, rg_arr)


def _best_snapshot(snapshots, min_size=6):
    """Pick the snapshot with the most clusters of size >= min_size (richest scaling range)."""
    best, best_n = None, -1
    for snp in snapshots:
        n = int((snp["sizes"] >= min_size).sum())
        if n > best_n:
            best, best_n = snp, n
    return best


def plot_cluster(sites, ax, title=None):
    pts = np.array(list(sites), dtype=int)
    if pts.size == 0:
        ax.text(0.5, 0.5, "empty cluster",
                ha="center", va="center", fontsize=12)
        return
    ax.scatter(pts[:, 1], pts[:, 0], s=4, c="black")
    ax.set_aspect("equal")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title(title or f"Cluster (s={len(sites)})")
    ax.invert_yaxis()


def make_figure(out, fname="cca_results.png"):
    sim = out["sim"]
    snp = _best_snapshot(out["snapshots"])
    fig, ax = plt.subplots(3, 2, figsize=(12, 14))

    # (1) kinetics: N_c(t), mean size, largest cluster
    a = ax[0, 0]
    a.loglog(out["t"], out["Nc"], label=r"$N_c(t)$")
    a.loglog(out["t"], out["sbar"], label=r"$\langle s\rangle = M/N_c$")
    a.loglog(out["t"], out["smax"], label=r"$s_{\max}$")
    a.set_xlabel("sweeps (MCS)")
    a.set_ylabel("count / size")
    a.set_title(f"Kinetics  (N={sim.N}, M={sim.M}, T*={sim.Tstar})")
    a.legend(); a.grid(True, which="both", alpha=.3)

    # (2) polydisperse configuration at the snapshot used for scaling
    a = ax[0, 1]
    lab = snp["lattice"]
    ids = np.unique(lab)
    disp = np.zeros_like(lab)
    k = 1
    for cid in ids:
        if cid == 0:
            continue
        disp[lab == cid] = (k % 19) + 1
        k += 1
    disp = np.ma.masked_where(disp == 0, disp)
    a.imshow(disp, cmap="tab20", interpolation="nearest")
    a.set_title(f"Config at sweep {snp['sweep']}: "
                f"{(snp['sizes']>0).sum()} clusters")
    a.set_xticks([]); a.set_yticks([])

    # (3) R_g vs s on that snapshot, slope = 1/Df
    a = ax[1, 0]
    Df, (s_arr, rg_arr) = _fit_Df_from_snapshot(snp["sizes"], snp["rgs"])
    if len(s_arr) >= 3:
        a.loglog(s_arr, rg_arr, "o", ms=4, alpha=.6)
        ss = np.array([s_arr.min(), s_arr.max()])
        slope = 1.0 / Df
        c = np.log(rg_arr).mean() - slope * np.log(s_arr).mean()
        a.loglog(ss, np.exp(c) * ss ** slope, "r-",
                 label=fr"$R_g\sim s^{{1/D_f}}$, $D_f$={Df:.2f}")
        a.legend()
    a.set_xlabel("cluster size  s")
    a.set_ylabel(r"$R_g$")
    a.set_title(rf"Fractal scaling (sweep {snp['sweep']})")
    a.grid(True, which="both", alpha=.3)

    # (4) box counting on the largest cluster of the FINAL state
    big = max(sim.clusters.values(), key=len)

    a = ax[1, 1]
    Df_box, bc = sim.box_count_Df(big)
    if bc is not None:
        eps_vals, counts = bc
        a.loglog(1.0 / eps_vals, counts, "s-", ms=5,
                 label=fr"$D_f^{{box}}$={Df_box:.2f}")
        a.legend()
    a.set_xlabel(r"$1/\epsilon$")
    a.set_ylabel(r"box count $N(\epsilon)$")
    a.set_title(f"Box-counting, largest final cluster (s={len(big)})")
    a.grid(True, which="both", alpha=.3)

    # (5) scatter plot of the largest final cluster itself
    a = ax[2, 0]
    plot_cluster(big, a, title=f"Largest final cluster (s={len(big)})")

    # (6) binary mask of the largest final cluster
    a = ax[2, 1]
    pts = np.array(list(big), dtype=int)
    mins = pts.min(axis=0)
    mask = np.zeros((pts[:, 0].max() - mins[0] + 1,
                     pts[:, 1].max() - mins[1] + 1), dtype=int)
    mask[pts[:, 0] - mins[0], pts[:, 1] - mins[1]] = 1
    a.imshow(mask, cmap="Greys", interpolation="nearest", origin="upper")
    a.set_title("Largest cluster binary mask")
    a.set_xticks([]); a.set_yticks([])

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    return Df, Df_box


if __name__ == "__main__":
    import time
    t0 = time.time()
    # 250 sweeps: system is still polydisperse (many clusters of varied size),
    # which is what populates the R_g-vs-s scaling fit. Run longer to coalesce.
    out = run(N=150, phi=0.12, Tstar=1.0, sweeps=250, seed=1)
    Df, Df_box = make_figure(out, "cca_results.png")
    sim = out["sim"]
    print(f"runtime          : {time.time()-t0:.1f} s")
    print(f"particles M      : {sim.M}  (phi={sim.phi:.3f})")
    print(f"final N_clusters : {sim.n_clusters()}")
    print(f"final mean size  : {sim.mean_size():.2f}")
    print(f"final s_max      : {sim.max_size()}")
    print(f"D_f (Rg vs s)    : {Df:.3f}")
    print(f"D_f (box, big)   : {Df_box:.3f}")
