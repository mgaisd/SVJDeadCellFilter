#!/usr/bin/env python3
"""
FilterEtaPhiMaps.py

Loads QCD MC, data, and signal datasets, applies the dead-cell filter from the
outlier_centers txt files, and produces:

    1. DeadCellStudy-style normalised eta-phi maps for subleading jets as
         separate single-panel plots (one for QCD Simulation and one for CMS Data)
         for each period:
             • 2017
             • 2018 preHEM
             • 2018 postHEM  (HEM veto rectangle also shown as white)
         For both 2018 periods the combined filter (union of preHEM + HEM outlier
         centres) is used. Dead-cell vetoed bins are drawn as white circles; the
         HEM rectangle gets a red border for visual distinction.

  2. Dead-cell filter efficiency (fraction of events whose subleading jet
     survives the circular veto) for QCD MC and all signal samples.
     Saved as CSV files under OUTPUT_DIR/{year}/efficiencies_{year}.csv.

Usage:
    python FilterEtaPhiMaps.py                 # both years (default)
    python FilterEtaPhiMaps.py --years 2017
    python FilterEtaPhiMaps.py --years 2018
    python FilterEtaPhiMaps.py --years 2016    # data only, from DeadCellStudy ntuple cache

Requirements: uproot, numpy, matplotlib, mplhep, awkward, tqdm
"""

import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplhep as hep
import awkward as ak
from tqdm import tqdm

plt.style.use(hep.style.CMS)
plt.rcParams.update({
    "font.size":       14,
    "axes.labelsize":  16,
    "axes.titlesize":  16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})

# ─────────────────────────────────────────────────────────────────────────────
# Binning  (must match DeadCellStudy.py)
# ─────────────────────────────────────────────────────────────────────────────
ETA_BINS    = np.linspace(-2.4,   2.4, 50)
PHI_BINS    = np.linspace(-np.pi, np.pi, 50)
ETA_CENTERS = 0.5 * (ETA_BINS[:-1] + ETA_BINS[1:])
PHI_CENTERS = 0.5 * (PHI_BINS[:-1] + PHI_BINS[1:])

# Dead-cell veto ΔR² threshold  (must match DeadCellStudy.py::DEAD_CELL_RAD)
DEAD_CELL_RAD = 0.028816 * 0.35   # ≈ 0.010086



# HEM-affected region boundaries (2018)
HEM_ETA_MIN, HEM_ETA_MAX = -3.05, -1.35
HEM_PHI_MIN, HEM_PHI_MAX = -1.62, -0.82
HEM_RUN_THRESHOLD        = 319077   # data runs ≥ this are HEM-affected

JET_ETA_BRANCH = "Jet_eta"
JET_PHI_BRANCH = "Jet_phi"

XROOTD_SERVER = "cmsdcache-kit-disk.gridka.de:1094"
FILE_WORKERS  = 8

OUTPUT_DIR = "plots_filter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────────────────────────────────────────
QCD_DATASETS = [
    "QCD_HT300to500",
    "QCD_HT500to700",
    "QCD_HT700to1000",
    "QCD_HT1000to1500",
    "QCD_HT1500to2000",
    "QCD_HT2000toInf",
]

YEAR_CFG = {
    # 2016: data only (no MC), read from the per-file cache that
    # DeadCellStudy.py writes when deriving the filter from the ntuples
    "2016": {
        "lumi":         36.3,   # fb^-1 -- TODO verify scouting lumi with brilcalc
        "filter_txt":   "plots_deadcell/2016/outlier_centers_2016.txt",
        "ntuple_cache": "cache_ntuples/2016",
        "data_only":    True,
        "hem":          False,
    },
    "2017": {
        "lumi":      41.53,
        "base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_JEC/2017/s_channel_scouting_pre_selection/nominal",
        "sig_base":  "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims/2017/s_channel_scouting_pre_selection",
        "filter_txt": "plots_deadcell/2017/outlier_centers_2017.txt",
        "data_datasets": ["Run2017C", "Run2017D", "Run2017E", "Run2017F"],
        "hem": False,
    },
    "2018": {
        "lumi":      59.74,
        "lumi_pre":  21.1,
        "lumi_hem":  38.64,
        "base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_JEC/2018/s_channel_scouting_pre_selection/nominal",
        "sig_base":  "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims/2018/s_channel_scouting_pre_selection",
        # combined filter is used for both periods in the analysis
        "filter_txt":        "plots_deadcell/2018/outlier_centers_2018.txt",
        "filter_txt_preHEM": "plots_deadcell/2018/outlier_centers_2018_preHEM.txt",
        "filter_txt_HEM":    "plots_deadcell/2018/outlier_centers_2018_HEM.txt",
        "data_datasets": ["Run2018A", "Run2018B", "Run2018C", "Run2018D"],
        "hem": True,
    },
}

# Signal samples: { label: [dataset_folder, hex_color] }
SIG_DICT = {
    "mMed-500":  ["s-channel_mMed-500_mDark-20_rinv-0.3",  "#1f77b4"],
    "mMed-600":  ["s-channel_mMed-600_mDark-20_rinv-0.3",  "#ff7f0e"],
    "mMed-700":  ["s-channel_mMed-700_mDark-20_rinv-0.3",  "#2ca02c"],
    "mMed-800":  ["s-channel_mMed-800_mDark-20_rinv-0.3",  "#d62728"],
    "mMed-900":  ["s-channel_mMed-900_mDark-20_rinv-0.3",  "#9467bd"],
    "mMed-1000": ["s-channel_mMed-1000_mDark-20_rinv-0.3", "#8c564b"],
    "mMed-1100": ["s-channel_mMed-1100_mDark-20_rinv-0.3", "#e377c2"],
    "mMed-1200": ["s-channel_mMed-1200_mDark-20_rinv-0.3", "#7f7f7f"],
    "mMed-1300": ["s-channel_mMed-1300_mDark-20_rinv-0.3", "#bcbd22"],
    "mMed-1400": ["s-channel_mMed-1400_mDark-20_rinv-0.3", "#17becf"],
    "mMed-1500": ["s-channel_mMed-1500_mDark-20_rinv-0.3", "#aec7e8"],
    "rinv-0.3":  ["s-channel_mMed-1000_mDark-20_rinv-0.3", "#ffbb78"],
    "rinv-0.5":  ["s-channel_mMed-1000_mDark-20_rinv-0.5", "#98df8a"],
    "rinv-0.7":  ["s-channel_mMed-1000_mDark-20_rinv-0.7", "#ff9896"],
}


# ─────────────────────────────────────────────────────────────────────────────
# File listing
# ─────────────────────────────────────────────────────────────────────────────

def get_file_list(dataset_path: str) -> list:
    """List ROOT files in *dataset_path* via xrdfs."""
    lfn_prefix  = f"root://{XROOTD_SERVER}/"
    remote_path = dataset_path.replace(lfn_prefix, "")
    cmd = ["xrdfs", XROOTD_SERVER, "ls", remote_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return [
            f"{lfn_prefix}{line.strip()}"
            for line in result.stdout.splitlines()
            if line.strip().endswith(".root")
        ]
    except Exception as exc:
        print(f"  Warning: could not list {dataset_path}: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Single-file readers
# ─────────────────────────────────────────────────────────────────────────────

def _read_mc_file(file_path: str, lumi: float) -> dict | None:
    """Read one MC ROOT file.

    Returns a dict with eta_sub, phi_sub, weights (weighted by lumi × xsec
    × prefire), restricted to events with ≥ 2 jets.  Returns None on failure.
    """
    name = Path(file_path).name
    try:
        with uproot.open(file_path, timeout=300) as f:
            try:
                n_initial = f["CutFlow"]["Initial"].array(library="np")[0]
                xsec      = f["Metadata"]["GenCrossSection"].array(library="np")[0]
            except Exception as exc:
                tqdm.write(f"    Warning: normalisation info missing in {name}: {exc}")
                return None
            try:
                arrs = f["Events"].arrays(
                    [JET_ETA_BRANCH, JET_PHI_BRANCH, "prefire"], library="ak"
                )
            except Exception as exc:
                tqdm.write(f"    Warning: Events unreadable in {name}: {exc}")
                return None

        has_sub = ak.num(arrs[JET_ETA_BRANCH]) >= 2
        # weight = xsec * lumi * 1e3 * prefire  (divide by n_initial_total later)
        partial_w = xsec * lumi * 1e3 * ak.to_numpy(arrs["prefire"][has_sub])
        eta = ak.to_numpy(arrs[JET_ETA_BRANCH][has_sub, 1])
        phi = ak.to_numpy(arrs[JET_PHI_BRANCH][has_sub, 1])
        return {"eta_sub": eta, "phi_sub": phi, "weights": partial_w, "n_initial": n_initial}

    except Exception as exc:
        tqdm.write(f"    Error opening {name}: {exc}")
        return None


def _read_data_file(file_path: str, hem_split: bool = False) -> dict | tuple:
    """Read one data ROOT file.

    If *hem_split* is False returns a dict (or None on failure).
    If *hem_split* is True returns a (pre_dict_or_None, hem_dict_or_None) tuple.
    Weights are all 1.
    """
    name = Path(file_path).name
    try:
        branches = [JET_ETA_BRANCH, JET_PHI_BRANCH]
        if hem_split:
            branches.append("run")

        with uproot.open(file_path, timeout=300) as f:
            try:
                arrs = f["Events"].arrays(branches, library="ak")
            except Exception as exc:
                tqdm.write(f"    Warning: Events unreadable in {name}: {exc}")
                return (None, None) if hem_split else None

        def _extract(a):
            has_sub = ak.num(a[JET_ETA_BRANCH]) >= 2
            n = int(ak.sum(has_sub))
            return {
                "eta_sub": ak.to_numpy(a[JET_ETA_BRANCH][has_sub, 1]),
                "phi_sub": ak.to_numpy(a[JET_PHI_BRANCH][has_sub, 1]),
                "weights": np.ones(n),
            }

        if hem_split:
            is_pre = arrs["run"] < HEM_RUN_THRESHOLD
            return _extract(arrs[is_pre]), _extract(arrs[~is_pre])

        return _extract(arrs)

    except Exception as exc:
        tqdm.write(f"    Error opening {name}: {exc}")
        return (None, None) if hem_split else None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level readers
# ─────────────────────────────────────────────────────────────────────────────

def _concat_dicts(parts: list) -> dict | None:
    """Concatenate a list of {"eta_sub", "phi_sub", "weights"} dicts."""
    good = [p for p in parts if p is not None]
    if not good:
        return None
    return {
        "eta_sub": np.concatenate([p["eta_sub"] for p in good]),
        "phi_sub": np.concatenate([p["phi_sub"] for p in good]),
        "weights": np.concatenate([p["weights"]  for p in good]),
    }


def read_mc_dataset(label: str, dataset_path: str, lumi: float,
                    show_progress: bool = True) -> dict | None:
    """Read all ROOT files for one MC dataset."""
    files = get_file_list(dataset_path)
    if not files:
        return None

    results = [None] * len(files)
    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
        fmap = {pool.submit(_read_mc_file, fp, lumi): i for i, fp in enumerate(files)}
        it   = as_completed(fmap)
        if show_progress:
            it = tqdm(it, total=len(files), desc=f"    {label}", unit="file",
                      dynamic_ncols=True)
        for fut in it:
            results[fmap[fut]] = fut.result()

    good = [r for r in results if r is not None]
    if not good:
        return None
    n_initial_total = sum(r["n_initial"] for r in good)
    d = _concat_dicts(good)
    # divide the concatenated weight array by total n_initial in one operation
    d["weights"] /= n_initial_total
    return d


def read_all_qcd(base_path: str, lumi: float) -> dict | None:
    """Read and merge all QCD HT-binned MC datasets."""
    print("  Reading QCD MC ...")
    parts = []
    for dname in QCD_DATASETS:
        d = read_mc_dataset(dname, f"{base_path}/{dname}", lumi)
        status = f"{len(d['eta_sub'])} events" if d is not None else "FAILED"
        print(f"    {dname}: {status}")
        if d is not None:
            parts.append(d)
    return _concat_dicts(parts)


def read_all_data(base_path: str, data_datasets: list,
                  hem_split: bool = False):
    """Read all data datasets.

    Returns:
        hem_split=False  ->  single dict (or None)
        hem_split=True   ->  (pre_dict_or_None, hem_dict_or_None)
    """
    print("  Reading data ...")
    pre_parts, hem_parts, nosplit_parts = [], [], []

    for dname in data_datasets:
        files = get_file_list(f"{base_path}/{dname}")
        if not files:
            print(f"    {dname}: no files")
            continue

        results = [None] * len(files)
        with ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
            fmap = {
                pool.submit(_read_data_file, fp, hem_split): i
                for i, fp in enumerate(files)
            }
            with tqdm(total=len(files), desc=f"    {dname}", unit="file",
                      dynamic_ncols=True) as pbar:
                for fut in as_completed(fmap):
                    results[fmap[fut]] = fut.result()
                    pbar.update(1)

        if hem_split:
            for r in results:
                p, h = r   # always a 2-tuple for hem_split=True
                if p is not None:
                    pre_parts.append(p)
                if h is not None:
                    hem_parts.append(h)
        else:
            for r in results:
                if r is not None:
                    nosplit_parts.append(r)

        print(f"    {dname}: ok ({len(files)} files)")

    if hem_split:
        return _concat_dicts(pre_parts), _concat_dicts(hem_parts)
    return _concat_dicts(nosplit_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Filter utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_filter_centers(txt_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse an outlier_centers txt file and return (eta_arr, phi_arr)."""
    etas, phis = [], []
    with open(txt_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            etas.append(float(parts[0]))
            phis.append(float(parts[1]))
    return np.array(etas), np.array(phis)


def build_veto_bin_mask(filter_eta: np.ndarray, filter_phi: np.ndarray,
                        veto_rad: float = DEAD_CELL_RAD) -> np.ndarray:
    """Boolean mask (n_eta, n_phi): True for bins that contain a filter centre.

    Each filter centre is snapped to its nearest bin (argmin distance), so
    exactly the bins identified as outliers are masked — no radius expansion.
    *veto_rad* is kept as a parameter for API compatibility but is not used
    for the visualisation mask.
    """
    mask = np.zeros((len(ETA_CENTERS), len(PHI_CENTERS)), dtype=bool)
    for ec, pc in zip(filter_eta, filter_phi):
        i = int(np.argmin(np.abs(ETA_CENTERS - ec)))
        j = int(np.argmin(np.abs(PHI_CENTERS - pc)))
        mask[i, j] = True
    return mask



# ─────────────────────────────────────────────────────────────────────────────
# Normalisation  (identical to DeadCellStudy.py::normalize_by_eta_bin)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_by_eta_bin(h2d: np.ndarray) -> np.ndarray:
    """Per-eta-bin normalisation: divide by max, subtract mean, clip negatives."""
    h_norm = np.zeros_like(h2d)
    for i in range(h2d.shape[0]):
        row     = h2d[i, :].copy()
        max_val = np.max(row)
        if max_val > 0:
            row /= max_val
        row        -= np.mean(row)
        h_norm[i, :] = np.maximum(0.0, row)
    return h_norm


def build_norm_map(data: dict) -> np.ndarray:
    """Weighted 2-D eta-phi histogram normalised per eta bin."""
    h2d, _, _ = np.histogram2d(
        data["eta_sub"], data["phi_sub"],
        bins=[ETA_BINS, PHI_BINS],
        weights=data["weights"],
    )
    return _normalize_by_eta_bin(h2d)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_filtered_map(
    h_map:        np.ndarray,
    filter_eta:   np.ndarray,
    filter_phi:   np.ndarray,
    period_label: str,
    lumi:         float,
    out_stem:     str,
    is_data:      bool = False,
    mark_hem:     bool = False,
    veto_rad:     float = DEAD_CELL_RAD,
):
    """Single-panel DeadCellStudy-style normalized eta-phi map.

    Dead-cell veto positions are drawn as white circles. When *mark_hem* is
    True the HEM rectangle is filled white with a red border.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    pcm = ax.pcolormesh(ETA_BINS, PHI_BINS, h_map.T, cmap="viridis", rasterized=True)
    fig.colorbar(pcm, ax=ax, pad=0.01).set_label("Normalized enhancement")

    # HEM veto: white-filled rectangle with red border (postHEM only)
    if mark_hem:
        hem_fill = mpatches.Rectangle(
            (HEM_ETA_MIN, HEM_PHI_MIN),
            HEM_ETA_MAX - HEM_ETA_MIN,
            HEM_PHI_MAX - HEM_PHI_MIN,
            linewidth=0, facecolor="white", zorder=5,
        )
        hem_border = mpatches.Rectangle(
            (HEM_ETA_MIN, HEM_PHI_MIN),
            HEM_ETA_MAX - HEM_ETA_MIN,
            HEM_PHI_MAX - HEM_PHI_MIN,
            linewidth=2, edgecolor="red", facecolor="none", zorder=6,
        )
        ax.add_patch(hem_fill)
        ax.add_patch(hem_border)
        ax.text(
            0.5 * (HEM_ETA_MIN + HEM_ETA_MAX), HEM_PHI_MAX + 0.05,
            "HEM veto", ha="center", va="bottom", color="red", fontsize=11,
            transform=ax.transData, zorder=7,
        )

    veto_r = np.sqrt(veto_rad)
    for ec, pc in zip(filter_eta, filter_phi):
        ax.add_patch(mpatches.Circle(
            (ec, pc), radius=veto_r,
            facecolor="white", edgecolor="none", zorder=8,
        ))

    ax.set_xlabel(r"AK4 Jet 2 $\eta$", ha="right", x=1.0)
    ax.set_ylabel(r"AK4 Jet 2 $\phi$", ha="right", y=1.0)
    ax.set_xlim(ETA_BINS[0], ETA_BINS[-1])
    ax.set_ylim(PHI_BINS[0], PHI_BINS[-1])

    title = (r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Data)}$" if is_data
             else r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Simulation)}$")
    ax.set_title(title, loc="left", pad=6)

    year_label = period_label
    ax.text(1, 1, rf"{lumi:.1f} fb$^{{-1}}$ ({year_label}, 13 TeV)",
            transform=ax.transAxes, ha="right", va="bottom")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = f"{out_stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Efficiency computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficiency(
    data:        dict,
    filter_eta:  np.ndarray,
    filter_phi:  np.ndarray,
    veto_rad:    float = DEAD_CELL_RAD,
) -> tuple[float, float, float]:
    """Return (pre_yield, post_yield, efficiency) for events with ≥2 jets.

    An event fails the veto if its subleading jet lies within *veto_rad* (ΔR²)
    of any filter centre.
    """
    if data is None or len(data["weights"]) == 0:
        return 0.0, 0.0, float("nan")

    pre = float(np.sum(data["weights"]))
    if pre <= 0:
        return 0.0, 0.0, float("nan")

    if len(filter_eta) == 0:
        return pre, pre, 1.0

    eta = data["eta_sub"][:, None]   # (N, 1)
    phi = data["phi_sub"][:, None]   # (N, 1)

    deta = eta - filter_eta[None, :]
    dphi = phi - filter_phi[None, :]
    dphi = np.where(dphi >  np.pi, dphi - 2.0 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2.0 * np.pi, dphi)

    passes = ~np.any(deta**2 + dphi**2 < veto_rad, axis=1)
    post   = float(np.sum(data["weights"][passes]))
    return pre, post, post / pre


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _load_ntuple_cache(cache_dir: str) -> dict | None:
    """Subleading good AK4 jets of all selected events in DeadCellStudy's ntuple cache."""
    import glob
    files = sorted(glob.glob(os.path.join(cache_dir, "*", "*.npz")))
    if not files:
        return None
    eta, phi = [], []
    for f in files:
        with np.load(f) as d:
            m = d["n_good"] >= 2
            eta.append(d["eta1"][m].astype(np.float64))
            phi.append(d["phi1"][m].astype(np.float64))
    print(f"  Loaded {len(files)} cached files from {cache_dir}")
    eta, phi = np.concatenate(eta), np.concatenate(phi)
    return {"eta_sub": eta, "phi_sub": phi, "weights": np.ones(len(eta))}


def _run_data_only_year(year: str, cfg: dict, out_dir: str):
    """Data-only year (2016): filtered map and data veto efficiency."""
    data = _load_ntuple_cache(cfg["ntuple_cache"])
    if data is None:
        print(f"  No cached ntuple data in {cfg['ntuple_cache']} -- "
              f"run DeadCellStudy.py --years {year} first.")
        return
    filter_eta, filter_phi = load_filter_centers(cfg["filter_txt"])
    print(f"  Filter: {len(filter_eta)} centres loaded from {cfg['filter_txt']}")

    plot_filtered_map(
        build_norm_map(data), filter_eta, filter_phi,
        period_label=year,
        lumi=cfg["lumi"],
        out_stem=os.path.join(out_dir, f"etaphi_filter_{year}_data"),
        is_data=True,
        mark_hem=False,
    )

    pre, post, eff = compute_efficiency(data, filter_eta, filter_phi)
    print(f"    Data  pre={pre:.3e}  post={post:.3e}  eff={eff:.4f}")
    csv_path = os.path.join(out_dir, f"efficiencies_{year}.csv")
    with open(csv_path, "w") as fh:
        fh.write("sample,pre_yield,post_yield,efficiency\n")
        fh.write(f"Data,{pre:.6e},{post:.6e},{eff:.6f}\n")
    print(f"\n  Efficiencies saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply dead-cell filter from txt, plot eta-phi maps, "
                    "and compute efficiencies for QCD and signal.",
        epilog="""
Examples:
  python FilterEtaPhiMaps.py                 # 2017 and 2018
  python FilterEtaPhiMaps.py --years 2017
  python FilterEtaPhiMaps.py --years 2018

Output  (plots_filter/{year}/):
    etaphi_filter_{period}_{mc|data}.{png,pdf}   – separate maps with white veto regions
  efficiencies_{year}.csv            – QCD and signal filter efficiencies
        """
    )
    parser.add_argument("--years", nargs="+", default=["2017", "2018"],
                        help="Year(s) to process (default: 2017 2018)")
    args = parser.parse_args()

    for year in args.years:
        if year not in YEAR_CFG:
            print(f"\nWarning: year {year} not configured, skipping.")
            continue

        cfg     = YEAR_CFG[year]
        out_dir = os.path.join(OUTPUT_DIR, year)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Year {year}")
        print(f"{'='*60}")

        if cfg.get("data_only"):
            _run_data_only_year(year, cfg, out_dir)
            continue

        # ── Read QCD MC ──────────────────────────────────────────────────────
        qcd = read_all_qcd(cfg["base_path"], cfg["lumi"])
        if qcd is None:
            print(f"  No QCD data for {year}, skipping.")
            continue

        # ── Load filter centres ──────────────────────────────────────────────
        filter_eta, filter_phi = load_filter_centers(cfg["filter_txt"])
        print(f"  Filter: {len(filter_eta)} centres loaded from {cfg['filter_txt']}")

        # ── Efficiency rows (accumulated for the CSV) ────────────────────────
        eff_rows = []

        if cfg["hem"]:
            # ── 2018: split data into preHEM and postHEM periods ─────────────
            print("\n  Splitting data by HEM run threshold ...")
            data_pre, data_hem = read_all_data(
                cfg["base_path"], cfg["data_datasets"], hem_split=True
            )

            # -- Plot preHEM --
            if qcd is not None and data_pre is not None:
                print("\n  Building maps for 2018 preHEM ...")
                h_mc_pre   = build_norm_map(qcd)
                h_data_pre = build_norm_map(data_pre)
                plot_filtered_map(
                    h_mc_pre, filter_eta, filter_phi,
                    period_label="2018 preHEM",
                    lumi=cfg["lumi"],
                    out_stem=os.path.join(out_dir, "etaphi_filter_2018_preHEM_mc"),
                    is_data=False,
                    mark_hem=False,
                )
                plot_filtered_map(
                    h_data_pre, filter_eta, filter_phi,
                    period_label="2018 preHEM",
                    lumi=cfg["lumi_pre"],
                    out_stem=os.path.join(out_dir, "etaphi_filter_2018_preHEM_data"),
                    is_data=True,
                    mark_hem=False,
                )
            else:
                print("  Skipping preHEM map (missing QCD or data).")

            # -- Plot postHEM --
            if qcd is not None and data_hem is not None:
                print("\n  Building maps for 2018 postHEM ...")
                h_mc_hem   = build_norm_map(qcd)
                h_data_hem = build_norm_map(data_hem)
                plot_filtered_map(
                    h_mc_hem, filter_eta, filter_phi,
                    period_label="2018 postHEM",
                    lumi=cfg["lumi"],
                    out_stem=os.path.join(out_dir, "etaphi_filter_2018_postHEM_mc"),
                    is_data=False,
                    mark_hem=True,
                )
                plot_filtered_map(
                    h_data_hem, filter_eta, filter_phi,
                    period_label="2018 postHEM",
                    lumi=cfg["lumi_hem"],
                    out_stem=os.path.join(out_dir, "etaphi_filter_2018_postHEM_data"),
                    is_data=True,
                    mark_hem=True,
                )
            else:
                print("  Skipping postHEM map (missing QCD or data).")

            # ── Efficiencies: QCD on full 2018 (combined filter) ─────────────
            print("\n  Computing 2018 QCD efficiency ...")
            pre, post, eff = compute_efficiency(qcd, filter_eta, filter_phi)
            eff_rows.append({"sample": "QCD",
                             "pre_yield": pre, "post_yield": post, "efficiency": eff})
            print(f"    QCD  pre={pre:.3e}  post={post:.3e}  eff={eff:.4f}")

        else:
            # ── 2017: single period ───────────────────────────────────────────
            print("\n  Reading data for 2017 ...")
            data = read_all_data(
                cfg["base_path"], cfg["data_datasets"], hem_split=False
            )

            if qcd is not None and data is not None:
                print("\n  Building maps for 2017 ...")
                h_mc   = build_norm_map(qcd)
                h_data = build_norm_map(data)
                plot_filtered_map(
                    h_mc, filter_eta, filter_phi,
                    period_label="2017",
                    lumi=cfg["lumi"],
                    out_stem=os.path.join(out_dir, "etaphi_filter_2017_mc"),
                    is_data=False,
                    mark_hem=False,
                )
                plot_filtered_map(
                    h_data, filter_eta, filter_phi,
                    period_label="2017",
                    lumi=cfg["lumi"],
                    out_stem=os.path.join(out_dir, "etaphi_filter_2017_data"),
                    is_data=True,
                    mark_hem=False,
                )
            else:
                print("  Skipping 2017 map (missing QCD or data).")

            # ── Efficiency: QCD ───────────────────────────────────────────────
            print("\n  Computing 2017 QCD efficiency ...")
            pre, post, eff = compute_efficiency(qcd, filter_eta, filter_phi)
            eff_rows.append({"sample": "QCD",
                             "pre_yield": pre, "post_yield": post, "efficiency": eff})
            print(f"    QCD  pre={pre:.3e}  post={post:.3e}  eff={eff:.4f}")

        # ── Signal efficiencies ──────────────────────────────────────────────
        print(f"\n  Computing signal efficiencies for {year} ...")
        for sig_label, (sig_folder, _) in SIG_DICT.items():
            sig_path = f"{cfg['sig_base']}/{sig_folder}"
            sig_data = read_mc_dataset(sig_label, sig_path, cfg["lumi"],
                                       show_progress=False)
            if sig_data is None:
                print(f"    {sig_label}: no data, skipping.")
                continue

            pre, post, eff = compute_efficiency(sig_data, filter_eta, filter_phi)
            eff_rows.append({"sample": sig_label,
                             "pre_yield": pre, "post_yield": post, "efficiency": eff})
            print(f"    {sig_label:<12}  pre={pre:.3e}  post={post:.3e}  eff={eff:.4f}")

        # ── Save efficiency CSV ──────────────────────────────────────────────
        if eff_rows:
            csv_path = os.path.join(out_dir, f"efficiencies_{year}.csv")
            with open(csv_path, "w") as fh:
                fh.write("sample,pre_yield,post_yield,efficiency\n")
                for row in eff_rows:
                    fh.write(
                        f"{row['sample']},{row['pre_yield']:.6e},"
                        f"{row['post_yield']:.6e},{row['efficiency']:.6f}\n"
                    )
            print(f"\n  Efficiencies saved to {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
