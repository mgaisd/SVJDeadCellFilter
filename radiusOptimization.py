#!/usr/bin/env python3
"""
radiusOptimization.py

Fix the dead-cell phi-spike filter threshold at n_sigma=5 and scan the veto
radius (the multiplier applied to the bin-size² = 0.0288) from 0.05 to 1.2.

For each radius multiplier:
  1. Build the QCD eta-phi spike map (same normalization as DeadCellStudy.py).
  2. Mark bins > mean + N_SIGMA * std as "bad" (constant threshold).
  3. Veto events whose subleading jet falls within sqrt(mult * RAD_SQUARED) in ΔR
     of any bad bin centre.
  4. Compute weighted yields for QCD and each signal hypothesis.
  5. Evaluate FOM = sqrt(2 * ((S+B)*log(1+S/B) - S)) and plot vs radius multiplier.

Years: 2017 (full) and 2018 pre-HEM only.
Outputs: PDF + PNG plots in OUTPUT_DIR.
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)
plt.rcParams.update({
    "font.size":        14,
    "axes.labelsize":   16,
    "axes.titlesize":   16,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  11,
})

# ---------------------------------------------------------------------------
# Binning – must match DeadCellStudy.py
# ---------------------------------------------------------------------------
ETA_BINS = np.linspace(-2.4,   2.4,  50)
PHI_BINS = np.linspace(-np.pi, np.pi, 50)

# ---------------------------------------------------------------------------
# Filter configuration
# ---------------------------------------------------------------------------
N_SIGMA     = 5.0           # fixed threshold for bad-bin identification
# ΔR² of the bin diagonal = deta² + dphi²  (computed from actual bin edges)
_deta = (ETA_BINS[1] - ETA_BINS[0])
_dphi = (PHI_BINS[1] - PHI_BINS[0])
#RAD_SQUARED = _deta**2 + _dphi**2   # ≈ 0.02604 (one bin's ΔR²)
RAD_SQUARED = 0.0288 

# Radius multipliers to scan: veto_radius² = mult * RAD_SQUARED
RADIUS_VALS = np.round(np.arange(0.05, 1.25, 0.05), 3).tolist()  # 0.05 … 1.20

# ---------------------------------------------------------------------------
# Remote file access
# ---------------------------------------------------------------------------
XROOTD_SERVER = "cmsdcache-kit-disk.gridka.de:1094"
FILE_WORKERS  = 8

JET_ETA_BRANCH = "Jet_eta"
JET_PHI_BRANCH = "Jet_phi"

# ---------------------------------------------------------------------------
# Per-year configuration
# ---------------------------------------------------------------------------
YEARS = {
    "2017": {
        "label":         "2017",
        "lumi":          41.53,      # fb^-1
        "qcd_base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_JEC/2017/s_channel_scouting_pre_selection/nominal",
        "sig_base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims/2017/s_channel_scouting_pre_selection",
    },
    "2018": {
        "label":         "2018",
        "lumi":          59.74,      # fb^-1
        "qcd_base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_JEC/2018/s_channel_scouting_pre_selection/nominal",
        "sig_base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims/2018/s_channel_scouting_pre_selection",
    },
}

QCD_DATASETS = [
    "QCD_HT300to500",
    "QCD_HT500to700",
    "QCD_HT700to1000",
    "QCD_HT1000to1500",
    "QCD_HT1500to2000",
    "QCD_HT2000toInf",
]

# Signal samples: { label: [dataset_folder_name, hex_color] }
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

OUTPUT_DIR = "plots_radiusOpt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------

def get_file_list(dataset_path: str) -> list:
    """List ROOT files in *dataset_path* via xrdfs."""
    lfn_prefix = f"root://{XROOTD_SERVER}/"
    remote_path = dataset_path.replace(lfn_prefix, "")
    cmd = ["xrdfs", XROOTD_SERVER, "ls", remote_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return [
            f"{lfn_prefix}{line.strip()}"
            for line in result.stdout.splitlines()
            if line.strip().endswith(".root")
        ]
    except Exception as exc:
        print(f"  Warning: could not list {dataset_path}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Single-file reader
# ---------------------------------------------------------------------------

def _read_single_file(file_path: str, lumi: float) -> dict | None:
    """
    Return a dict with:
      eta_sub  – subleading-jet eta per event (1-D numpy array)
      phi_sub  – subleading-jet phi per event (1-D numpy array)
      weights  – per-event weight = xsec * lumi * 1e3 * prefire / N_initial

    Only events with ≥2 jets are kept.
    Returns None on any failure.
    """
    name = Path(file_path).name
    try:
        with uproot.open(file_path) as f:
            try:
                n_initial = f["CutFlow"]["Initial"].array(library="np")[0]
                xsec      = f["Metadata"]["GenCrossSection"].array(library="np")[0]
            except Exception as exc:
                print(f"    Warning: normalisation info missing in {name}: {exc}")
                return None

            try:
                arrs = f["Events"].arrays(
                    [JET_ETA_BRANCH, JET_PHI_BRANCH, "prefire"],
                    library="ak",
                )
            except Exception as exc:
                print(f"    Warning: Events tree unreadable in {name}: {exc}")
                return None

        eta     = arrs[JET_ETA_BRANCH]
        phi     = arrs[JET_PHI_BRANCH]
        prefire = arrs["prefire"]

        # --- require subleading jet ---
        has_sublead = ak.num(eta) >= 2
        eta     = eta[has_sublead]
        phi     = phi[has_sublead]
        prefire = prefire[has_sublead]

        eta_sub = ak.to_numpy(eta[:, 1])
        phi_sub = ak.to_numpy(phi[:, 1])
        # partial weight: xsec * lumi * 1e3 * prefire  (divide by n_initial_total later)
        weights = xsec * lumi * 1e3 * ak.to_numpy(prefire)

        return {"eta_sub": eta_sub, "phi_sub": phi_sub, "weights": weights, "n_initial": n_initial}

    except Exception as exc:
        print(f"    Error opening {name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Dataset readers
# ---------------------------------------------------------------------------

def read_dataset(label: str, dataset_path: str, lumi: float) -> dict | None:
    """Read all ROOT files for one dataset and concatenate into a single dict."""
    print(f"  Reading {label} ...")
    files = get_file_list(dataset_path)
    if not files:
        print(f"    No files found in {dataset_path}")
        return None
    print(f"    Found {len(files)} file(s)")

    results = [None] * len(files)
    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
        future_map = {
            pool.submit(_read_single_file, fp, lumi): i
            for i, fp in enumerate(files)
        }
        for fut in as_completed(future_map):
            results[future_map[fut]] = fut.result()

    good = [r for r in results if r is not None]
    if not good:
        print(f"    No data read.")
        return None

    print(f"    {len(good)}/{len(files)} files ok")
    d = {
        "eta_sub": np.concatenate([r["eta_sub"] for r in good]),
        "phi_sub": np.concatenate([r["phi_sub"] for r in good]),
        "weights": np.concatenate([r["weights"]  for r in good]),
    }
    n_initial_total = sum(r["n_initial"] for r in good)
    d["weights"] /= n_initial_total
    return d


def read_qcd_all(qcd_base_path: str, lumi: float) -> dict | None:
    """Read and merge all QCD HT-binned datasets."""
    all_datasets = []
    for dname in QCD_DATASETS:
        d = read_dataset(dname, f"{qcd_base_path}/{dname}", lumi)
        if d is not None:
            all_datasets.append(d)
    if not all_datasets:
        return None
    return {
        "eta_sub": np.concatenate([d["eta_sub"] for d in all_datasets]),
        "phi_sub": np.concatenate([d["phi_sub"] for d in all_datasets]),
        "weights": np.concatenate([d["weights"]  for d in all_datasets]),
    }


# ---------------------------------------------------------------------------
# Dead-cell spike map  (identical to DeadCellStudy.py::normalize_by_eta_bin)
# ---------------------------------------------------------------------------

def build_spike_map(data: dict) -> np.ndarray:
    """Build normalized 2-D enhancement map from *data*."""
    h2d, _, _ = np.histogram2d(
        data["eta_sub"],
        data["phi_sub"],
        bins=[ETA_BINS, PHI_BINS],
        weights=data["weights"],
    )
    h_norm = np.zeros_like(h2d)
    for i in range(h2d.shape[0]):
        row = h2d[i, :].copy()
        max_val = np.max(row)
        if max_val > 0:
            row /= max_val
        row -= np.mean(row)
        h_norm[i, :] = np.maximum(0.0, row)
    return h_norm


def get_bad_bin_mask(spike_map: np.ndarray, n_sigma: float) -> np.ndarray:
    """Boolean mask: True = bad bin (spike_map > mean + n_sigma * std)."""
    mean = np.mean(spike_map)
    std  = np.std(spike_map)
    return spike_map > mean + n_sigma * std


# ---------------------------------------------------------------------------
# Yield computation
# ---------------------------------------------------------------------------

def compute_yield(data: dict, bad_mask: np.ndarray, veto_radius: float) -> float:
    """
    Sum event weights for events whose subleading jet is NOT within
    sqrt(veto_radius) in ΔR of any bad bin centre.

    *veto_radius* is compared directly against deta**2 + dphi**2 (i.e. it
    is already ΔR²).
    """
    eta_centers = 0.5 * (ETA_BINS[:-1] + ETA_BINS[1:])
    phi_centers = 0.5 * (PHI_BINS[:-1] + PHI_BINS[1:])

    bad_i, bad_j = np.where(bad_mask)
    if len(bad_i) == 0:
        return float(np.sum(data["weights"]))

    bad_eta = eta_centers[bad_i]        # (N_bad,)
    bad_phi = phi_centers[bad_j]        # (N_bad,)

    eta_jet = data["eta_sub"][:, None]  # (N_events, 1)
    phi_jet = data["phi_sub"][:, None]  # (N_events, 1)

    deta = eta_jet - bad_eta
    dphi = phi_jet - bad_phi
    dphi = np.where(dphi >  np.pi, dphi - 2.0 * np.pi, dphi)  # wrap to (-π, π]
    dphi = np.where(dphi < -np.pi, dphi + 2.0 * np.pi, dphi)

    passes = ~np.any(deta**2 + dphi**2 < veto_radius, axis=1)
    return float(np.sum(data["weights"][passes]))


# ---------------------------------------------------------------------------
# FOM
# ---------------------------------------------------------------------------

def fom(B: float | np.ndarray, S: float | np.ndarray) -> float | np.ndarray:
    """Asymptotic profile-likelihood FOM: sqrt(2*((S+B)*ln(1+S/B) - S))."""
    return np.sqrt(2.0 * ((S + B) * np.log(1.0 + S / B) - S))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fom_by_sig_para(
    radius_vals: list,
    metrics: dict,
    sig_para_name: str,
    year_label: str,
    year_key: str,
    output_dir: str,
):
    """
    Plot normalized FOM vs veto-radius multiplier for all signal hypotheses
    whose label contains *sig_para_name* (e.g. "mMed", "rinv").

    Each signal hypothesis is one step-curve.  A dashed horizontal line shows
    the FOM without any phi-spike filter (pre-filter baseline).

    The x-axis is the multiplier m, where the actual veto is:
        ΔR² < m × RAD_SQUARED  (= m × 0.0288)
    """
    fig, ax = plt.subplots()

    pre_bkg  = metrics["pre_bkg_yield"]
    bkg_vals = np.array([metrics["bkg_yields"][r] for r in radius_vals])

    any_plotted = False
    for sig_lab, sig_data in metrics["sig_data"].items():
        if sig_para_name not in sig_lab:
            continue

        color    = sig_data["color"]
        pre_sig  = sig_data["pre_yield"]
        sig_vals = np.array([sig_data["yields"][r] for r in radius_vals])

        if pre_bkg <= 0 or pre_sig <= 0:
            continue
        valid = (bkg_vals > 0) & (sig_vals > 0)
        if not np.any(valid):
            continue

        fom_curve = np.where(valid, fom(bkg_vals, sig_vals), np.nan)
        norm      = np.nanmax(fom_curve)
        if norm == 0 or np.isnan(norm):
            continue

        ax.step(
            radius_vals, fom_curve / norm,
            where="mid", label=sig_lab, color=color,
        )
        any_plotted = True

    if not any_plotted:
        plt.close()
        return

    ax.set_xlabel(r"Filter Radius Multiplier $m$  $(r^2 = m \times 0.0288)$")
    ax.set_ylabel("Normalized FOM")
    ax.legend(loc=(1.04, 0))
    hep.cms.label("Private Work", data=False, year=year_label, ax=ax)

    stem = os.path.join(output_dir, f"FOM_{year_key}_{sig_para_name}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=150)
        print(f"  Saved {stem}.{ext}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for year_key, cfg in YEARS.items():
        lumi          = cfg["lumi"]
        year_label    = cfg["label"]
        qcd_base_path = cfg["qcd_base_path"]
        sig_base_path = cfg["sig_base_path"]

        year_outdir = os.path.join(OUTPUT_DIR, year_key)
        os.makedirs(year_outdir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {year_label}  --  L = {lumi} fb^-1")
        print(f"{'='*60}")

        # --- Read QCD ---
        print("\nReading QCD datasets...")
        qcd_all = read_qcd_all(qcd_base_path, lumi)
        if qcd_all is None:
            print(f"  No QCD data for {year_label}, skipping.")
            continue

        # --- Build spike map and bad-bin mask (fixed N_SIGMA) ---
        print(f"\nBuilding dead-cell spike map (N_SIGMA = {N_SIGMA}) ...")
        spike_map = build_spike_map(qcd_all)
        bad_mask  = get_bad_bin_mask(spike_map, N_SIGMA)
        n_bad = int(np.sum(bad_mask))
        print(f"  {n_bad} bad bins / {spike_map.size} total at sigma={N_SIGMA}")

        # --- Pre-compute veto radii ---
        veto_radii = {r: r * RAD_SQUARED for r in RADIUS_VALS}

        # --- Compute QCD yields for all radius multipliers ---
        pre_bkg_yield = float(np.sum(qcd_all["weights"]))
        bkg_yields    = {r: compute_yield(qcd_all, bad_mask, veto_radii[r])
                         for r in RADIUS_VALS}

        print(f"\n  QCD pre-filter yield : {pre_bkg_yield:.3e}")
        for r in RADIUS_VALS:
            y = bkg_yields[r]
            print(f"  QCD radius={r:.2f}  yield: {y:.3e}  (eff={y/pre_bkg_yield:.3f})")

        # --- Read signal samples ---
        sig_data = {}
        for sig_lab, (sig_dname, color) in SIG_DICT.items():
            d = read_dataset(sig_lab, f"{sig_base_path}/{sig_dname}", lumi)
            if d is None:
                continue
            pre_sig_yield = float(np.sum(d["weights"]))
            sig_yields    = {r: compute_yield(d, bad_mask, veto_radii[r])
                             for r in RADIUS_VALS}
            sig_data[sig_lab] = {
                "color":     color,
                "pre_yield": pre_sig_yield,
                "yields":    sig_yields,
            }

        metrics = {
            "pre_bkg_yield": pre_bkg_yield,
            "bkg_yields":    bkg_yields,
            "sig_data":      sig_data,
        }

        # --- Save efficiencies CSV ---
        csv_path = os.path.join(year_outdir, f"efficiencies_{year_key}.csv")
        with open(csv_path, "w") as f:
            rad_header = ",".join(f"eff_r{r:.2f}" for r in RADIUS_VALS)
            f.write(f"sample,pre_yield,{rad_header}\n")

            bkg_effs = ",".join(
                f"{bkg_yields[r]/pre_bkg_yield:.6f}" for r in RADIUS_VALS
            )
            f.write(f"QCD,{pre_bkg_yield:.6e},{bkg_effs}\n")

            for sig_lab, sdata in sig_data.items():
                pre = sdata["pre_yield"]
                sig_effs = ",".join(
                    f"{sdata['yields'][r]/pre:.6f}" if pre > 0 else "nan"
                    for r in RADIUS_VALS
                )
                f.write(f"{sig_lab},{pre:.6e},{sig_effs}\n")

        print(f"\n  Saved efficiencies to {csv_path}")

        # --- Per-year FOM-vs-radius plots (one plot per sig_para_name group) ---
        print("\nGenerating per-year FOM plots...")
        for para in ("mMed", "rinv"):
            plot_fom_by_sig_para(RADIUS_VALS, metrics, para,
                                 year_label, year_key, year_outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
