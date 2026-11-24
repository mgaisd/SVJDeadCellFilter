#!/usr/bin/env python3
"""
CompareDeadCellFilters.py - compare two dead-cell outlier-centre lists

Compares the filter derived from skims (plots_deadcell/{year}) with the one
derived from ntuples with the on-the-fly pre-selection
(plots_deadcell_ntuples/{year}), both from DeadCellStudy.py:
  - overlap of the outlier bins (common / reference-only / new-only),
    split by the source tag of the reference list (data, mc, data&mc)
  - overlay of both lists in the eta-phi plane
  - fraction of events vetoed by either filter, evaluated on the same
    selected events (subleading good AK4 jet from cache_ntuples/{year})

Usage:
    python CompareDeadCellFilters.py --year 2017
"""

import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)

ETA_BINS = np.linspace(-2.4,  2.4, 50)   # must match DeadCellStudy.py
PHI_BINS = np.linspace(-np.pi, np.pi, 50)
DEAD_CELL_RAD = 0.028816 * 0.35           # skimmer convention: dist^2 < rad


def load_centers(path: str):
    """Return dict {(eta, phi): source} rounded to the 6 digits of the file."""
    out = {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            out[(round(float(parts[0]), 5), round(float(parts[1]), 5))] = (
                parts[2] if len(parts) > 2 else "?")
    return out


def veto_mask(eta, phi, centers, chunk=200_000):
    """Chunked over events to keep memory at ~100 MB (an unchunked version
    with ~4M events x ~50 centres needs several GB)."""
    out = np.zeros(len(eta), dtype=bool)
    if not centers:
        return out
    c = np.array(list(centers))
    for i in range(0, len(eta), chunk):
        deta = eta[i:i + chunk, None] - c[None, :, 0]
        dphi = (phi[i:i + chunk, None] - c[None, :, 1] + np.pi) % (2 * np.pi) - np.pi
        out[i:i + chunk] = np.any(deta ** 2 + dphi ** 2 < DEAD_CELL_RAD, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", default="2017")
    ap.add_argument("--ref", default=None, help="reference list (default: skim-based)")
    ap.add_argument("--new", default=None, help="new list (default: ntuple-based)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <dir of new list>/comparison)")
    ap.add_argument("--labels", nargs=2, default=["skim-based", "ntuple-based"],
                    metavar=("REF", "NEW"), help="legend labels for the two lists")
    args = ap.parse_args()
    y = args.year
    ref_path = args.ref or f"plots_deadcell/{y}/outlier_centers_{y}.txt"
    new_path = args.new or f"plots_deadcell_ntuples/{y}/outlier_centers_{y}.txt"
    out_dir = args.out or os.path.join(os.path.dirname(new_path), "comparison")
    lab_ref, lab_new = args.labels
    os.makedirs(out_dir, exist_ok=True)

    ref, new = load_centers(ref_path), load_centers(new_path)
    common = set(ref) & set(new)
    lines = [f"Reference: {ref_path}  ({len(ref)} bins)",
             f"New:       {new_path}  ({len(new)} bins)",
             f"Common: {len(common)}   reference-only: {len(set(ref) - set(new))}"
             f"   new-only: {len(set(new) - set(ref))}", "",
             "Reference bins by source tag -> found in new list:"]
    for tag in sorted(set(ref.values())):
        keys = [k for k, v in ref.items() if v == tag]
        n_found = sum(k in new for k in keys)
        lines.append(f"  {tag:<10} {n_found:>4} / {len(keys):<4}")
    ref_data = {k for k, v in ref.items() if "data" in v}
    lines.append(f"  any data   {len(ref_data & set(new)):>4} / {len(ref_data):<4}"
                 "   (reference bins flagged in data)")

    lines += ["", "Reference-only bins (eta, phi, source):"]
    lines += [f"  {e:+.4f} {p:+.4f}  {ref[(e, p)]}" for e, p in sorted(set(ref) - set(new))]
    lines += ["", "New-only bins (eta, phi):"]
    lines += [f"  {e:+.4f} {p:+.4f}" for e, p in sorted(set(new) - set(ref))]

    # --- veto fractions on the same selected events ---
    files = glob.glob(f"cache_ntuples/{y}/*/*.npz")
    if files:
        eta, phi = [], []
        for f in files:
            with np.load(f) as d:
                m = d["n_good"] >= 2
                eta.append(d["eta1"][m]); phi.append(d["phi1"][m])
        eta = np.concatenate(eta).astype(np.float64)
        phi = np.concatenate(phi).astype(np.float64)
        v_ref, v_new = veto_mask(eta, phi, ref), veto_mask(eta, phi, new)
        n = len(eta)
        lines += ["", f"Veto fractions on {n} selected ntuple events "
                      f"({len(files)} files, subleading good AK4 jet):",
                  f"  reference filter : {v_ref.mean():.4%}",
                  f"  new filter       : {v_new.mean():.4%}",
                  f"  both             : {(v_ref & v_new).mean():.4%}",
                  f"  reference only   : {(v_ref & ~v_new).mean():.4%}",
                  f"  new only         : {(~v_ref & v_new).mean():.4%}"]

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, f"comparison_{y}.txt"), "w") as fh:
        fh.write(report + "\n")

    # --- overlay plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    styles = {"common": dict(marker="o", color="tab:green", label=f"common ({len(common)})"),
              "ref":    dict(marker="s", color="tab:red",
                             label=f"{lab_ref} only ({len(set(ref) - set(new))})"),
              "new":    dict(marker="^", color="tab:blue",
                             label=f"{lab_new} only ({len(set(new) - set(ref))})")}
    for key, pts in [("common", common), ("ref", set(ref) - set(new)),
                     ("new", set(new) - set(ref))]:
        if pts:
            e, p = zip(*pts)
            ax.scatter(e, p, s=40, facecolors="none", linewidths=1.8,
                       edgecolors=styles[key]["color"], marker=styles[key]["marker"],
                       label=styles[key]["label"])
    ax.set_xlim(ETA_BINS[0], ETA_BINS[-1]); ax.set_ylim(PHI_BINS[0], PHI_BINS[-1])
    ax.set_xticks(np.arange(-2.4, 2.41, 0.8))
    ax.grid(alpha=0.3)
    ax.set_xlabel(r"AK4 Jet 2 $\eta$", ha="right", x=1.0)
    ax.set_ylabel(r"AK4 Jet 2 $\phi$", ha="right", y=1.0)
    ax.set_title(r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Data)}$", loc="left", pad=6)
    ax.text(1, 1, f"{y}", transform=ax.transAxes, ha="right", va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=13)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"filter_overlay_{y}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
