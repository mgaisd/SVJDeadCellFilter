#!/usr/bin/env python3
"""
DeadCellStudy.py - Jet eta-phi distribution analyzer for CMS data

Reads MC (QCD HT-binned) and Data for the requested year(s), then for each
source and each jet (leading / subleading) produces:
  - raw eta-phi map (log-z)
  - normalised eta-phi map (per-eta-bin normalisation)
  - normalised map with 5-sigma outlier bins marked

Outlier bin centres (MC, Data, combined) for the subleading jet are written
to a text file per year/period for use as dead-cell filter inputs.

For 2018 the HEM issue is handled automatically: pre-HEM and HEM periods
are read and plotted separately.

2016 (data only) is read from the unskimmed ntuples with the s-channel
scouting pre-selection applied on the fly (no JEC, no phi-spike filter), so
the filter can be derived before the skims are made.  Per-file results are
cached in cache_ntuples/ so interrupted runs resume where they stopped.

Usage:
------
    python DeadCellStudy.py                      # 2017 (default)
    python DeadCellStudy.py --years 2017 2018    # both years
    python DeadCellStudy.py --years 2016 [--vfp-split]   # 2016 from ntuples
    python DeadCellStudy.py --years 2017 --from-ntuples  # validate ntuple path

Output  (plots_deadcell/{year}/):
    etaphi_ak4_{lead|sublead}_{year}_{mc|data}[_period].{png,pdf}
    etaphi_ak4_{lead|sublead}_{year}_{mc|data}[_period]_normalized.{png,pdf}
    etaphi_ak4_{lead|sublead}_{year}_{mc|data}[_period]_outliers.{png,pdf}
    outlier_centers_{year}[_period].txt

Requirements:
    uproot, numpy, matplotlib, mplhep, awkward, tqdm
    XRootD client tools (xrdfs) for remote file access
"""

import os
import time
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import awkward as ak
from tqdm import tqdm

# Set CMS style
plt.style.use(hep.style.CMS)

# Global font size configuration
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 16

# ---------------------------------------------------------------------------
# Per-year configuration
# ---------------------------------------------------------------------------
# Configuration for Monte Carlo samples
MC_YEARS = {
    # "2016": {
    #     "lumi":      16.8,   # fb^-1  (2016 legacy)
    #     "base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_JEC/2016/s_channel_scouting_pre_selection/nominal",
    # },
    "2017": {
        "lumi":      41.53,  # fb^-1
        "base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_JEC/2017/s_channel_scouting_pre_selection/nominal",
    },
    "2018": {
        "lumi":      59.74,  # fb^-1
        "base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_JEC/2018/s_channel_scouting_pre_selection/nominal",
        # HEM issue configuration - split MC into two periods matching data
        "hem_issue": True,
        "lumi_pre_hem": 21.1,   # fb^-1 (pre-HEM period, no HEM veto)
        "lumi_hem": 38.64,      # fb^-1 (HEM-affected period, apply HEM veto)
    },
}

# MC Datasets to process (same for all years)
MC_DATASET_NAMES = [
    "QCD_HT300to500",
    "QCD_HT500to700",
    "QCD_HT700to1000",
    "QCD_HT1000to1500",
    "QCD_HT1500to2000",
    "QCD_HT2000toInf",
]

# Configuration for data samples
DATA_YEARS = {
    "2017": {
        "lumi":      41.53,  # fb^-1
        "base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_JEC/2017/s_channel_scouting_pre_selection/nominal",
        "datasets": [
            "Run2017C",
            "Run2017D",
            "Run2017E",
            "Run2017F",
        ],
    },
    "2018": {
        "lumi":      59.74,  # fb^-1
        "base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_JEC/2018/s_channel_scouting_pre_selection/nominal",
        "datasets": [
            "Run2018A",
            "Run2018B",
            "Run2018C",
            "Run2018D",
        ],
        # HEM issue configuration (runs >= 319077 are affected)
        "hem_issue": True,
        "hem_run_threshold": 319077,
        "lumi_pre_hem": 21.1,   # fb^-1 (pre-HEM period)
        "lumi_hem": 38.64,      # fb^-1 (HEM-affected period)
    },
}

# Configuration for reading data directly from the (unskimmed) ntuples.
# The s-channel scouting pre-selection is applied on the fly (without JEC and
# without the phi-spike filter itself), so the filter can be derived before
# the skims are produced.  Years only present here (2016) are always read
# from ntuples; for other years use --from-ntuples (validation).
NTUPLE_TREE = "mmtree/Events"
GOLDEN_JSON_DIR = "/work/mgais/SVJProcessing/data/golden_json"
NTUPLE_DATA_YEARS = {
    "2016": {
        "lumi":      36.3,   # fb^-1  golden JSON 2016 -- TODO verify scouting lumi with brilcalc
        "base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_ntuples/Data/2016_v3",
        "datasets": [
            "Run2016B-v1",
            "Run2016B-v2",
            "Run2016C-v2",
            "Run2016D-v2",
            "Run2016E-v2",
            "Run2016F-v1",
            "Run2016G-v1",
            "Run2016H-v1",
        ],
        "golden_json": f"{GOLDEN_JSON_DIR}/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
        # PV_isValidVtx (and PV_ndof) are always 0 in the 2016 scouting vertex
        # collection -> only apply the |z| and rho cuts of the good-PV filter
        "pv_require_valid": False,
        # optional preVFP/postVFP split (--vfp-split): runs < threshold are preVFP (HIPM)
        "vfp_run_threshold": 278769,
        "lumi_pre_vfp":  19.5,  # fb^-1
        "lumi_post_vfp": 16.8,  # fb^-1
    },
    "2017": {
        "lumi":      41.53,  # fb^-1
        "base_path": "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_ntuples/Data/2017_v2",
        "datasets": [
            "Run2017C",
            "Run2017D",
            "Run2017E",
            "Run2017F",
        ],
        "golden_json": f"{GOLDEN_JSON_DIR}/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
    },
}
NTUPLE_CACHE_DIR = "cache_ntuples"
NTUPLE_STEP_SIZE = 100_000  # events per chunk (keeps memory per worker bounded)
NTUPLE_RETRIES = 4          # attempts per file / directory listing
NTUPLE_RETRY_WAIT = 60      # s, multiplied by the attempt number

# AK4 jet branch prefix — adjust if the ntuples use a different naming convention
JET_ETA_BRANCH = "Jet_eta"
JET_PHI_BRANCH = "Jet_phi"

# Histogram binning for eta-phi maps
ETA_BINS = np.linspace(-2.4,  2.4, 50)
PHI_BINS = np.linspace(-np.pi, np.pi, 50)

# Parallel workers
FILE_WORKERS = 8

# Sigma threshold for marking outlier bins in the normalized eta-phi maps
N_SIGMA = 4.0

# Dead-cell veto radius in eta-phi space  (half-diagonal of cell × optimised factor)
DEAD_CELL_RAD = 0.028816 * 0.35

OUTPUT_DIR = "plots_deadcell"
os.makedirs(OUTPUT_DIR, exist_ok=True)
XROOTD_SERVER = "cmsdcache-kit-disk.gridka.de:1094"


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------

def get_file_list(dataset_path: str) -> list:
    """List ROOT files in *dataset_path* via xrdfs."""
    lfn_prefix = f"root://{XROOTD_SERVER}/"
    remote_path = dataset_path.replace(lfn_prefix, "")
    try:
        from XRootD import client
        status, listing = client.FileSystem(f"root://{XROOTD_SERVER}").dirlist(remote_path)
        if listing is not None:
            return sorted(f"{lfn_prefix}{remote_path.rstrip('/')}/{e.name}"
                          for e in listing if e.name.endswith(".root"))
        print(f"  Warning: could not list {dataset_path}: {status.message}")
        return []
    except ImportError:
        pass
    cmd = ["xrdfs", XROOTD_SERVER, "ls", remote_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        files = [
            f"{lfn_prefix}{line.strip()}"
            for line in result.stdout.splitlines()
            if line.strip().endswith(".root")
        ]
        return files
    except Exception as exc:
        print(f"  Warning: could not list {dataset_path}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Single-file reader
# ---------------------------------------------------------------------------

def _process_jets(eta, phi, prefiring_wgt, weight_per_event, is_data=False):
    """Extract leading/subleading jet arrays and compute per-event weights.

    In addition to the usual lead/sub arrays, also stores:
      eta_lead_paired / phi_lead_paired / w_lead_paired
          Jet-1 position for events that *also* have a jet-2 (index-aligned
          with eta_sub so a per-event sublead veto can be applied later).
      eta_lead_1only / phi_lead_1only / w_lead_1only
          Jet-1 position for events with *exactly* 1 jet (these always pass
          any sublead-based veto).
    """
    has_lead    = ak.num(eta) >= 1
    has_sublead = ak.num(eta) >= 2
    has_1only   = ak.num(eta) == 1

    eta_lead = ak.to_numpy(eta[has_lead, 0])
    phi_lead = ak.to_numpy(phi[has_lead, 0])
    w_lead = (np.ones(len(eta_lead)) if is_data
              else weight_per_event * ak.to_numpy(prefiring_wgt[has_lead]))

    eta_sub = ak.to_numpy(eta[has_sublead, 1])
    phi_sub = ak.to_numpy(phi[has_sublead, 1])
    w_sub = (np.ones(len(eta_sub)) if is_data
             else weight_per_event * ak.to_numpy(prefiring_wgt[has_sublead]))

    # Jet-1 aligned with jet-2 (same events, same order as eta_sub)
    eta_lead_paired = ak.to_numpy(eta[has_sublead, 0])
    phi_lead_paired = ak.to_numpy(phi[has_sublead, 0])
    w_lead_paired = (np.ones(len(eta_lead_paired)) if is_data
                     else weight_per_event * ak.to_numpy(prefiring_wgt[has_sublead]))

    # Jet-1 for events with exactly 1 jet (no sublead to veto on)
    eta_lead_1only = ak.to_numpy(eta[has_1only, 0])
    phi_lead_1only = ak.to_numpy(phi[has_1only, 0])
    w_lead_1only = (np.ones(len(eta_lead_1only)) if is_data
                    else weight_per_event * ak.to_numpy(prefiring_wgt[has_1only]))

    return {
        "weight_per_event":  weight_per_event,
        "eta_lead":          eta_lead,          "phi_lead":          phi_lead,          "w_lead":          w_lead,
        "eta_sub":           eta_sub,            "phi_sub":           phi_sub,            "w_sub":           w_sub,
        "eta_lead_paired":   eta_lead_paired,    "phi_lead_paired":   phi_lead_paired,    "w_lead_paired":   w_lead_paired,
        "eta_lead_1only":    eta_lead_1only,     "phi_lead_1only":    phi_lead_1only,     "w_lead_1only":    w_lead_1only,
    }


def _read_single_file(file_path: str, lumi: float, is_data: bool = False,
                      split_hem: bool = False, lumi_pre_hem: float = None,
                      lumi_hem: float = None, hem_run_threshold: int = None):
    """Open one ROOT file and return jet arrays needed for eta-phi maps.

    Returns a dict, a (pre_hem_dict, hem_dict) tuple when split_hem=True, or None on failure.
    """
    name = Path(file_path).name
    try:
        with uproot.open(file_path) as f:
            # --- normalisation ---
            if is_data:
                weight_per_event = 1.0
                xsec = n_initial = None
            else:
                try:
                    n_initial = f["CutFlow"]["Initial"].array(library="np")[0]
                except Exception as exc:
                    tqdm.write(f"    Warning: CutFlow missing in {name}: {exc}")
                    return None
                try:
                    xsec = f["Metadata"]["GenCrossSection"].array(library="np")[0]
                except Exception as exc:
                    tqdm.write(f"    Warning: GenCrossSection missing in {name}: {exc}")
                    return None
                # partial weight: xsec * lumi * 1e3 (no /n_initial yet — divided by
                # n_initial_total across all files in read_dataset)
                weight_per_event = xsec * lumi * 1e3

            # --- read branches ---
            branch_list = [JET_ETA_BRANCH, JET_PHI_BRANCH]
            if is_data and split_hem:
                branch_list.append("run")
            if not is_data:
                branch_list.append("prefire")
                if split_hem:
                    branch_list.append("HEMVeto")
            try:
                arrs = f["Events"].arrays(branch_list, library="ak")
            except Exception as exc:
                tqdm.write(f"    Warning: Events unreadable in {name}: {exc}")
                return None

            # --- HEM splitting (returns a tuple) ---
            if split_hem:
                if is_data:
                    is_pre = arrs["run"] < hem_run_threshold
                    a_pre, a_hem = arrs[is_pre], arrs[~is_pre]
                    r_pre = _process_jets(a_pre[JET_ETA_BRANCH], a_pre[JET_PHI_BRANCH],
                                          None, 1.0, is_data=True)
                    r_hem = _process_jets(a_hem[JET_ETA_BRANCH], a_hem[JET_PHI_BRANCH],
                                          None, 1.0, is_data=True)
                else:
                    np.random.seed(hash(file_path) % (2**32))
                    frac_pre = lumi_pre_hem / (lumi_pre_hem + lumi_hem)
                    is_pre = ak.Array(np.random.random(len(arrs[JET_ETA_BRANCH])) < frac_pre)
                    prefire = arrs["prefire"]
                    r_pre = _process_jets(
                        arrs[JET_ETA_BRANCH][is_pre], arrs[JET_PHI_BRANCH][is_pre],
                        prefire[is_pre], xsec * lumi_pre_hem * 1e3)
                    eta_h = arrs[JET_ETA_BRANCH][~is_pre]
                    phi_h = arrs[JET_PHI_BRANCH][~is_pre]
                    pre_h = prefire[~is_pre]
                    try:
                        veto = arrs["HEMVeto"][~is_pre] > 0
                        eta_h, phi_h, pre_h = eta_h[veto], phi_h[veto], pre_h[veto]
                    except Exception as exc:
                        tqdm.write(f"    Warning: HEMVeto branch missing in {name}: {exc}")
                    r_hem = _process_jets(eta_h, phi_h, pre_h, xsec * lumi_hem * 1e3)
                    r_pre["n_initial"] = n_initial
                    r_hem["n_initial"] = n_initial
                return (r_pre, r_hem)

            # --- normal (non-split) read ---
            eta = arrs[JET_ETA_BRANCH]
            phi = arrs[JET_PHI_BRANCH]
            if not is_data:
                try:
                    prefiring_wgt = arrs["prefire"]
                except Exception as exc:
                    tqdm.write(f"    Warning: prefire branch missing in {name}, using weight=1: {exc}")
                    prefiring_wgt = ak.Array(np.ones(len(eta)))
            else:
                prefiring_wgt = None
            r = _process_jets(eta, phi, prefiring_wgt, weight_per_event, is_data)
            if not is_data:
                r["n_initial"] = n_initial
            return r
    except Exception as exc:
        tqdm.write(f"    Error opening {name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Dataset reader
# ---------------------------------------------------------------------------

def _build_dataset_dict(name: str, results: list) -> dict:
    """Concatenate per-file results into one dataset dict."""
    return {
        "name":            name,
        "n_ok":            len(results),
        "eta_lead":        np.concatenate([r["eta_lead"]        for r in results]),
        "phi_lead":        np.concatenate([r["phi_lead"]        for r in results]),
        "w_lead":          np.concatenate([r["w_lead"]          for r in results]),
        "eta_sub":         np.concatenate([r["eta_sub"]         for r in results]),
        "phi_sub":         np.concatenate([r["phi_sub"]         for r in results]),
        "w_sub":           np.concatenate([r["w_sub"]           for r in results]),
        "eta_lead_paired": np.concatenate([r["eta_lead_paired"] for r in results]),
        "phi_lead_paired": np.concatenate([r["phi_lead_paired"] for r in results]),
        "w_lead_paired":   np.concatenate([r["w_lead_paired"]   for r in results]),
        "eta_lead_1only":  np.concatenate([r["eta_lead_1only"]  for r in results]),
        "phi_lead_1only":  np.concatenate([r["phi_lead_1only"]  for r in results]),
        "w_lead_1only":    np.concatenate([r["w_lead_1only"]    for r in results]),
    }


def read_dataset(dataset_name: str, dataset_path: str, lumi: float, is_data: bool = False,
                 split_hem: bool = False, lumi_pre_hem: float = None,
                 lumi_hem: float = None, hem_run_threshold: int = None):
    """Read all ROOT files for *dataset_name* and accumulate eta-phi arrays.

    Returns a dataset dict, a (pre_hem_dict, hem_dict) tuple when split_hem=True, or None if empty.
    """
    print(f"\n  Processing {dataset_name} ...")
    files = get_file_list(dataset_path)
    if not files:
        print(f"    No files found.")
        return None
    print(f"    Found {len(files)} file(s)")

    raw = [None] * len(files)
    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_read_single_file, fp, lumi, is_data, split_hem,
                        lumi_pre_hem, lumi_hem, hem_run_threshold): i
            for i, fp in enumerate(files)
        }
        with tqdm(total=len(files), desc=f"    {dataset_name}", unit="file",
                  dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_idx):
                raw[future_to_idx[future]] = future.result()
                pbar.update(1)

    if split_hem:
        pre_results = [r[0] for r in raw if isinstance(r, tuple)]
        hem_results = [r[1] for r in raw if isinstance(r, tuple)]
        if not pre_results and not hem_results:
            print(f"    No data read.")
            return None
        result_pre = _build_dataset_dict(f"{dataset_name}_preHEM", pre_results) if pre_results else None
        result_hem = _build_dataset_dict(f"{dataset_name}_HEM",    hem_results) if hem_results else None
        if not is_data:
            ref = pre_results if pre_results else hem_results
            n_initial_total = sum(r["n_initial"] for r in ref)
            for wkey in ("w_lead", "w_sub", "w_lead_paired", "w_lead_1only"):
                if result_pre is not None:
                    result_pre[wkey] /= n_initial_total
                if result_hem is not None:
                    result_hem[wkey] /= n_initial_total
        if result_pre:
            print(f"    {dataset_name} (pre-HEM): {result_pre['n_ok']}/{len(files)} files ok")
        if result_hem:
            print(f"    {dataset_name} (HEM): {result_hem['n_ok']}/{len(files)} files ok")
        return (result_pre, result_hem)

    good = [r for r in raw if r is not None and not isinstance(r, tuple)]
    if not good:
        print(f"    No data read.")
        return None
    result = _build_dataset_dict(dataset_name, good)
    if not is_data:
        n_initial_total = sum(r["n_initial"] for r in good)
        for wkey in ("w_lead", "w_sub", "w_lead_paired", "w_lead_1only"):
            result[wkey] /= n_initial_total
    print(f"    {dataset_name}: {result['n_ok']}/{len(files)} files ok")
    return result


# ---------------------------------------------------------------------------
# Ntuple reader with on-the-fly pre-selection
# ---------------------------------------------------------------------------
# Mirrors SVJProcessing/analysis_configs/s_channel_scouting_pre_selection.py
# for data, minus JEC, HEM veto and the phi-spike filter (which is what we
# derive here).  The lepton veto only needs pt/eta/miniIso, so it is applied
# even if the electron/muon ID branches are missing (2016 ntuples lack
# Muon_isGlobal/isTracker, which makes the skimmer skip the veto entirely).

_JET_ID_FIELDS = [
    "pt", "eta", "phi", "mass",
    "chargedMultiplicity", "nConstituents",
    "chargedHadronEnergyFraction", "neutralHadronEnergyFraction",
    "photonEnergyFraction", "muonEnergyFraction", "electronEnergyFraction",
]
NTUPLE_BRANCHES = (
    ["run", "lumSec", "scouting_trig", "ScoutMET_pt", "ScoutMET_phi",
     "PV_isValidVtx", "PV_x", "PV_y", "PV_z",
     "Electron_pt", "Electron_eta", "Electron_combinedMiniIso",
     "Muon_pt", "Muon_eta", "Muon_combinedMiniIso"]
    + [f"FatJet_{f}" for f in _JET_ID_FIELDS]
    + [f"Jet_{f}" for f in _JET_ID_FIELDS]
)

# Cut-flow labels in the order they are applied
NTUPLE_CUTS = ["Initial", "GoldenJSON", "Trigger", "GoodJetsAK8", "nJetsAK8Gt2",
               "IsolatedLeptonVeto", "goodVerticesFilter", "GapJetVeto",
               "RT_selection", "MT_selection", "DeltaPhiMin_selection"]


def _load_golden_json(path: str) -> dict:
    import json
    with open(path) as fh:
        return {int(run): np.array(ranges) for run, ranges in json.load(fh).items()}


def _golden_json_mask(run: np.ndarray, lumi: np.ndarray, golden: dict) -> np.ndarray:
    mask = np.zeros(len(run), dtype=bool)
    for r in np.unique(run):
        ranges = golden.get(int(r))
        if ranges is None:
            continue
        sel = run == r
        ls = lumi[sel][:, None]
        mask[sel] = np.any((ls >= ranges[:, 0]) & (ls <= ranges[:, 1]), axis=1)
    return mask


def _jet_id(a, prefix: str):
    """Tight-like scouting jet ID (sequences_s_channel_scouting.add_*_jet_id_branch)."""
    return ((abs(a[f"{prefix}_eta"]) <= 2.6)
            & (a[f"{prefix}_chargedMultiplicity"] > 0)
            & (a[f"{prefix}_chargedHadronEnergyFraction"] > 0)
            & (a[f"{prefix}_neutralHadronEnergyFraction"] < 0.9)
            & (a[f"{prefix}_nConstituents"] > 1)
            & (a[f"{prefix}_photonEnergyFraction"] < 0.9)
            & (a[f"{prefix}_muonEnergyFraction"] < 0.8)
            & (a[f"{prefix}_electronEnergyFraction"] < 0.8))


def _select_ntuple_chunk(a, golden: dict, cut_flow: np.ndarray,
                         pv_require_valid: bool = True):
    """Apply the pre-selection to one chunk; return (selected events, good AK4 mask)."""
    def step(mask):
        nonlocal a
        a = a[mask]
        cut_flow[step.i] += len(a)
        step.i += 1
    step.i = 1
    cut_flow[0] += len(a)

    step(_golden_json_mask(ak.to_numpy(a["run"]), ak.to_numpy(a["lumSec"]), golden))
    step(a["scouting_trig"] == 1)

    # all analysis AK8 jets (pt>150, |eta|<2.4) must pass the jet ID
    ak8_analysis = (a["FatJet_pt"] > 150) & (abs(a["FatJet_eta"]) < 2.4)
    ak8_id = _jet_id(a, "FatJet")
    step(ak.all(ak8_id[ak8_analysis], axis=1))
    ak8_good = (a["FatJet_pt"] > 150) & (abs(a["FatJet_eta"]) < 2.4) & _jet_id(a, "FatJet")
    step(ak.sum(ak8_good, axis=1) >= 2)

    n_veto = (
        ak.sum((a["Electron_pt"] > 10) & (abs(a["Electron_eta"]) < 2.4)
               & (abs(a["Electron_combinedMiniIso"]) < 0.1), axis=1)
        + ak.sum((a["Muon_pt"] > 10) & (abs(a["Muon_eta"]) < 2.4)
                 & (abs(a["Muon_combinedMiniIso"]) < 0.4), axis=1)
    )
    step(n_veto == 0)

    good_pv = (abs(a["PV_z"]) <= 24) & ((a["PV_x"] ** 2 + a["PV_y"] ** 2) < 4)
    if pv_require_valid:
        good_pv = good_pv & (a["PV_isValidVtx"] == 1)
    step(ak.any(good_pv, axis=1))

    # gap jet veto: leading AK4 jet (any) with pt > 1000 and photon fraction > 0.7
    lead_pt  = ak.fill_none(ak.firsts(a["Jet_pt"]), 0)
    lead_phf = ak.fill_none(ak.firsts(a["Jet_photonEnergyFraction"]), 0)
    step(~((lead_pt > 1000) & (lead_phf > 0.7)))

    # MT / RT / DeltaPhiMin with the good AK8 jets and scouting MET
    good = (a["FatJet_pt"] > 150) & (abs(a["FatJet_eta"]) < 2.4) & _jet_id(a, "FatJet")
    pt, eta = a["FatJet_pt"][good], a["FatJet_eta"][good]
    phi, m  = a["FatJet_phi"][good], a["FatJet_mass"][good]
    px = ak.to_numpy(pt[:, 0] * np.cos(phi[:, 0]) + pt[:, 1] * np.cos(phi[:, 1]))
    py = ak.to_numpy(pt[:, 0] * np.sin(phi[:, 0]) + pt[:, 1] * np.sin(phi[:, 1]))
    pz = ak.to_numpy(pt[:, 0] * np.sinh(eta[:, 0]) + pt[:, 1] * np.sinh(eta[:, 1]))
    e0 = np.sqrt((pt[:, 0] * np.cosh(eta[:, 0])) ** 2 + m[:, 0] ** 2)
    e1 = np.sqrt((pt[:, 1] * np.cosh(eta[:, 1])) ** 2 + m[:, 1] ** 2)
    e  = ak.to_numpy(e0 + e1)
    m2_jj  = np.maximum(e ** 2 - px ** 2 - py ** 2 - pz ** 2, 0)
    met    = ak.to_numpy(a["ScoutMET_pt"])
    metphi = ak.to_numpy(a["ScoutMET_phi"])
    mt = np.sqrt(np.maximum(
        m2_jj + 2 * (np.sqrt(m2_jj + px ** 2 + py ** 2) * met
                     - (px * met * np.cos(metphi) + py * met * np.sin(metphi))), 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rt = np.where(mt > 0, met / mt, np.inf)
    dphi = abs((phi - a["ScoutMET_phi"] + np.pi) % (2 * np.pi) - np.pi)
    dphi_min = ak.to_numpy(ak.min(dphi, axis=1))

    a_rt = rt > 0.15
    step(a_rt)
    mt, dphi_min = mt[a_rt], dphi_min[a_rt]
    a_mt = mt > 650
    step(a_mt)
    step(dphi_min[a_mt] < 0.8)

    ak4_good = (a["Jet_pt"] > 30) & (abs(a["Jet_eta"]) < 2.4) & _jet_id(a, "Jet")
    return a, ak4_good


def _read_ntuple_file(file_path: str, golden: dict, cache_path: str,
                      pv_require_valid: bool = True):
    """Read one ntuple file, apply the pre-selection and cache the leading /
    subleading good AK4 jet positions per selected event (npz).

    Returns a dict with per-event arrays (run, n_good, eta0, phi0, eta1, phi1)
    plus the cut flow, or None on failure.
    """
    if os.path.exists(cache_path):
        with np.load(cache_path) as d:
            return {k: d[k] for k in d.files}

    name = Path(file_path).name
    for attempt in range(NTUPLE_RETRIES):
        cut_flow = np.zeros(len(NTUPLE_CUTS))
        parts = []
        try:
            _read_ntuple_chunks(file_path, golden, pv_require_valid, cut_flow, parts)
            break
        except Exception as exc:
            if attempt + 1 == NTUPLE_RETRIES:
                tqdm.write(f"    Error reading {name} (giving up after "
                           f"{NTUPLE_RETRIES} attempts): {exc}")
                return None
            time.sleep(NTUPLE_RETRY_WAIT * (attempt + 1))

    keys = ["run", "n_good", "eta0", "phi0", "eta1", "phi1"]
    res = {k: (np.concatenate([p[k] for p in parts]) if parts
               else np.zeros(0, dtype=np.float32)) for k in keys}
    res["cut_flow"] = cut_flow
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path + ".tmp.npz", **res)
    os.replace(cache_path + ".tmp.npz", cache_path)
    return res


def _read_ntuple_chunks(file_path, golden, pv_require_valid, cut_flow, parts):
    """Read *file_path* chunk-wise, fill *cut_flow* and append selected jets to *parts*."""
    with uproot.open(file_path, timeout=300) as f:
        tree = f[NTUPLE_TREE]
        if tree.num_entries > 0:
            for a in tree.iterate(NTUPLE_BRANCHES, step_size=NTUPLE_STEP_SIZE,
                                  library="ak"):
                sel, ak4_good = _select_ntuple_chunk(a, golden, cut_flow,
                                                     pv_require_valid)
                if len(sel) == 0:
                    continue
                jeta = ak.pad_none(sel["Jet_eta"][ak4_good], 2)
                jphi = ak.pad_none(sel["Jet_phi"][ak4_good], 2)
                parts.append({
                    "run":    ak.to_numpy(sel["run"]).astype(np.int32),
                    "n_good": ak.to_numpy(ak.sum(ak4_good, axis=1)).astype(np.int16),
                    "eta0":   ak.to_numpy(ak.fill_none(jeta[:, 0], np.nan)).astype(np.float32),
                    "phi0":   ak.to_numpy(ak.fill_none(jphi[:, 0], np.nan)).astype(np.float32),
                    "eta1":   ak.to_numpy(ak.fill_none(jeta[:, 1], np.nan)).astype(np.float32),
                    "phi1":   ak.to_numpy(ak.fill_none(jphi[:, 1], np.nan)).astype(np.float32),
                })


def _ntuple_events_to_dataset(name: str, ev: dict, n_ok: int) -> dict:
    """Turn cached per-event jet positions into the dataset dict used by the
    plotting / outlier code (same keys as _build_dataset_dict, data weights = 1)."""
    n = ev["n_good"]
    has_lead, has_sub, has_1only = n >= 1, n >= 2, n == 1
    d = {"name": name, "n_ok": n_ok}
    for tag, mask, idx in [("lead", has_lead, 0), ("sub", has_sub, 1),
                           ("lead_paired", has_sub, 0), ("lead_1only", has_1only, 0)]:
        d[f"eta_{tag}"] = ev[f"eta{idx}"][mask].astype(np.float64)
        d[f"phi_{tag}"] = ev[f"phi{idx}"][mask].astype(np.float64)
        d[f"w_{tag}"]   = np.ones(int(mask.sum()))
    return d


def read_ntuple_data(year: str, cfg: dict, max_files: int = None, workers: int = 4):
    """Read all eras for *year* from ntuples.

    Returns (per-event dict concatenated over all eras, total cut flow, n_ok).
    """
    golden = _load_golden_json(cfg["golden_json"])
    all_parts, cut_flow, n_ok_total = [], np.zeros(len(NTUPLE_CUTS)), 0
    missing = []   # human-readable list of what could not be read

    for era in cfg["datasets"]:
        print(f"\n  Processing {era} (ntuples) ...")
        files = []
        for attempt in range(NTUPLE_RETRIES):
            files = get_file_list(f"{cfg['base_path']}/{era}")
            if files:
                break
            if attempt + 1 < NTUPLE_RETRIES:
                print(f"    Listing empty/failed, retrying in "
                      f"{NTUPLE_RETRY_WAIT * (attempt + 1)} s ...")
                time.sleep(NTUPLE_RETRY_WAIT * (attempt + 1))
        if max_files:
            files = files[:max_files]
        if not files:
            print("    No files found.")
            missing.append(f"{era}: could not list files")
            continue
        print(f"    Found {len(files)} file(s)")

        cache_dir = os.path.join(NTUPLE_CACHE_DIR, year, era)
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_read_ntuple_file, fp, golden,
                                os.path.join(cache_dir, Path(fp).stem + ".npz"),
                                cfg.get("pv_require_valid", True))
                    for fp in files]
            with tqdm(total=len(files), desc=f"    {era}", unit="file",
                      dynamic_ncols=True) as pbar:
                for fut in as_completed(futs):
                    r = fut.result()
                    if r is not None:
                        results.append(r)
                    pbar.update(1)

        n_failed = len(files) - len(results)
        print(f"    {era}: {len(results)}/{len(files)} files ok"
              + (f"  ({n_failed} FAILED -- rerun to retry, cached files are skipped)"
                 if n_failed else ""))
        if n_failed:
            missing.append(f"{era}: {n_failed}/{len(files)} files failed")
        n_ok_total += len(results)
        for r in results:
            cut_flow += r["cut_flow"]
            all_parts.append(r)

    if not all_parts:
        return None, cut_flow, 0, missing
    ev = {k: np.concatenate([p[k] for p in all_parts])
          for k in ["run", "n_good", "eta0", "phi0", "eta1", "phi1"]}
    return ev, cut_flow, n_ok_total, missing


def _print_cut_flow(cut_flow: np.ndarray):
    print("\n  Cut flow (data, ntuple pre-selection):")
    for i, (label, n) in enumerate(zip(NTUPLE_CUTS, cut_flow)):
        eff = n / cut_flow[i - 1] if i and cut_flow[i - 1] else 1.0
        print(f"    {label:<24} {n:>14.0f}   ({eff:6.1%})")


# ---------------------------------------------------------------------------
# Histogram accumulation
# ---------------------------------------------------------------------------

def _accumulate_2d(datasets: list, slot: str):
    """
    Sum weighted 2D (eta, phi) histograms over all datasets in the list.
    *slot* is 'lead' or 'sub'.
    """
    h_total = np.zeros((len(ETA_BINS) - 1, len(PHI_BINS) - 1))
    for d in datasets:
        h, _, _ = np.histogram2d(
            d[f"eta_{slot}"],
            d[f"phi_{slot}"],
            bins=[ETA_BINS, PHI_BINS],
            weights=d[f"w_{slot}"],
        )
        h_total += h
    return h_total


def normalize_by_eta_bin(h2d: np.ndarray):
    """
    Normalize each eta bin to isolate outliers/enhancements.
    Assumes jets are isotropic in phi but not eta.
    
    For each eta bin (row) i:
    1. B'_iφ = B_iφ / max(B_iφ)  (divide by max)
    2. B''_iφ = max(0, B'_iφ - <B'_iφ>)  (subtract average, clip negatives)
    
    Args:
        h2d: 2D histogram with shape (n_eta_bins, n_phi_bins)
    
    Returns:
        Normalized histogram with same shape
    """
    h_norm = np.zeros_like(h2d)
    
    for i in range(h2d.shape[0]):  # Loop over eta bins
        eta_slice = h2d[i, :]
        
        # Step 1: Divide by max (avoid division by zero)
        max_val = np.max(eta_slice)
        if max_val > 0:
            normalized = eta_slice / max_val
        else:
            normalized = eta_slice
        
        # Step 2: Subtract average and clip negatives
        mean_val = np.mean(normalized)
        normalized = normalized - mean_val
        normalized = np.maximum(0, normalized)  # Clip negatives
        
        h_norm[i, :] = normalized
    
    return h_norm


def find_outlier_centers(h_norm: np.ndarray, n_sigma: float = N_SIGMA):
    """Return (eta_centers, phi_centers) arrays for bins above mean + n_sigma*std.

    Consistent with the outlier marking in plot_eta_phi_2d.
    """
    threshold = np.mean(h_norm) + n_sigma * np.std(h_norm)
    ii, jj = np.where(h_norm > threshold)
    eta_centers = 0.5 * (ETA_BINS[:-1] + ETA_BINS[1:])
    phi_centers = 0.5 * (PHI_BINS[:-1] + PHI_BINS[1:])
    return eta_centers[ii], phi_centers[jj]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eta_phi_2d(h2d: np.ndarray, jet_num: int, year: str, lumi: float, out_stem: str,
                    is_data: bool = False, normalized: bool = False, n_sigma: float = None,
                    output_dir: str = None, period: str = None):
    """Draw a 2D eta-phi map and save PNG + PDF.

    If *n_sigma* is provided the map is assumed already normalized and outlier bins are marked.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    fig, ax = plt.subplots(figsize=(10, 8))

    if normalized or n_sigma is not None:
        pcm = ax.pcolormesh(ETA_BINS, PHI_BINS, h2d.T, cmap="viridis", rasterized=True)
        cbar_label = r"Normalized enhancement"
    else:
        h_plot = h2d.T.copy()
        h_plot[h_plot <= 0] = np.nan
        vmin = np.nanmin(h_plot[h_plot > 0]) if np.any(h_plot > 0) else 1e-10
        pcm = ax.pcolormesh(ETA_BINS, PHI_BINS, h_plot, cmap="viridis",
                            norm=LogNorm(vmin=vmin), rasterized=True)
        cbar_label = r"Number of jets" if is_data else r"Weighted entries [fb$^{-1}$]"

    fig.colorbar(pcm, ax=ax, pad=0.01).set_label(cbar_label)

    ax.set_xlabel(rf"AK4 Jet {jet_num} $\eta$", ha="right", x=1.0)
    ax.set_ylabel(rf"AK4 Jet {jet_num} $\phi$", ha="right", y=1.0)

    title = (r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Data)}$" if is_data
             else r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Simulation)}$")
    ax.set_title(title, loc="left", pad=6)

    year_label = f"{year} {period}" if period else year
    ax.text(1, 1, rf"{lumi:.1f} fb$^{{-1}}$ ({year_label}, 13 TeV)",
            transform=ax.transAxes, ha="right", va="bottom")

    if n_sigma is not None:
        outlier_mask = h2d > np.mean(h2d) + n_sigma * np.std(h2d)
        eta_centers = 0.5 * (ETA_BINS[:-1] + ETA_BINS[1:])
        phi_centers = 0.5 * (PHI_BINS[:-1] + PHI_BINS[1:])
        for i, j in zip(*np.where(outlier_mask)):
            ax.plot(eta_centers[i], phi_centers[j], 'o',
                    markerfacecolor='none', markeredgecolor='white',
                    markersize=8, markeredgewidth=2)
        ax.text(0.02, 0.02, rf"{n_sigma:g}$\sigma$ outliers: {np.sum(outlier_mask)} bins",
                transform=ax.transAxes, ha="left", va="bottom", color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{out_stem}.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _collect_datasets(cfg: dict, is_data: bool):
    """Read all datasets for *cfg*, return (nosplit, pre_hem, post_hem) lists."""
    has_hem = cfg.get("hem_issue", False)
    nosplit, pre_hem, post_hem = [], [], []
    dataset_names = cfg["datasets"] if is_data else MC_DATASET_NAMES
    for dname in dataset_names:
        result = read_dataset(
            dname, f"{cfg['base_path']}/{dname}", cfg["lumi"],
            is_data=is_data,
            split_hem=has_hem,
            lumi_pre_hem=cfg.get("lumi_pre_hem"),
            lumi_hem=cfg.get("lumi_hem"),
            hem_run_threshold=cfg.get("hem_run_threshold"),
        )
        if isinstance(result, tuple):
            if result[0] is not None:
                pre_hem.append(result[0])
            if result[1] is not None:
                post_hem.append(result[1])
        elif result is not None:
            nosplit.append(result)
    return nosplit, pre_hem, post_hem


def _write_outlier_centers(year: str, period: str | None, out_dir: str,
                            src_info: list):
    """Write (or append) an outlier-centres file from *src_info*.

    *src_info* is a list of ``(src_tag, eta_arr, phi_arr)`` tuples.
    The file is named ``outlier_centers_{year}[_{period}].txt``.
    """
    eta_all = np.concatenate([e for _, e, _ in src_info])
    phi_all = np.concatenate([p for _, _, p in src_info])
    if eta_all.size == 0:
        return

    combined = np.unique(np.stack([eta_all, phi_all], axis=1), axis=0)
    eta_comb, phi_comb = combined[:, 0], combined[:, 1]
    print(f"  Combined jet-2 unique centres: {len(eta_comb)}"
          f"  (period={period or 'all'})")

    src_sets = {tag: set(zip(e.tolist(), p.tolist())) for tag, e, p in src_info}

    period_suffix = f"_{period}" if period else ""
    centers_path = os.path.join(out_dir, f"outlier_centers_{year}{period_suffix}.txt")
    with open(centers_path, "w") as fh:
        fh.write(f"# Jet-2 dead-cell outlier centres for {year}"
                 f"{' (' + period + ')' if period else ''}\n")
        fh.write(f"# N_SIGMA={N_SIGMA:g}  rad={DEAD_CELL_RAD:.6f}\n")
        fh.write("# eta_center   phi_center   source\n")
        for e, p in zip(eta_comb, phi_comb):
            key = (float(e), float(p))
            in_src = [s for s, st in src_sets.items() if key in st]
            fh.write(f"  {e:+.6f}   {p:+.6f}   {'&'.join(sorted(in_src))}\n")
    print(f"  Outlier centres saved to {centers_path}")


def _process_period(year: str,
                    mc_datasets: list, data_datasets: list,
                    lumi_mc: float, lumi_data: float,
                    out_dir: str, period: str = None):
    """Produce all plots and the outlier-centres file for one period.

    For each source (MC / Data) and each jet (leading / subleading):
      - raw eta-phi map
      - normalised eta-phi map
      - normalised map with N_SIGMA outlier markers

    Jet-2 outlier centres from MC and Data are combined, deduplicated, and
    written to outlier_centers_{year}[_period].txt with a source column.
    """
    if not mc_datasets and not data_datasets:
        print(f"  No datasets for {year} {period or ''}, skipping.")
        return

    period_suffix = f"_{period}" if period else ""
    os.makedirs(out_dir, exist_ok=True)

    sources = []
    if mc_datasets:
        sources.append((mc_datasets, "mc", lumi_mc, False))
    if data_datasets:
        sources.append((data_datasets, "data", lumi_data, True))

    jet2_eta_per_source, jet2_phi_per_source = [], []

    for datasets, src_tag, lumi, is_data in sources:
        h_lead = _accumulate_2d(datasets, "lead")
        h_sub  = _accumulate_2d(datasets, "sub")

        for jet_num, slot, h_raw in [(1, "lead", h_lead), (2, "sub", h_sub)]:
            jet_tag = "lead" if slot == "lead" else "sublead"
            stem    = f"etaphi_ak4_{jet_tag}_{year}_{src_tag}{period_suffix}"
            h_norm  = normalize_by_eta_bin(h_raw)

            plot_eta_phi_2d(h_raw,  jet_num, year, lumi, stem,
                            is_data=is_data, output_dir=out_dir, period=period)
            plot_eta_phi_2d(h_norm, jet_num, year, lumi, f"{stem}_normalized",
                            is_data=is_data, normalized=True,
                            output_dir=out_dir, period=period)
            plot_eta_phi_2d(h_norm, jet_num, year, lumi, f"{stem}_outliers",
                            is_data=is_data, n_sigma=N_SIGMA,
                            output_dir=out_dir, period=period)

            if slot == "sub":
                e, p = find_outlier_centers(h_norm, N_SIGMA)
                print(f"  {src_tag.upper()} jet-2 outliers: {len(e)} bins")
                jet2_eta_per_source.append(e)
                jet2_phi_per_source.append(p)

    # --- per-period outlier centres file ---
    if not jet2_eta_per_source:
        return []

    src_info = [(src_tag, eta_arr, phi_arr)
                for (_, src_tag, _, _), eta_arr, phi_arr
                in zip(sources, jet2_eta_per_source, jet2_phi_per_source)]

    _write_outlier_centers(year, period, out_dir, src_info)

    return src_info


def _run_ntuple_year(year: str, args):
    """Data-only dead-cell study from ntuples (pre-selection applied on the fly)."""
    cfg = NTUPLE_DATA_YEARS[year]
    # years that only exist as ntuples write to the standard output directory
    # (consumed by FilterEtaPhiMaps / SVJProcessing); validation runs do not
    out_root = OUTPUT_DIR if year not in DATA_YEARS else f"{OUTPUT_DIR}_ntuples"
    out_dir  = os.path.join(out_root, year)

    print(f"\n{'='*60}")
    print(f"  Year {year}  --  Data from ntuples, lumi = {cfg['lumi']} fb^-1"
          + (f"  (max {args.max_files} files/era)" if args.max_files else ""))
    print(f"{'='*60}")

    ev, cut_flow, n_ok, missing = read_ntuple_data(year, cfg, args.max_files, args.workers)
    _print_cut_flow(cut_flow)
    if missing:
        print(f"\n  INCOMPLETE input for {year}:")
        for m in missing:
            print(f"    - {m}")
        if not args.allow_missing:
            print("  No maps / outlier file written. Rerun (cached files are skipped) "
                  "or pass --allow-missing.")
            raise SystemExit(1)
        print("  --allow-missing given: continuing with partial input.")
    if ev is None:
        print(f"  No data for {year}, skipping.")
        return

    if args.vfp_split and "vfp_run_threshold" in cfg:
        is_pre = ev["run"] < cfg["vfp_run_threshold"]
        all_info = []
        for period, mask, lumi in [("preVFP",  is_pre,  cfg["lumi_pre_vfp"]),
                                   ("postVFP", ~is_pre, cfg["lumi_post_vfp"])]:
            sub = {k: v[mask] for k, v in ev.items()}
            ds  = _ntuple_events_to_dataset(f"{year}_{period}", sub, n_ok)
            all_info += _process_period(year, [], [ds], None, lumi,
                                        out_dir, period=period) or []
        if all_info:
            print(f"\n  Writing combined (preVFP + postVFP) filter list ...")
            _write_outlier_centers(year, None, out_dir, all_info)
    else:
        ds = _ntuple_events_to_dataset(year, ev, n_ok)
        _process_period(year, [], [ds], None, cfg["lumi"], out_dir, period=None)


def main():
    parser = argparse.ArgumentParser(
        description="Jet eta-phi dead-cell study: reads MC + Data, produces "
                    "raw, normalised, and outlier plots for both jets.",
        epilog="""
Examples:
  python DeadCellStudy.py                      # 2017 (default)
  python DeadCellStudy.py --years 2017 2018    # both years
  python DeadCellStudy.py --years 2016         # 2016 data from ntuples
  python DeadCellStudy.py --years 2017 --from-ntuples --max-files 20

Output per year (plots_deadcell/{year}/):
  etaphi_ak4_{lead|sublead}_{year}_{mc|data}[_period].{png,pdf}
  etaphi_ak4_{lead|sublead}_{year}_{mc|data}[_period]_normalized.{png,pdf}
  etaphi_ak4_{lead|sublead}_{year}_{mc|data}[_period]_outliers.{png,pdf}
  outlier_centers_{year}[_period].txt
        """
    )
    parser.add_argument("--years", nargs="+", default=["2017"],
                        help="Year(s) to process (default: 2017)")
    parser.add_argument("--from-ntuples", action="store_true",
                        help="Read data from the unskimmed ntuples with the pre-selection "
                             "applied on the fly, also for years that have skims "
                             "(validation; output goes to plots_deadcell_ntuples/). "
                             "Years without skims (2016) always use the ntuples.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Ntuple mode: only read the first N files per era (testing)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Ntuple mode: parallel file readers (default: 4)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Ntuple mode: write outputs even if some eras/files "
                             "could not be read (default: abort)")
    parser.add_argument("--vfp-split", action="store_true",
                        help="Ntuple mode, 2016: derive preVFP/postVFP maps separately "
                             "and write their union as the combined filter")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Processing years: {', '.join(args.years)}")
    print(f"{'='*60}")

    for year in args.years:
        if year in NTUPLE_DATA_YEARS and (args.from_ntuples or year not in DATA_YEARS):
            _run_ntuple_year(year, args)
            continue
        if year not in MC_YEARS or year not in DATA_YEARS:
            print(f"\nWarning: Year {year} not fully configured, skipping.")
            continue

        mc_cfg   = MC_YEARS[year]
        data_cfg = DATA_YEARS[year]
        has_hem  = mc_cfg.get("hem_issue", False) or data_cfg.get("hem_issue", False)
        out_dir  = os.path.join(OUTPUT_DIR, year)

        print(f"\n{'='*60}")
        print(f"  Year {year}  --  MC lumi = {mc_cfg['lumi']} fb^-1"
              f"  /  Data lumi = {data_cfg['lumi']} fb^-1")
        print(f"{'='*60}")

        print(f"\n  Reading MC for {year} ...")
        mc_nosplit, mc_pre, mc_hem = _collect_datasets(mc_cfg, is_data=False)

        print(f"\n  Reading Data for {year} ...")
        da_nosplit, da_pre, da_hem = _collect_datasets(data_cfg, is_data=True)

        if has_hem:
            pre_info = _process_period(year, mc_pre, da_pre,
                                       mc_cfg.get("lumi_pre_hem", mc_cfg["lumi"]),
                                       data_cfg.get("lumi_pre_hem", data_cfg["lumi"]),
                                       out_dir, period="preHEM")
            hem_info = _process_period(year, mc_hem, da_hem,
                                       mc_cfg.get("lumi_hem", mc_cfg["lumi"]),
                                       data_cfg.get("lumi_hem", data_cfg["lumi"]),
                                       out_dir, period="HEM")
            # Merge both periods into a single combined filter list
            all_info = pre_info + hem_info
            if all_info:
                print(f"\n  Writing combined (preHEM + HEM) filter list ...")
                _write_outlier_centers(year, None, out_dir, all_info)
        else:
            _process_period(year, mc_nosplit, da_nosplit,
                            mc_cfg["lumi"], data_cfg["lumi"],
                            out_dir, period=None)

    print("\nDone.")


if __name__ == "__main__":
    main()
