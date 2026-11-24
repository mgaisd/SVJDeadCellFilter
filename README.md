# SVJDeadCellFilter

Custom filter against artificial MET from ECAL dead cells for the SVJ **scouting** analysis.
Similar custom filters exist for the s-channel and t-channel analyses, but not for scouting.

The code should work with the [SVJ conda environment](https://github.com/jniedzie/SVJanalysis_wiki/wiki/Creating-SVJ-virtual-environment).
Requirements: `uproot numpy matplotlib mplhep awkward tqdm`, plus the XRootD client (`xrdfs`) for remote file access.

## How the filter works

1. Fill eta-phi maps of the **subleading good AK4 jet** (50 x 50 bins over |eta| < 2.4, full phi) for QCD MC and data.
2. Normalise each eta slice and flag bins more than `N_SIGMA = 4` standard deviations above the slice mean as hot/dead-cell bins.
3. Write the bin centres to `outlier_centers_{year}.txt`. The `source` column says whether a bin was flagged in data, MC or both.
4. Veto an event if its subleading AK4 jet satisfies `deta^2 + dphi^2 < 0.028816 * 0.35` (≈ 0.0101) with respect to any of these centres.
   Here 0.028816 is the bin size squared, and the factor 0.35 was chosen by `radiusOptimization.py`.

For 2018 the pre-HEM and HEM periods are derived separately, and `outlier_centers_2018.txt` is their union.
2016 has no skims, so it is derived from the unskimmed ntuples with the s-channel scouting pre-selection applied on the fly.

## Filter files (the actual result)

```
plots_deadcell/2016/outlier_centers_2016.txt
plots_deadcell/2017/outlier_centers_2017.txt
plots_deadcell/2018/outlier_centers_2018.txt          # union, use this one
plots_deadcell/2018/outlier_centers_2018_preHEM.txt
plots_deadcell/2018/outlier_centers_2018_HEM.txt
plots_deadcell_ntuples/2017/outlier_centers_2017.txt  # 2017 re-derived from ntuples (cross-check only)
```

Format: header lines start with `#`, then one `eta_center  phi_center  source` per line.
The same files are used in production in
[SVJProcessing](https://github.com/CMS-SVJ-scouting/SVJProcessing) under `data/dead_cells/`, through
`apply_scouting_phi_spike_filter()` in `analysis_configs/sequences_s_channel_scouting.py`.
2016APV uses the 2016 file.

### Minimal example of applying the filter

```python
import numpy as np, awkward as ak

def load_centers(path):
    c = np.loadtxt(path, usecols=(0, 1), comments="#", ndmin=2)
    return c[:, 0], c[:, 1]

def dead_cell_mask(jet_eta, jet_phi, eta_c, phi_c, rad=0.028816 * 0.35):
    """True for events that PASS the filter. jet_eta/phi: good AK4 jets (jagged)."""
    eta = ak.to_numpy(ak.fill_none(ak.pad_none(jet_eta, 2)[:, 1], 99.0))  # <2 jets: always pass
    phi = ak.to_numpy(ak.fill_none(ak.pad_none(jet_phi, 2)[:, 1], 99.0))  # <2 jets: always pass
    deta = eta[:, None] - eta_c[None, :]
    dphi = (phi[:, None] - phi_c[None, :] + np.pi) % (2 * np.pi) - np.pi
    return ~np.any(deta**2 + dphi**2 < rad, axis=1)
```

## Scripts

Run them in the order below. Input paths (KIT dCache skims and ntuples) and the golden JSON directory are set at the top of each script.
Edit `base_path` / `GOLDEN_JSON_DIR` if you don't have access to them.

### 1. `DeadCellStudy.py`: derive the filter

```bash
python DeadCellStudy.py                       # 2017 (default)
python DeadCellStudy.py --years 2017 2018     # 2018 is split into preHEM/HEM automatically
python DeadCellStudy.py --years 2016          # 2016 data from ntuples (cached in cache_ntuples/, resumes if interrupted)
python DeadCellStudy.py --years 2016 --vfp-split          # derive preVFP/postVFP separately, write their union
python DeadCellStudy.py --years 2017 --from-ntuples       # re-derive from ntuples (output: plots_deadcell_ntuples/)
```

Ntuple-mode options: `--max-files N` (testing), `--workers N` (parallel readers, default 4) and `--allow-missing` (write output even if some files failed).

Output in `plots_deadcell/{year}/`: raw, normalised and outlier-marked eta-phi maps for the leading and subleading jet (MC and data), plus `outlier_centers_*.txt`.

### 2. `FilterEtaPhiMaps.py`: plot the filter and its efficiency

```bash
python FilterEtaPhiMaps.py                    # 2017 and 2018 (default)
python FilterEtaPhiMaps.py --years 2018
python FilterEtaPhiMaps.py --years 2016       # data only, uses the DeadCellStudy ntuple cache
```

Reads the `outlier_centers_*.txt` files. It produces normalised subleading-jet eta-phi maps for QCD and data with the vetoed regions drawn as circles
(and the HEM region for 2018 post-HEM), plus filter efficiencies for QCD and all signal points.
Output: `plots_filter/{year}/`, with efficiencies in `efficiencies_{year}.csv`.

### 3. Working-point optimisation (optional)

```bash
python nSigmaOptimization.py      # scan N_SIGMA (2 to 7.5), FOM = sqrt(2((S+B)ln(1+S/B) - S))  -> plots_nSigmaOpt/
python radiusOptimization.py      # scan veto-radius factor (0.05 to 1.2)                     -> plots_radiusOpt/
```

Both rebuild the QCD map on the fly from the skims, and they cover 2017 and 2018 pre-HEM.

### 4. `CompareDeadCellFilters.py`: cross-check two filter lists

```bash
python CompareDeadCellFilters.py --year 2017
python CompareDeadCellFilters.py --year 2017 --ref A.txt --new B.txt --labels A B --out outdir
```

By default it compares the skim-based list (`plots_deadcell/`) with the ntuple-based one (`plots_deadcell_ntuples/`).
It reports common and unique bins, an eta-phi overlay and the fraction of vetoed events. Run `DeadCellStudy.py --from-ntuples` first, since this needs `cache_ntuples/{year}`.

## Notes

- These jobs read many files over XRootD. Run them on a portal or batch node, not on a desktop machine.
- Plots and caches are git-ignored. Only the `outlier_centers_*.txt` files are tracked.
