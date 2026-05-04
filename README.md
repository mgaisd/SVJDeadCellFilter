# SVJDeadCellFilter

This repo is meant to create a custom filter for artificial MET due to ECAL dead cells. Similar custom filters have been made for s-channel and t-channel analyses, but not for scouting.

The code should work with the [SVJ conda environment](https://github.com/jniedzie/SVJanalysis_wiki/wiki/Creating-SVJ-virtual-environment).

## DeadCellStudy.py

### Overview

`DeadCellStudy.py` is a Python script that analyzes jet distributions in eta-phi space to identify potential dead cells or problematic regions in the CMS calorimeter. The script processes ROOT files from either Monte Carlo simulations or real collision data and creates 2D distribution maps.

### What it does

- **Reads ROOT files**: Accesses files remotely via XRootD or locally
- **Extracts jet kinematics**: Retrieves eta and phi coordinates for AK4 jets
- **Applies proper weighting**: 
  - For MC: Cross-section weighted to integrated luminosity
  - For data: Simple event counting
- **Creates 2D histograms**: Shows distribution of jets in eta-phi space
- **Generates plots**: Publication-quality CMS-style plots in PNG and PDF formats
- **Parallel processing**: Efficiently processes multiple files simultaneously

The script analyzes both leading and subleading jets separately, which can help identify regions with abnormal jet activity that might indicate detector issues.

### Requirements

```bash
pip install uproot numpy matplotlib mplhep awkward tqdm
```

You also need XRootD client tools installed on your system for remote file access:
```bash
# On Ubuntu/Debian
sudo apt-get install xrootd-client

# On RHEL/CentOS
sudo yum install xrootd-client
```

### Usage

#### Basic Usage

**Process Monte Carlo samples (default):**
```bash
python DeadCellStudy.py
```

**Process real data:**
```bash
python DeadCellStudy.py --data
```

#### Selecting Years

**Process a specific year:**
```bash
python DeadCellStudy.py --years 2017
```

**Process multiple years:**
```bash
python DeadCellStudy.py --years 2016 2017 2018
```

**Combine options (data + specific years):**
```bash
python DeadCellStudy.py --data --years 2017
```

#### Command-Line Options

- `--data`: Process real collision data instead of Monte Carlo simulation
- `--years YEAR [YEAR ...]`: Specify which year(s) to process (default: 2017)

#### Help

```bash
python DeadCellStudy.py --help
```

### Configuration

Before running, you may need to adjust the configuration in the script:

1. **File paths**: Update `MC_YEARS` and `DATA_YEARS` dictionaries with correct paths to your datasets
2. **Dataset names**: Modify `MC_DATASET_NAMES` or `DATA_DATASET_NAMES` to match your file structure
3. **Jet branches**: If your ROOT files use different branch names, update `JET_ETA_BRANCH` and `JET_PHI_BRANCH`
4. **Binning**: Adjust `ETA_BINS` and `PHI_BINS` for different histogram granularity
5. **Parallel workers**: Change `FILE_WORKERS` to control the number of parallel file readers

### Output

The script creates a `plots_deadcell/` directory with the following files:

- `etaphi_ak4_lead_YEAR_mc.png/pdf`: Leading jet distributions for MC
- `etaphi_ak4_sublead_YEAR_mc.png/pdf`: Subleading jet distributions for MC  
- `etaphi_ak4_lead_YEAR_data.png/pdf`: Leading jet distributions for data
- `etaphi_ak4_sublead_YEAR_data.png/pdf`: Subleading jet distributions for data

Each plot shows:
- 2D histogram of jet eta vs phi
- Color scale indicating jet density (weighted for MC, raw counts for data)
- CMS style formatting
- Year and luminosity information

### Example Workflow

1. **Check MC distributions first:**
   ```bash
   python DeadCellStudy.py --years 2017
   ```

2. **Compare with data:**
   ```bash
   python DeadCellStudy.py --data --years 2017
   ```

3. **Look for discrepancies** in the eta-phi plots that might indicate dead cells or detector issues

### Troubleshooting

- **XRootD connection issues**: Verify you can access the remote storage with `xrdfs`
- **Missing branches**: Check that your ROOT files contain the required branches (`Jet_eta`, `Jet_phi`)
- **Memory issues**: Reduce `FILE_WORKERS` if you encounter memory problems
- **No files found**: Verify the paths in `MC_YEARS` or `DATA_YEARS` dictionaries

### Notes

- The script uses parallel processing to speed up file reading - progress bars show real-time status
- MC samples are weighted by cross-section and luminosity for proper normalization
- Data samples use simple event counting (unit weights)
- The script skips files that cannot be read and continues processing remaining files