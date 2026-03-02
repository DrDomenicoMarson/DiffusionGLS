# Diffusion Coefficient Fitting

Python class library to determine the diffusion coefficient of a time series using a Generalized Least Squares (GLS) minimization procedure, which accounts for the correlation of MSD data.

Please read and cite the reference: J. Bullerjahn, S. v. Bülow, G. Hummer, Optimal estimates of diffusion coeffcients from molecular dynamics
simulations, Journal of Chemical Physics 153, 024116 (2020).

The input trajectory is analyzed as a whole and split into segments. For each segment, a quality factor Q is computed, indicating how well the trajectory fits a model of random diffusion with noise. The analysis is done for different time steps $\Delta t_n$ of the trajectory. Given the quality factor analysis, the user decides on a time step/diffusion coefficient pair to use.

## Installation

```bash
pip install -e .
```

## Basic Example

### From text files
```python
import Dfit

res = Dfit.Dcov(fz='mytrajectory.dat', m=20, tmin=1.0, tmax=100.0, nseg=150)
res.run_Dfit()
res.analysis(tc=10.0)
res.finite_size_correction(T=300, eta=0.001, L=10.0, tc=10.0)
```

### From MDAnalysis Universe
```python
import Dfit
import MDAnalysis as mda

u = mda.Universe('topology.tpr', 'trajectory.xtc')
res = Dfit.Dcov(universe=u, selection='resname SOL', tmax=50, dt=0.5)
res.run_Dfit()
res.analysis(tc=10.0)
```

### Multiple trajectory files (multi-molecule mode)
```python
res = Dfit.Dcov(fz=['traj1.dat', 'traj2.dat', 'traj3.dat'], tmax=50.0)
res.run_Dfit()
res.analysis(tc='auto')  # automatically select tc where Q ≈ 0.5
```

## Input for Dfit.Dcov()

### Trajectory input (one of):
* **fz** (str | Path | list): Filename(s) of input trajectory. Format: Center-of-mass position x [y z ...] in nm. Each row corresponds to a timestep indicated in `dt`. The number of columns determines the number of dimensions. No header; whitespace-separated columns. If a list is provided, each trajectory is treated as one molecule/segment.
* **universe** (MDAnalysis.Universe): MDAnalysis Universe with topology and trajectory loaded. Each residue in the selection is treated as a separate molecule.
* **selection** (str): MDAnalysis selection string (only used with `universe`).

### Optional parameters:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dt` | float | 1.0 | Timestep in units of `time_unit` |
| `m` | int | 20 | Number of MSD values used per lag step |
| `tmin` | float | None | Minimum lag time in `time_unit` (defaults to `dt`) |
| `tmax` | float | 100.0 | Maximum lag time in `time_unit` |
| `d2max` | float | 1e-10 | Convergence criterion for GLS iteration |
| `nitmax` | int | 100 | Maximum number of iterations in GLS procedure |
| `nseg` | int | None | Number of segments (default: auto, `N / (100*tmax)`) |
| `fout` | str | 'D_analysis' | Base name for output files (no extension) |
| `imgfmt` | str | 'pdf' | Output format for plot: `'pdf'` or `'png'` |
| `n_jobs` | int | -1 | Number of parallel workers. -1 = all CPUs |
| `time_unit` | str | 'ps' | Time unit for inputs/outputs: `'ps'` or `'ns'` |
| `diffusion_unit` | str | 'cm2/s' | Diffusion unit for outputs: `'cm2/s'` or `'nm2/ps'` |
| `normalize_lengths` | bool | False | Truncate multi-file trajectories to shortest length |
| `progress` | bool | True | Show progress bar during fitting |

## Running the fit

```python
res.run_Dfit(save_model=False)
```

* **save_model** (bool): If `True`, saves the full object to `{fout}.pkl` via pickle, allowing later loading with `Dcov.load('file.pkl')`.

## Analysis

```python
res.analysis(tc=10.0, fout_prefix=None)
```

* **tc** (float | `'auto'`): Lag time for the diffusion coefficient estimate. Must be a multiple of `dt`. Set to `'auto'` to automatically select the time point where Q is closest to 0.5.
* **fout_prefix** (str | None): Custom base name for output files. Default: `{fout}.tc_{tc}`.

The analysis can be repeated with different values of `tc`. A red vertical line indicates the chosen timestep in the plot.

## Finite-size correction

```python
res.finite_size_correction(T=300, eta=0.001, L=10.0, tc=10.0)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tc` | float | Lag time from the analysis step |
| `T` | float | Temperature in Kelvin |
| `eta` | float | Viscosity in Pa·s |
| `L` | float | Edge length of cubic simulation box in nm |
| `boxtype` | str | Only `'cubic'` currently supported |

## Output

### Files
* `{fout}.tc_{tc}.dat`: Summary including diffusion coefficient and Q-factor analysis.
* `{fout}.tc_{tc}.{imgfmt}`: Plots showing D(t), Q(t), and per-segment distribution.

### Stored attributes
| Attribute | Description |
|-----------|-------------|
| `res.D` | Optimal diffusion coefficient estimates per lag time |
| `res.Dstd` | Predicted standard deviation of D per lag time |
| `res.Dempstd` | Empirical standard deviation of D per lag time |
| `res.Dsem_pred` | Predicted SEM (= Dstd / √nseg) |
| `res.Dsem_emp` | Empirical SEM (= Dempstd / √nseg) |
| `res.q_m` | Mean quality factor per lag time |
| `res.q_std` | Std. dev. of quality factor per lag time |
| `res.tc_selected` | Selected tc value (after calling `analysis`) |
| `res.Dcor` | Finite-size corrected D (after calling `finite_size_correction`) |

## Persistence

```python
# Save after fitting
res.run_Dfit(save_model=True)

# Load later
res = Dfit.Dcov.load('D_analysis.pkl')
res.analysis(tc=10.0)
```
