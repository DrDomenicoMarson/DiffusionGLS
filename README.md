# Diffusion Coefficient Fitting

`Dfit` is a Python library for estimating translational diffusion
coefficients from trajectory data using the generalized least-squares (GLS)
framework from:

J. Bullerjahn, S. v. Bulow, G. Hummer, *Optimal estimates of diffusion
coefficients from molecular dynamics simulations*, J. Chem. Phys. 153, 024116
(2020).

The package evaluates diffusion as a function of lag time, quantifies
goodness-of-fit with a Q-factor, and reports uncertainty estimates.

## Installation

```bash
pip install -e .
```

## Workflow Order

The API is Python-only (no CLI). A standard workflow is:

```python
import Dfit

res = Dfit.Dcov(...)
res.run_Dfit()
res.analysis(tc=10.0)  # or tc="auto"
res.finite_size_correction(T=300, eta=0.001, L=10.0, tc=10.0)  # optional
```

Lifecycle constraints:

- `run_Dfit()` must be called before `analysis()`.
- `analysis()` must be called before `finite_size_correction()`.

## Capabilities

- Single trajectory analysis via automatic or user-defined segmentation.
- Multi-trajectory / multi-molecule analysis from a list of text trajectories.
- MDAnalysis integration (`Universe` or `AtomGroup`) with per-residue handling.
- Automatic lag-time choice with `tc="auto"` (Q-factor target near 0.5).
- Repeated analysis on the same fitted object with different `tc` values.
- Finite-size correction for 3D cubic boxes.
- Model persistence to and from pickle (`save_model=True`, `Dcov.load`).

## Input Modes

Use exactly one trajectory source:

- `fz`: text file path or list of paths.
- `universe`: MDAnalysis `Universe` or `AtomGroup`.

`selection` is only used with `universe` when a `Universe` is supplied.

## Units and Conversions

- Internal time unit is ps.
- Internal diffusion unit is nm^2/ps.
- `time_unit` supports: `ps`, `ns`.
- `diffusion_unit` supports: `cm2/s`, `nm2/ps`.

`dt` behavior:

- `dt=None` (default): adopt timestep from the reader.
- Text files (`fz`): reader default is `1.0 ps`.
- MDAnalysis inputs: reader timestep comes from trajectory metadata.
- If explicit `dt` is provided and differs from reader timestep, a warning is
  emitted and the user-provided value is used.

## Constructor Parameters (`Dfit.Dcov`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fz` | `str \| Path \| Sequence[str \| Path] \| None` | `None` | Text trajectory input path(s). |
| `universe` | MDAnalysis object or `None` | `None` | MDAnalysis trajectory source. |
| `selection` | `str \| None` | `None` | MDAnalysis selection string for `Universe` input. |
| `m` | `int` | `20` | Number of MSD points per lag-time fit window. |
| `tmin` | `float \| None` | `None` | Minimum lag time in `time_unit`; defaults to one lag step. |
| `tmax` | `float` | `100.0` | Maximum lag time in `time_unit`. |
| `dt` | `float \| None` | `None` | Timestep in `time_unit`; `None` uses reader timestep. |
| `d2max` | `float` | `1e-10` | Iterative GLS convergence threshold. |
| `nitmax` | `int` | `100` | Maximum GLS iterations. |
| `nseg` | `int \| None` | `None` | Number of segments (single-trajectory mode). |
| `imgfmt` | `str` | `'pdf'` | Plot format (`pdf` or `png`). |
| `fout` | `str` | `'D_analysis'` | Output filename prefix. |
| `n_jobs` | `int` | `-1` | Worker count (`0` means serial). |
| `normalize_lengths` | `bool` | `False` | Truncate unequal text trajectories to shortest length. |
| `time_unit` | `str` | `'ps'` | Time unit for API input/output values. |
| `diffusion_unit` | `str` | `'cm2/s'` | Diffusion unit for reported values. |
| `progress` | `bool` | `True` | Enable lag-step progress bar. |

## Running the Fit

```python
res.run_Dfit(save_model=False)
```

- Performs GLS fitting across lag times and segments/molecules.
- Populates model arrays (`a2`, `s2`, `q`, `s2var`) and diagnostics.
- Optional `save_model=True` writes `{fout}.pkl`.

## Analysis

```python
res.analysis(tc=10.0, fout_prefix=None)
```

- `tc`: lag time in `time_unit`, or `"auto"` to choose lag with Q nearest 0.5.
- `fout_prefix`: optional custom output prefix. Default is `{fout}.tc_{tc}`.

Analysis can be repeated with different `tc` values after a single fit run.

## Finite-Size Correction

```python
res.finite_size_correction(T=300, eta=0.001, L=10.0, tc=10.0)
```

Constraints:

- Requires `analysis()` to have run first.
- Only implemented for 3D data.
- Only `boxtype='cubic'` is supported.

Parameters:

| Parameter | Type | Description |
|---|---|---|
| `tc` | `float` | Lag time (in `time_unit`) from analysis grid. |
| `T` | `float` | Temperature in Kelvin. |
| `eta` | `float` | Viscosity in Pa*s. |
| `L` | `float` | Cubic box edge length in nm. |
| `boxtype` | `str` | Box geometry (`'cubic'` only). |

## Validation and Constraints

- `m` must be at least 2 after internal clamping.
- Input trajectories must provide at least 2 frames.
- In multi-trajectory mode, all dimensions must match.
- In multi-trajectory text mode, unequal lengths require
  `normalize_lengths=True`.
- In multi mode, very large `tmax` may be clamped to a feasible value.
- `tc` must lie on the computed lag grid and be a multiple of `dt`.

## Outputs

Generated files:

- `{fout}.tc_{tc}.dat`: text summary (statistics + lag scan table).
- `{fout}.tc_{tc}.{imgfmt}`: plot with D(t), Q(t), and per-segment
  distribution.
- `{fout}.pkl`: optional serialized object if `save_model=True`.

Key attributes and availability:

| Attribute | Available After | Description |
|---|---|---|
| `a2`, `s2`, `s2var`, `q` | `run_Dfit()` | Raw fit outputs over lag and segment dimensions. |
| `D`, `Dstd`, `Dempstd` | `analysis()` | Main diffusion estimate and uncertainties per lag. |
| `Dsem_pred`, `Dsem_emp` | `analysis()` | Predicted and empirical SEM estimates. |
| `q_m`, `q_std` | `analysis()` | Mean and std. deviation of Q-factor per lag. |
| `tc_selected`, `tc_selected_idx` | `analysis()` | Selected lag-time value/index used for summary. |
| `Dcor` | `finite_size_correction()` | Finite-size corrected diffusion series. |

## Known Practical Limits

- MDAnalysis mode currently materializes full per-residue trajectories in
  memory; very large systems/trajectories can be memory intensive.
- The package currently exposes a Python API only (no command-line interface).

## Basic Examples

### 1) Text trajectory file

```python
import Dfit

res = Dfit.Dcov(fz="mytrajectory.dat", m=20, tmin=1.0, tmax=100.0, nseg=150)
res.run_Dfit()
res.analysis(tc=10.0)
res.finite_size_correction(T=300, eta=0.001, L=10.0, tc=10.0)
```

### 2) MDAnalysis trajectory

```python
import Dfit
import MDAnalysis as mda

u = mda.Universe("topology.tpr", "trajectory.xtc")
res = Dfit.Dcov(universe=u, selection="resname SOL", tmax=50)
res.run_Dfit()
res.analysis(tc=10.0)
```

### 3) Multiple text trajectories (multi-molecule mode)

```python
import Dfit

res = Dfit.Dcov(fz=["traj1.dat", "traj2.dat", "traj3.dat"], tmax=50.0)
res.run_Dfit()
res.analysis(tc="auto")
```

## Persistence

```python
# Save after fitting
res.run_Dfit(save_model=True)

# Load later
res = Dfit.Dcov.load("D_analysis.pkl")
res.analysis(tc=10.0)
```
