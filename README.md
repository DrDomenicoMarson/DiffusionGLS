# Diffusion Coefficient Fitting

`Dfit` is a Python library for estimating translational diffusion
coefficients from trajectory data using the generalized least-squares (GLS)
framework from:

J. Bullerjahn, S. v. Bulow, G. Hummer, *Optimal estimates of diffusion
coefficients from molecular dynamics simulations*, J. Chem. Phys. 153, 024116
(2020).

The package evaluates diffusion as a function of lag time, quantifies
goodness-of-fit with a Q-factor, and reports both trajectory-level conditional
precision and reproducibility across independently prepared simulation boxes.

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
res.analysis(tc=10.0)  # or tc="auto", auto_min_tc=50.0
res.finite_size_correction(T=300, eta=0.001, L=10.0, tc=10.0)  # optional
```

Lifecycle constraints:

- `run_Dfit()` must be called before `analysis()`.
- `analysis()` must be called before `finite_size_correction()`.

## Capabilities

- Single trajectory analysis via automatic or user-defined segmentation.
- Multi-trajectory / multi-molecule analysis from a list of text trajectories.
- MDAnalysis integration (`Universe`/`AtomGroup`, single or pooled) with
  per-residue handling.
- Cluster-aware analysis for trajectories nested within independently prepared
  boxes or replicas.
- Diagnostic automatic lag-time suggestion with equal cluster weighting and an
  optional lower bound via `auto_min_tc`.
- Repeated analysis on the same fitted object with different `tc` values.
- Finite-size correction for 3D cubic boxes.
- Model persistence to and from pickle (`save_model=True`, `Dcov.load`).

## Input Modes

Use exactly one trajectory source:

- `fz`: text file path or list of paths.
- `universe`: MDAnalysis `Universe` or `AtomGroup`.
- `universes`: sequence of MDAnalysis `Universe`/`AtomGroup` objects to pool.

`selection` is used with MDAnalysis inputs when a `Universe` is supplied.

One `Dcov` object should represent one physical condition (for example, one
polymer/force-field/penetrant combination). Use `cluster_ids` to identify
independently prepared boxes within that condition:

- Every entry in `universes` is a separate cluster by default.
- Text files in `fz` belong to one cluster by default because independence
  cannot be inferred from filenames.
- Explicit `cluster_ids` align with Universes or text files. Repeated IDs group
  multiple sources into one cluster.
- Segments from one long trajectory remain in one cluster.

## Units and Conversions

- Internal time unit is ps.
- Internal diffusion unit is nm^2/ps.
- `time_unit` supports: `ps`, `ns`.
- `diffusion_unit` supports: `cm2/s`, `nm2/ps`.

`dt` behavior:

- Text files contain no time metadata, so `dt` is required.
- MDAnalysis inputs adopt the trajectory timestep when `dt=None`.
- If explicit `dt` differs from MDAnalysis metadata, a warning is emitted and
  the user-provided value is used.

## Constructor Parameters (`Dfit.Dcov`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fz` | `str \| Path \| Sequence[str \| Path] \| None` | `None` | Text trajectory input path(s). |
| `universe` | MDAnalysis object or `None` | `None` | MDAnalysis trajectory source. |
| `universes` | `Sequence[MDAnalysis object] \| None` | `None` | Multiple MDAnalysis trajectory sources pooled in one run. |
| `selection` | `str \| None` | `None` | MDAnalysis selection string for `Universe` input(s). |
| `cluster_ids` | `Sequence[object] \| None` | `None` | Cluster IDs aligned with input sources; repeated IDs group sources. |
| `m` | `int` | `20` | Number of MSD points per lag-time fit window. |
| `tmin` | `float \| None` | `None` | Minimum lag time in `time_unit`; defaults to one lag step. |
| `tmax` | `float` | `100.0` | Maximum lag time in `time_unit`. |
| `dt` | `float \| None` | `None` | Timestep in `time_unit`; required for text input and optional for MDAnalysis. |
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
result = res.analysis(tc=10.0, fout_prefix=None, auto_min_tc=None)
```

- `tc`: one common lag time in `time_unit` for all clusters. A reviewed numeric
  cutoff is recommended for publication.
- `tc="auto"`: diagnostic suggestion minimizing the equal-cluster mean of
  `abs(Q_cluster - 0.5)`. This is not an automatic diffusion-plateau test.
- `fout_prefix`: optional custom output prefix. Default is `{fout}.tc_{tc}`.
- `auto_min_tc`: optional lower bound for `tc="auto"` in `time_unit`; the
  auto search starts from the first computed lag `>= auto_min_tc`.

Analysis can be repeated with different `tc` values after a single fit run.
It returns an `AnalysisResult` dataclass containing pooled, per-cluster, and
across-cluster statistics.

### Interpreting uncertainty

- Pooled and per-cluster predicted/empirical SEMs quantify conditional
  precision among the trajectories in the sampled boxes. They do not make
  penetrants sharing one polymer matrix independent box realizations.
- The across-cluster estimate is the equal-weight mean of cluster estimates.
  Its sample SD describes observed box-to-box variability, and every cluster
  estimate is reported.
- The propagated within-cluster conditional SEM uses each cluster's error but
  is kept separate from the between-cluster sample SD. They are not added in
  quadrature because the observed cluster spread already contains estimation
  noise.
- Bootstrap, inverse-variance weighting, and random-effects meta-analysis are
  intentionally not applied by default, especially when only three boxes are
  available.

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

- `m` must be at least 2. With `m=2`, Q is unavailable and `tc="auto"` is
  rejected.
- Input trajectories must provide at least 2 frames.
- Text trajectories require a positive finite `dt`.
- In multi-trajectory mode, all dimensions must match.
- In multi-trajectory text mode, unequal lengths require
  `normalize_lengths=True`.
- In pooled MDAnalysis mode (`universes`), all inputs must share the same
  frame count and `dt`.
- Infeasible explicit `m`, `tmax`, and `nseg` combinations raise instead of
  being silently changed.
- Explicit feasible `nseg` is honored. If fewer than 100 strided coordinate
  samples remain at `tmax`, a sampling-adequacy warning reports the actual
  count and automatic recommendation.
- Single-trajectory segmentation uses every frame; segment lengths may differ
  by one frame.
- `tc` must lie on the computed lag grid and be a multiple of `dt`.
- `auto_min_tc` is only valid with `tc="auto"` and is rounded up to the first
  lag on the computed grid that satisfies the requested lower bound.

## Outputs

Generated files:

- `{fout}.tc_{tc}.dat`: concise selected-cutoff report with pooled, cluster,
  and across-cluster statistics.
- `{fout}.tc_{tc}.csv`: complete lag-dependent pooled, cluster, and
  across-cluster data.
- `{fout}.tc_{tc}.{imgfmt}`: plot with D(t), Q(t), and per-segment
  distribution augmented by cluster curves and means.
- `{fout}.pkl`: optional serialized object if `save_model=True`.

Key attributes and availability:

| Attribute | Available After | Description |
|---|---|---|
| `a2`, `s2`, `s2var`, `q` | `run_Dfit()` | Raw fit outputs over lag and segment dimensions. |
| `D`, `Dstd`, `Dempstd` | `analysis()` | Pooled diffusion estimate and predicted/empirical member SD per lag. |
| `Dsem_pred`, `Dsem_emp` | `analysis()` | Pooled predicted and empirical conditional SEM. |
| `q_m`, `q_std` | `analysis()` | Mean and std. deviation of Q-factor per lag. |
| `cluster_D`, `cluster_Dstd`, `cluster_Dempstd` | `analysis()` | Per-cluster estimate and member-level uncertainty series. |
| `cluster_Dsem_pred`, `cluster_Dsem_emp` | `analysis()` | Per-cluster conditional SEM series. |
| `D_cluster_mean`, `D_cluster_sd` | `analysis()` | Equal-weight cluster mean and between-cluster sample SD. |
| `D_cluster_sem_pred`, `D_cluster_sem_emp` | `analysis()` | Propagated within-cluster conditional SEM series. |
| `tc_selected`, `tc_selected_idx` | `analysis()` | Final lag-time value/index used for summary. |
| `tc_auto_unbounded`, `tc_auto_unbounded_idx`, `auto_min_tc_used` | `analysis()` | Unconstrained auto lag and applied lower-bound metadata when `tc="auto"` is used. |
| `Dcor` | `finite_size_correction()` | Finite-size corrected diffusion series. |

## Known Practical Limits

- MDAnalysis mode currently materializes full per-residue trajectories in
  memory; very large systems/trajectories can be memory intensive.
- The package currently exposes a Python API only (no command-line interface).

## Basic Examples

### 1) Text trajectory file

```python
import Dfit

res = Dfit.Dcov(
    fz="mytrajectory.dat",
    dt=1.0,
    m=20,
    tmin=1.0,
    tmax=100.0,
    nseg=150,
)
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

### 3) Pooled MDAnalysis trajectories (replicas)

```python
import Dfit
import MDAnalysis as mda

u1 = mda.Universe("topol1.tpr", "traj_box1.xtc")
u2 = mda.Universe("topol2.tpr", "traj_box2.xtc")
u3 = mda.Universe("topol3.tpr", "traj_box3.xtc")

res = Dfit.Dcov(
    universes=[u1, u2, u3],
    cluster_ids=["box_1", "box_2", "box_3"],  # optional; inferred by source
    selection="resname SOL",
    tmax=50,
)
res.run_Dfit()
result = res.analysis(tc=10.0)
print(result.across_clusters)
```

### 4) Multiple text trajectories (multi-molecule mode)

```python
import Dfit

res = Dfit.Dcov(
    fz=["box1_mol1.dat", "box1_mol2.dat", "box2_mol1.dat"],
    cluster_ids=["box_1", "box_1", "box_2"],
    dt=1.0,
    tmax=50.0,
)
res.run_Dfit()
res.analysis(tc=10.0)
```

## Persistence

```python
# Save after fitting
res.run_Dfit(save_model=True)

# Load later
res = Dfit.Dcov.load("D_analysis.pkl")
res.analysis(tc=10.0)
```
