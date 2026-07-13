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
suggestion = res.suggest_tc(validation_window=10.0, min_tc=50.0)  # optional
if suggestion.tc is None:
    raise RuntimeError("No common plateau; inspect cutoff diagnostics.")
result = res.analysis(tc=suggestion.tc)  # explicit reviewed numeric cutoff
res.finite_size_correction(T=300, eta=0.001, L=10.0, tc=result.tc)  # optional
```

Lifecycle constraints:

- `run_Dfit()` must be called before `suggest_tc()`.
- `run_Dfit()` must be called before `analysis()`.
- `analysis()` must be called before `finite_size_correction()`.

## Capabilities

- Single trajectory analysis via automatic or user-defined segmentation.
- Multi-trajectory / multi-molecule analysis from a list of text trajectories.
- MDAnalysis integration (`Universe`/`AtomGroup`, single or pooled) with
  per-residue handling.
- Cluster-aware analysis for trajectories nested within independently prepared
  boxes or replicas.
- Auditable common-cutoff suggestion requiring sustained diffusion and Q-factor
  agreement in every cluster.
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

## Publication Cutoff Suggestion

```python
suggestion = res.suggest_tc(
    validation_window=10.0,
    min_tc=50.0,
    relative_drift_tolerance=0.05,
    q_tolerance=0.10,
    persistence_windows=2,
    blocks_per_window=5,
    candidate_step=None,
)

if suggestion.tc is None:
    raise RuntimeError("No common plateau passed; inspect the diagnostics.")

# Deliberately make the reviewed value explicit in the final analysis.
result = res.analysis(tc=suggestion.tc)
```

`suggest_tc()` is separate from `analysis()` by design. It produces a
reproducible, inspectable suggestion but does not certify that a cutoff is
appropriate for a particular scientific claim. The final report should use a
reviewed numeric `tc`.

### Selection rule

One `Dcov` object represents one physical condition, and a common cutoff is
selected only among the clusters/boxes in that object. For each candidate lag
and cluster, every validation window is divided into contiguous blocks. If
`Dbar[c,w,b]` and `Qbar[c,w,b]` are the block means for cluster `c`, consecutive
window `w`, and block `b`, the diagnostics are

```text
relative_D_drift[c,w] =
    (max_b Dbar[c,w,b] - min_b Dbar[c,w,b]) / abs(mean_b Dbar[c,w,b])

Q_deviation[c,w] = max_b abs(Qbar[c,w,b] - 0.5)
```

A cluster passes a candidate only when, in every consecutive validation
window:

- `relative_D_drift <= relative_drift_tolerance`; and
- `Q_deviation <= q_tolerance`.

The procedure records the earliest passing candidate for every cluster. It
then selects the earliest candidate at or after the latest individual onset
where all clusters pass simultaneously. This revalidation matters because a
cluster can leave an acceptable region at longer lag times. If no common
candidate passes, `suggestion.tc` is `None`; the closest failing candidate is
reported for diagnosis but is never silently selected.

Block means reduce sensitivity to isolated lag-grid fluctuations. They do not
make neighboring lag points independent observations, and the procedure does
not calculate slope p-values or treat failure to reject a trend as evidence of
a plateau.

### Parameters and defaults

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `validation_window` | `float` | required | Physical width of each validation window in `time_unit`; must be a multiple of `dt`. Choose and report a scientifically meaningful duration. |
| `min_tc` | `float \| None` | `None` | Lower search bound. It is rounded up to the first stored lag; `None` starts at `tmin`. |
| `relative_drift_tolerance` | `float` | `0.05` | Maximum 5% block-mean relative range of `D(t)` in every window. This is a pragmatic equivalence band, not a universal threshold. |
| `q_tolerance` | `float` | `0.10` | Requires every block-mean cluster Q value to remain in `[0.4, 0.6]`. |
| `persistence_windows` | `int` | `2` | Number of consecutive validation windows that must pass; neighboring closed windows share only their boundary lag. |
| `blocks_per_window` | `int` | `5` | Number of contiguous block means per window. |
| `candidate_step` | `float \| None` | `None` | Nominal candidate spacing in `time_unit`; `None` uses one tenth of `validation_window`, rounded to at least one frame. The final feasible candidate is also evaluated if off-spacing. |
| `fout_prefix` | `str \| None` | `None` | Diagnostic output base; defaults to `{fout}.tc_suggestion`. |
| `make_plot` | `bool` | `True` | Write the cutoff-diagnostic plot. |

The defaults are intentionally transparent rather than statistically
automatic. For a paper, report the chosen validation window and tolerances and
repeat the suggestion with at least one stricter and one looser practical
drift tolerance. A large shift in the suggested cutoff is evidence that the
plateau is not well determined. Use distinct `fout_prefix` values so the
sensitivity runs do not overwrite one another.

### Role of uncertainty

The candidate report includes each cluster's predicted and empirical
within-cluster conditional SEM at every candidate. These values contextualize
the precision of `D_c(tc)`, but they are not used as inverse-variance weights
and do not relax the plateau criterion. Doing so could allow a noisy or
trajectory-rich box to hide systematic drift or dominate the other boxes.

`suggest_tc()` requires `m >= 3` because Q is undefined for `m=2`. A requested
validation horizon must fit completely inside the calculated `tmin..tmax` lag
range. In particular, two 10 ns persistence windows require at least 20 ns of
calculated lag range after a candidate. The latest feasible candidate is

```text
latest_candidate = tmax - persistence_windows * validation_window
```

up to lag-grid rounding. For example, `tmax=100 ns`, two 10 ns windows, and
`m=10` can only evaluate candidates through 80 ns. Evaluating a 95 ns cutoff
would leave only 5 ns for validation; this requires an explicitly shorter
horizon and should be described as a weaker stability check.

### Return contract

`suggest_tc()` returns an immutable `TcSuggestion` dataclass:

- `tc`: suggested numeric cutoff, or `None` when no common plateau passes;
- `status`: `"suggested"` or `"no_common_plateau"`;
- `cluster_onsets`: immutable `ClusterTcOnset` entries for every box;
- `selected_candidate`: selected `TcCandidateDiagnostic`, or `None`;
- `closest_candidate`: lowest-violation diagnostic candidate, never an
  implicit fallback;
- `candidates`: complete immutable candidate diagnostics;
- the applied units, lower bound, window, persistence, block, candidate-step,
  and tolerance settings.

Each `TcCandidateDiagnostic` contains the candidate time, validation-horizon
end, common pass flag, worst normalized violation, and one
`ClusterTcCandidateDiagnostic` per box. The cluster diagnostic contains
`D_c(tc)`, predicted/empirical conditional SEM, maximum relative D drift,
maximum Q deviation, separate criterion flags, and the combined pass flag.

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

- `m` must be at least 2. With `m=2`, Q is unavailable and both `tc="auto"`
  and `suggest_tc()` are rejected.
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

- `{fout}.tc_suggestion.dat`: cutoff rule, status, individual cluster onsets,
  and selected or closest-failing candidate diagnostics.
- `{fout}.tc_suggestion.csv`: complete candidate-by-cluster diagnostics,
  thresholds, conditional SEMs, and pass/fail columns.
- `{fout}.tc_suggestion.{imgfmt}`: cluster `D(t)`/Q curves, Q acceptance band,
  cluster-onset markers, and the selected validation horizon.
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
| `D`, `Dstd`, `Dempstd` | `suggest_tc()` or `analysis()` | Pooled diffusion estimate and predicted/empirical member SD per lag. |
| `Dsem_pred`, `Dsem_emp` | `suggest_tc()` or `analysis()` | Pooled predicted and empirical conditional SEM. |
| `q_m`, `q_std` | `suggest_tc()` or `analysis()` | Mean and std. deviation of Q-factor per lag. |
| `tc_suggestion` | `suggest_tc()` | Latest `TcSuggestion`, including all candidate and cluster diagnostics. |
| `cluster_D`, `cluster_Dstd`, `cluster_Dempstd` | `suggest_tc()` or `analysis()` | Per-cluster estimate and member-level uncertainty series. |
| `cluster_Dsem_pred`, `cluster_Dsem_emp` | `suggest_tc()` or `analysis()` | Per-cluster conditional SEM series. |
| `D_cluster_mean`, `D_cluster_sd` | `suggest_tc()` or `analysis()` | Equal-weight cluster mean and between-cluster sample SD. |
| `D_cluster_sem_pred`, `D_cluster_sem_emp` | `suggest_tc()` or `analysis()` | Propagated within-cluster conditional SEM series. |
| `tc_selected`, `tc_selected_idx` | `analysis()` | Final lag-time value/index used for summary. |
| `tc_auto_unbounded`, `tc_auto_unbounded_idx`, `auto_min_tc_used` | `analysis()` | Unconstrained auto lag and applied lower-bound metadata when `tc="auto"` is used. |
| `Dcor` | `finite_size_correction()` | Finite-size corrected diffusion series. |

## Known Practical Limits

- MDAnalysis mode currently materializes full per-residue trajectories in
  memory; very large systems/trajectories can be memory intensive.
- The package does not install a command-line interface; the `Example`
  directory includes a standalone fitted-pickle helper script.

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
    tmax=100,
    time_unit="ns",
)
res.run_Dfit()
result = res.analysis(tc=75.0)
print(result.across_clusters)
```

For a publication-oriented common-cutoff suggestion:

```python
suggestion = res.suggest_tc(
    validation_window=10.0,
    min_tc=50.0,
    relative_drift_tolerance=0.05,
    q_tolerance=0.10,
)
if suggestion.tc is None:
    print("No common plateau; inspect", f"{res.fout}.tc_suggestion.csv")
else:
    result = res.analysis(tc=suggestion.tc)
    print("Reviewed common cutoff:", suggestion.tc, suggestion.time_unit)
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
suggestion = res.suggest_tc(validation_window=10.0, min_tc=50.0)
if suggestion.tc is not None:
    res.analysis(tc=suggestion.tc)
```

Pickles written by version 0.4.0 after `run_Dfit()` already contain the fitted
lag-dependent arrays required by `suggest_tc()`. Loading them with the current
version does not require rerunning the GLS fit.

The complete fitted-pickle example is
[`Example/run_publication_tc_suggestion.py`](Example/run_publication_tc_suggestion.py):

```bash
python Example/run_publication_tc_suggestion.py D_analysis.pkl \
    --validation-window 10 --min-tc 50
```
