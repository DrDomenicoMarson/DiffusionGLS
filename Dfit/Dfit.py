# Code for the estimation of translational diffusion coefficients from simulation data
# version 1.0 (06/02/2020)
# Jakob Tomas Bullerjahn
# Soeren von Buelow (soeren.buelow@biophys.mpg.de)
# Gerhard Hummer

# Please read and cite the paper: 
# J. Bullerjahn, S. v. Buelow, G. Hummer: Optimal estimates of diffusion coefficients from molecular dynamics simulations, Journal of Chemical Physics 153, 024116 (2020).

import math
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import warnings
import concurrent.futures
import os
import pickle

import numpy as np
from tqdm import tqdm
from scipy.special import gammainc

from . import math_utils
from .trajectory_reader import get_reader, TrajectoryReader

TrajectoryInput = str | Path | Sequence[str | Path] | None

XI_CUBIC = 2.837297
BOLTZMANN_K = 1.380649e-23 # J/K
MIN_SAMPLES_AT_TMAX = 100


class SamplingAdequacyWarning(UserWarning):
    """Warn that an explicit segmentation is feasible but statistically weak."""


@dataclass(frozen=True)
class DiffusionStatistics:
    """Diffusion estimate and conditional trajectory-level uncertainty.

    Parameters
    ----------
    diffusion : float
        Diffusion coefficient in ``diffusion_unit``.
    predicted_sd : float
        RMS model-predicted standard deviation of member estimates.
    empirical_sd : float or None
        Sample standard deviation of member estimates, or ``None`` when fewer
        than two members are available.
    predicted_sem : float
        Model-predicted conditional standard error of the mean.
    empirical_sem : float or None
        Empirical conditional standard error of the mean, or ``None`` when
        unavailable.
    q_mean : float or None
        Mean Q-factor, or ``None`` for ``m=2``.
    q_sd : float or None
        Sample standard deviation of Q, or ``None`` when unavailable.
    """

    diffusion: float
    predicted_sd: float
    empirical_sd: float | None
    predicted_sem: float
    empirical_sem: float | None
    q_mean: float | None
    q_sd: float | None


@dataclass(frozen=True)
class ClusterStatistics:
    """Selected-cutoff statistics for one independent cluster.

    Parameters
    ----------
    cluster_id : str
        User-facing cluster identifier.
    n_trajectories : int
        Number of member trajectories assigned to the cluster.
    statistics : DiffusionStatistics
        Cluster-level estimate and conditional uncertainty.
    """

    cluster_id: str
    n_trajectories: int
    statistics: DiffusionStatistics


@dataclass(frozen=True)
class AcrossClusterStatistics:
    """Equal-weight aggregation across independently prepared clusters.

    Parameters
    ----------
    n_clusters : int
        Number of independent clusters.
    mean : float
        Equal-weight mean of cluster diffusion estimates.
    sample_sd : float or None
        Sample standard deviation across cluster estimates.
    propagated_predicted_sem : float
        Conditional predicted SEM propagated from cluster-level errors without
        changing the equal-weight point estimate.
    propagated_empirical_sem : float or None
        Propagated empirical conditional SEM, or ``None`` when any contributing
        cluster has fewer than two trajectories.
    """

    n_clusters: int
    mean: float
    sample_sd: float | None
    propagated_predicted_sem: float
    propagated_empirical_sem: float | None


@dataclass(frozen=True)
class AnalysisResult:
    """Selected-cutoff pooled and cluster-aware diffusion result.

    Parameters
    ----------
    tc : float
        Common selected cutoff in ``time_unit``.
    time_unit : str
        Time unit used for ``tc``.
    diffusion_unit : str
        Unit used by every diffusion statistic.
    pooled : DiffusionStatistics
        Trajectory-weighted pooled result conditional on the sampled clusters.
    clusters : tuple[ClusterStatistics, ...]
        Per-cluster results in first-appearance order.
    across_clusters : AcrossClusterStatistics
        Equal-weight aggregation and separated uncertainty components.
    """

    tc: float
    time_unit: str
    diffusion_unit: str
    pooled: DiffusionStatistics
    clusters: tuple[ClusterStatistics, ...]
    across_clusters: AcrossClusterStatistics


@dataclass(frozen=True)
class AutoTcSelection:
    """Resolved lag-time choice for ``analysis(tc="auto")``.

    Parameters
    ----------
    selected_idx : int
        Final zero-based lag-grid index used for the analysis output.
    selected_step : int
        Final lag step on the internal ``tmin..tmax`` grid.
    selected_tc_ps : float
        Final lag time in ps.
    selected_tc_disp : float
        Final lag time in the user's ``time_unit``.
    unbounded_idx : int
        Zero-based lag-grid index from the unconstrained auto selection.
    unbounded_step : int
        Unconstrained lag step on the internal ``tmin..tmax`` grid.
    unbounded_tc_disp : float
        Unconstrained lag time in the user's ``time_unit``.
    auto_min_tc_used : float or None
        First lag on the computed grid that satisfies the lower-bound search
        threshold, in the user's ``time_unit``.
    selected_score : float
        Equal-cluster mean absolute deviation of Q from 0.5 at the selected lag.
    selected_max_deviation : float
        Largest absolute cluster Q deviation from 0.5 at the selected lag.
    """

    selected_idx: int
    selected_step: int
    selected_tc_ps: float
    selected_tc_disp: float
    unbounded_idx: int
    unbounded_step: int
    unbounded_tc_disp: float
    auto_min_tc_used: float | None
    selected_score: float
    selected_max_deviation: float


@dataclass(frozen=True)
class ClusterTcCandidateDiagnostic:
    """Publication-cutoff diagnostics for one cluster at one candidate lag.

    Parameters
    ----------
    cluster_id : str
        User-facing cluster identifier.
    diffusion : float
        Cluster diffusion estimate at the candidate lag in ``diffusion_unit``.
    predicted_sem : float
        Predicted within-cluster conditional SEM at the candidate lag in
        ``diffusion_unit``.
    empirical_sem : float or None
        Empirical within-cluster conditional SEM at the candidate lag in
        ``diffusion_unit``, or ``None`` when unavailable.
    max_relative_drift : float
        Largest block-mean relative range of the cluster ``D(t)`` curve over
        the consecutive validation windows.
    max_q_deviation : float
        Largest absolute deviation of a block-mean cluster Q value from 0.5
        over the consecutive validation windows.
    diffusion_passes : bool
        Whether ``max_relative_drift`` satisfies the configured tolerance.
    q_passes : bool
        Whether ``max_q_deviation`` satisfies the configured tolerance.
    passes : bool
        Whether both the diffusion and Q criteria pass.
    """

    cluster_id: str
    diffusion: float
    predicted_sem: float
    empirical_sem: float | None
    max_relative_drift: float
    max_q_deviation: float
    diffusion_passes: bool
    q_passes: bool
    passes: bool


@dataclass(frozen=True)
class TcCandidateDiagnostic:
    """Common publication-cutoff diagnostics at one candidate lag.

    Parameters
    ----------
    tc : float
        Candidate lag time in ``time_unit``.
    validation_end : float
        End of the complete persistence horizon in ``time_unit``.
    passes : bool
        Whether every cluster passes both selection criteria.
    worst_normalized_violation : float
        Worst cluster criterion divided by its corresponding tolerance. Values
        no greater than one pass; this score is descriptive, not inferential.
    clusters : tuple[ClusterTcCandidateDiagnostic, ...]
        Per-cluster diagnostics in cluster order.
    """

    tc: float
    validation_end: float
    passes: bool
    worst_normalized_violation: float
    clusters: tuple[ClusterTcCandidateDiagnostic, ...]


@dataclass(frozen=True)
class ClusterTcOnset:
    """Earliest sustained candidate accepted for one cluster.

    Parameters
    ----------
    cluster_id : str
        User-facing cluster identifier.
    tc : float or None
        Earliest candidate lag passing the configured persistence criteria, or
        ``None`` when the cluster has no acceptable candidate.
    """

    cluster_id: str
    tc: float | None


@dataclass(frozen=True)
class TcSuggestion:
    """Auditable common-cutoff suggestion for publication-oriented analysis.

    Parameters
    ----------
    tc : float or None
        Earliest candidate at or after the latest individual cluster onset for
        which every cluster passes simultaneously. ``None`` means that no
        common plateau was found.
    status : {'suggested', 'no_common_plateau'}
        Machine-readable selection status.
    time_unit : str
        Unit used for cutoff and window values.
    diffusion_unit : str
        Unit used for diffusion and conditional SEM values.
    min_tc : float
        Applied lower search bound on the lag grid.
    validation_window : float
        Width of each validation window.
    persistence_windows : int
        Number of consecutive validation windows required.
    blocks_per_window : int
        Number of contiguous blocks used to summarize each window.
    candidate_step : float
        Spacing of evaluated cutoff candidates on the lag grid.
    relative_drift_tolerance : float
        Maximum accepted block-mean relative range of ``D(t)``.
    q_tolerance : float
        Maximum accepted block-mean absolute Q deviation from 0.5.
    cluster_onsets : tuple[ClusterTcOnset, ...]
        Earliest passing candidate for each cluster.
    selected_candidate : TcCandidateDiagnostic or None
        Diagnostics for the suggested common cutoff, or ``None`` when no
        common cutoff passes.
    closest_candidate : TcCandidateDiagnostic
        Candidate with the smallest worst normalized violation. This is
        diagnostic only and is never promoted to ``tc`` when it fails.
    candidates : tuple[TcCandidateDiagnostic, ...]
        Complete candidate diagnostics used by the selection.
    """

    tc: float | None
    status: str
    time_unit: str
    diffusion_unit: str
    min_tc: float
    validation_window: float
    persistence_windows: int
    blocks_per_window: int
    candidate_step: float
    relative_drift_tolerance: float
    q_tolerance: float
    cluster_onsets: tuple[ClusterTcOnset, ...]
    selected_candidate: TcCandidateDiagnostic | None
    closest_candidate: TcCandidateDiagnostic
    candidates: tuple[TcCandidateDiagnostic, ...]


def calc_q(n,m,a2_3D,s2_3D,msds_3D,a2full_3D,s2full_3D,ndim):
    """Evaluate the segment quality factor ``Q`` from the GLS fit.

    Parameters
    ----------
    n : int
        Number of lag steps in the strided segment minus one.
    m : int
        Number of MSD values used in the GLS fit window.
    a2_3D : float
        Segment-level 3D offset estimate (sum across dimensions), in nm^2.
    s2_3D : float
        Segment-level 3D slope estimate (sum across dimensions), in nm^2.
    msds_3D : ndarray of shape (m,)
        Segment-level 3D MSD values used by the fit, in nm^2.
    a2full_3D : float
        Reference 3D offset used to construct the covariance matrix.
    s2full_3D : float
        Reference 3D slope used to construct the covariance matrix.
    ndim : int
        Number of spatial dimensions represented in the trajectory.

    Returns
    -------
    float
        Goodness-of-fit quality factor in the interval [0, 1].

    Notes
    -----
    This follows Eq. 22 in Bullerjahn et al. (JCP 153, 024116, 2020).
    """

    if m <= 2:
        return np.nan

    c = math_utils.setupc(m,n) # setups of cov matrix
    cov = math_utils.calc_cov(n,m,c,a2full_3D,s2full_3D)
    cinv = math_utils.inv_mat(cov)

    chi2 = math_utils.calc_chi2(m,a2_3D,s2_3D,msds_3D,cinv,ndim)

    if chi2 <= 0:
        q = 1.0
    else:
        q = 1-gammainc( (m-2)/2.,chi2/2.) # goodness-of-fit Q
    return q

def analyze_chunk_task(
    chunk_data,
    step,
    m,
    dt,
    multi,
    ndim,
    d2max,
    nitmax,
    covariance_helpers,
    a2full_3D=0.0,
    s2full_3D=0.0,
):
    """Process one worker chunk for a given lag step.

    Parameters
    ----------
    chunk_data : list of ndarray
        Segment trajectories assigned to this worker. Each array has shape
        ``(n_frames_segment, ndim)`` in nm.
    step : int
        Lag stride in frame steps.
    m : int
        Number of MSD values used per lag step in the GLS fit.
    dt : float
        Timestep in ps used for user-facing diagnostics.
    multi : bool
        ``True`` for multi-trajectory mode where each input trajectory is one
        segment/molecule, ``False`` for single-trajectory segmentation mode.
    ndim : int
        Number of spatial dimensions.
    d2max : float
        Convergence threshold for iterative GLS updates.
    nitmax : int
        Maximum number of GLS iterations.
    covariance_helpers : dict[int, tuple[ndarray, ndarray]]
        Read-only cache mapping the actual strided segment length in steps to
        the ``m=2`` and ``m`` covariance helper matrices.
    a2full_3D : float, optional
        Reference 3D offset used for ``Q`` evaluation in single-trajectory
        mode.
    s2full_3D : float, optional
        Reference 3D slope used for ``Q`` evaluation in single-trajectory
        mode.

    Returns
    -------
    tuple[list[tuple[ndarray, ndarray, float, bool]] | None, Exception | None]
        ``(results, error)`` where ``results`` contains one entry per segment:
        ``(a2_per_dim, s2_per_dim, q_value, converged)``.  When a segment is
        too short, ``results`` is ``None`` and ``error`` carries the
        ``ValueError`` to propagate in the caller thread.
    """
    results = []
    
    for s_idx, seg_z in enumerate(chunk_data):
        # Stride it
        z_analyzed = seg_z[::step, :]

        if len(z_analyzed) <= m:
            # Calculate suggestions
            max_m_possible = len(z_analyzed) - 1
            segment_steps = len(seg_z) - 1
            max_lag_steps = max(1, segment_steps // m)
            max_tmax_possible = max_lag_steps * dt
            
            # We can't raise here easily without crashing the worker.
            # Return error info.
            return None, ValueError(f"Segment too short (length N={len(z_analyzed)}) to calculate m={m} MSD points at lag step={step} (t={step*dt} ps).\n"
                                    f"Suggestions:\n"
                                    f"  - Reduce m to at most {max_m_possible}\n"
                                    f"  - Reduce tmax below {max_lag_steps} steps (~{max_tmax_possible:.1f} ps)\n"
                                    f"  - Use a longer trajectory")

        msds = np.zeros((ndim, m))
        res_a2 = np.zeros(ndim)
        res_s2 = np.zeros(ndim)
        res_converged = True
        
        n_per_seg_step = len(z_analyzed) - 1

        # Analyze per dimension
        for d in range(ndim):
            z_dim = np.ascontiguousarray(z_analyzed[:, d])
            msds[d] = math_utils.compute_MSD_1D_first_m(z_dim, m)
            
            use_c2, use_cm = covariance_helpers[n_per_seg_step]
            
            res_a2[d], res_s2[d], conv = math_utils.calc_gls(n_per_seg_step, m, msds[d], d2max, nitmax, c2=use_c2, cm=use_cm)
            if not conv:
                res_converged = False
        
        msds_3D = np.sum(msds, axis=0)
        a2_3D_val = np.sum(res_a2)
        s2_3D_val = np.sum(res_s2)
        
        if multi:
            q_val = calc_q(n_per_seg_step, m, a2_3D_val, s2_3D_val, msds_3D, a2_3D_val, s2_3D_val, ndim)
        else:
            q_val = calc_q(n_per_seg_step, m, a2_3D_val, s2_3D_val, msds_3D, a2full_3D, s2full_3D, ndim)
            
        results.append((res_a2, res_s2, q_val, res_converged))
        
    return results, None


def analyze_full_traj_task(full_z, step, m, dt, ndim, d2max, nitmax):
    """Run the full-trajectory GLS fit for one lag step (single-trajectory mode).

    Parameters
    ----------
    full_z : ndarray of shape (n_frames, ndim)
        Complete trajectory array in nm.
    step : int
        Lag stride in frame steps.
    m : int
        Number of MSD values used in the GLS fit window.
    dt : float
        Timestep in ps (used only for error messages).
    ndim : int
        Number of spatial dimensions.
    d2max : float
        Convergence threshold for iterative GLS updates.
    nitmax : int
        Maximum number of GLS iterations.

    Returns
    -------
    tuple[ndarray, ndarray, bool] | tuple[None, None, Exception]
        ``(a2_per_dim, s2_per_dim, converged_all)`` on success, or
        ``(None, None, exc)`` if the trajectory is too short.
    """
    z_strided_check = full_z[::step, 0]  # check length via first dim
    if len(z_strided_check) <= m:
        exc = ValueError(
            f"Trajectory too short (length N={len(z_strided_check)}) to calculate "
            f"m={m} MSD points at lag time step={step} (t={step * dt} ps). "
            f"Please reduce tmax or m, or use a longer trajectory."
        )
        return None, None, exc

    res_a2 = np.zeros(ndim)
    res_s2 = np.zeros(ndim)
    converged_all = True
    for d in range(ndim):
        z_dim = np.ascontiguousarray(full_z[:, d][::step])
        n = len(z_dim) - 1
        msd = math_utils.compute_MSD_1D_first_m(z_dim, m)
        res_a2[d], res_s2[d], converged = math_utils.calc_gls(n, m, msd, d2max, nitmax)
        if not converged:
            converged_all = False
    return res_a2, res_s2, converged_all

class Dcov():
    """Estimate diffusion coefficients from trajectories using GLS on MSD data.

    The class supports two analysis modes:

    1. Single-trajectory mode: one long trajectory is split into segments.
    2. Multi-trajectory mode: each input trajectory/residue is a segment.

    A typical workflow is:

    1. Initialize :class:`Dcov` with trajectory input and fit configuration.
    2. Call :meth:`run_Dfit` to perform GLS estimation across lag steps.
    3. Optionally call :meth:`suggest_tc` for auditable common-cutoff
       diagnostics across clusters.
    4. Call :meth:`analysis` with a reviewed numeric cutoff to compute summary
       statistics and write outputs.
    5. Optionally call :meth:`finite_size_correction` (3D, cubic box only).

    Notes
    -----
    Internal diffusion units are nm^2/ps. Reported units are controlled through
    ``diffusion_unit``.
    """
    def __init__(self, fz: TrajectoryInput = None, universe=None, universes=None, selection=None,
                 cluster_ids=None,
                 m: int = 20, tmin: float | None = None, tmax: float = 100.0, dt: float | None = None,
                 d2max: float = 1e-10, nitmax: int = 100,
                 nseg: int | None = None, imgfmt: str = 'pdf', fout: str = 'D_analysis',
                 n_jobs: int = -1, normalize_lengths: bool = False, time_unit: str = 'ps',
                 diffusion_unit: str = 'cm2/s', progress: bool = True):
        """Initialize the diffusion estimator and validate analysis settings.

        Parameters
        ----------
        fz : str or Path or sequence[str | Path] or None, optional
            One trajectory file or multiple trajectory files in text format.
            Each row is a frame and columns are coordinate dimensions in nm.
            Provide exactly one trajectory source among ``fz``, ``universe``,
            and ``universes``.
        universe : MDAnalysis.Universe or MDAnalysis.AtomGroup, optional
            MDAnalysis object used to extract per-residue trajectories.
            Provide exactly one trajectory source among ``fz``, ``universe``,
            and ``universes``.
        universes : sequence[MDAnalysis.Universe | MDAnalysis.AtomGroup], optional
            Multiple MDAnalysis objects to pool in one analysis. Inputs must
            have matching frame count and timestep.
        selection : str, optional
            MDAnalysis selection string used only when ``universe`` is given as
            a Universe. For ``universes``, the same selection is applied to
            each Universe entry.
        cluster_ids : sequence[object] or None, optional
            Cluster identifiers aligned with text files, a single MDAnalysis
            input, or entries in ``universes``. Multiple Universes default to
            separate clusters; multiple text files default to one cluster.
        m : int, default=20
            Number of MSD values used in each lag-time GLS fit window.
        tmin : float or None, optional
            Minimum lag time in ``time_unit``. If ``None``, defaults to one
            frame step.
        tmax : float, default=100.0
            Maximum lag time in ``time_unit``.
        dt : float or None, optional
            Frame timestep in ``time_unit``. Metadata-free text input requires
            this value. MDAnalysis input adopts trajectory metadata when it is
            omitted.
        d2max : float, default=1e-10
            Convergence criterion for iterative GLS updates.
        nitmax : int, default=100
            Maximum number of GLS iterations.
        nseg : int or None, optional
            Number of segments for single-trajectory mode. If ``None``, uses an
            automatic heuristic.
        imgfmt : {'pdf', 'png'}, default='pdf'
            Plot output format.
        fout : str, default='D_analysis'
            Output file prefix.
        n_jobs : int, default=-1
            Number of parallel workers. ``-1`` uses executor default, ``0``
            forces serial execution.
        normalize_lengths : bool, default=False
            When multiple text trajectories have different lengths, truncate to
            shortest length if ``True``.
        time_unit : {'ps', 'ns'}, default='ps'
            Time unit for user input/output values.
        diffusion_unit : {'cm2/s', 'nm2/ps'}, default='cm2/s'
            Unit used for reported diffusion coefficients.
        progress : bool, default=True
            If ``True``, show a progress bar over lag steps.

        Raises
        ------
        ValueError
            If input trajectories are invalid, too short, or parameter
            combinations are infeasible.
        TypeError
            If an unsupported image format is requested.
        """

        # Initialize Reader
        self.reader: TrajectoryReader = get_reader(
            fz=fz,
            universe=universe,
            universes=universes,
            selection=selection,
            normalize_lengths=normalize_lengths,
            cluster_ids=cluster_ids,
        )
        
        # Use reader properties
        self.ndim = self.reader.ndim
        self.n = self.reader.n_frames - 1 # N is steps, n_frames is points

        if self.reader.n_frames < 2:
            raise ValueError(
                f"Trajectory too short: {self.reader.n_frames} frame(s) (need >= 2). "
                f"Provide a trajectory with at least 2 data points."
            )

        if hasattr(self.reader, 'lengths'):
            unique_lengths = set(self.reader.lengths)
            if len(unique_lengths) > 1 and not getattr(self.reader, 'normalized', False):
                raise ValueError(f"All input trajectories must have the same length. Found lengths: {sorted(unique_lengths)}. "
                                 f"Use normalize_lengths=True to truncate to the shortest trajectory.")

        # Time unit handling: accept inputs in ps or ns, convert to ps internally
        unit_map = {'ps': 1.0, 'ns': 1e3}
        if time_unit not in unit_map:
            raise ValueError(f"Unsupported time_unit '{time_unit}'. Use 'ps' or 'ns'.")
        self.time_unit = time_unit
        self.time_scale = unit_map[time_unit] # multiplier to convert chosen unit to ps

        # Diffusion unit handling: internal nm^2/ps; allow reporting in cm^2/s
        diff_map = {'nm2/ps': 1.0, 'cm2/s': 0.01}
        if diffusion_unit not in diff_map:
            raise ValueError(f"Unsupported diffusion_unit '{diffusion_unit}'. Use 'nm2/ps' or 'cm2/s'.")
        self.diffusion_unit = diffusion_unit
        self.diff_scale = diff_map[diffusion_unit]  # scale from nm^2/ps to desired unit
        self.progress = progress

        if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 2:
            raise ValueError("m must be an integer greater than or equal to 2.")

        if dt is None:
            if self.reader.dt is None:
                raise ValueError(
                    "dt is required for metadata-free text trajectories; "
                    "provide dt in the selected time_unit."
                )
            self.dt = float(self.reader.dt)  # reader metadata are already in ps
        else:
            if not isinstance(dt, (int, float, np.integer, np.floating)):
                raise TypeError("dt must be a positive finite number.")
            self.dt = dt * self.time_scale  # convert user-provided dt to ps
            if not math.isfinite(self.dt) or self.dt <= 0:
                raise ValueError("dt must be a positive finite number.")
            if self.reader.dt is not None and not math.isclose(self.dt, self.reader.dt, rel_tol=1e-5):
                warnings.warn(
                    f"User provided dt ({dt} {self.time_unit} = {self.dt} ps) differs from "
                    f"reader's dt ({self.reader.dt} ps). Using provided dt.",
                    UserWarning,
                )

        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be a positive finite number.")

        self.m = int(m)

        for name, value in (("tmin", tmin), ("tmax", tmax)):
            if value is not None:
                if not isinstance(value, (int, float, np.integer, np.floating)):
                    raise TypeError(f"{name} must be a positive finite number or None.")
                if not math.isfinite(float(value)) or float(value) <= 0:
                    raise ValueError(f"{name} must be a positive finite number.")
        
        # Convert tmin/tmax from ps to steps
        if tmin is None:
            self.tmin = 1
        else:
            self.tmin = int(round((tmin * self.time_scale) / self.dt))
            if self.tmin < 1:
                self.tmin = 1
                
        if tmax is None:
            self.tmax = int(round((100.0 * self.time_scale) / self.dt)) # Default 100 in chosen unit
        else:
            self.tmax = int(round((tmax * self.time_scale) / self.dt))

        if self.tmax < 1:
            raise ValueError("tmax resolves to fewer than one frame step; increase tmax or reduce dt.")
        if self.tmin > self.tmax:
            raise ValueError(
                f"tmin resolves to step {self.tmin}, which exceeds tmax step {self.tmax}."
            )
            
        self.d2max = d2max
        self.nitmax = nitmax

        if imgfmt not in ['pdf','png']:
            raise TypeError("Error! Choose 'pdf' or 'png' as output format.")
        self.imgfmt = imgfmt
        self.fout = fout
        self.n_jobs = n_jobs

        print(f'Num trajectories/molecules = {self.reader.n_trajs}')
        print(f'Num steps (num frames -1) = {self.n}')
        print(f'Num dimensions = {self.ndim}')
        
        # Segment Logic
        # Case 1: Multiple molecules (from list or MDAnalysis residues)
        # Case 2: Single long trajectory to be segmented
        
        self.n_molecules = self.reader.n_trajs
        
        if self.n_molecules > 1:
            print(f'Analyzing trajectories of {self.n_molecules} molecules.')
            self.multi = True # Keep flag for internal logic if needed, or refactor it away
            self.nseg = self.n_molecules
            self.nperseg = self.n
            self.segment_lengths = np.full(self.nseg, self.n + 1, dtype=int)
            self.segment_cluster_ids = tuple(self.reader.trajectory_cluster_ids)
        else:
            print('Analyzing single trajectory.')
            self.multi = False
            total_points = self.n + 1
            auto_nseg = int(total_points / (100. * self.tmax))
            
            if nseg is None:
                if auto_nseg < 1:
                    warnings.warn(
                        "Timeseries too short for automatic nseg heuristic; using nseg=1. "
                        "Reduce tmax or set nseg explicitly to control segmentation.",
                        UserWarning,
                    )
                    self.nseg = 1
                else:
                    self.nseg = auto_nseg
            else:
                if not isinstance(nseg, (int, np.integer)) or isinstance(nseg, bool):
                    raise TypeError("nseg must be an integer or None.")
                if nseg < 1:
                    raise ValueError('nseg must be at least 1')
                self.nseg = int(nseg)

            base_length, remainder = divmod(total_points, self.nseg)
            self.segment_lengths = np.array(
                [base_length + (1 if i < remainder else 0) for i in range(self.nseg)],
                dtype=int,
            )
            self.nperseg = int(np.min(self.segment_lengths)) - 1
            if self.nperseg < 1:
                raise ValueError(f'nseg={self.nseg} yields segment length < 2 points; reduce nseg or use a longer trajectory')
            source_cluster_id = self.reader.trajectory_cluster_ids[0]
            self.segment_cluster_ids = (source_cluster_id,) * self.nseg

        strided_points_at_tmax = np.array(
            [((length - 1) // self.tmax) + 1 for length in self.segment_lengths],
            dtype=int,
        )
        min_strided_points = int(np.min(strided_points_at_tmax))
        if min_strided_points <= self.m:
            raise ValueError(
                "Segment too short at the requested tmax: the shortest segment has "
                f"{min_strided_points} strided coordinate samples, but m={self.m} "
                f"requires at least {self.m + 1}. Reduce nseg, tmax, or m."
            )

        if not self.multi and nseg is not None and min_strided_points < MIN_SAMPLES_AT_TMAX:
            recommended_nseg = max(1, auto_nseg)
            warnings.warn(
                f"Explicit nseg={self.nseg} is feasible but leaves only "
                f"{min_strided_points} strided coordinate samples in the shortest "
                f"segment at tmax; the conservative guideline is {MIN_SAMPLES_AT_TMAX}. "
                f"The automatic heuristic would use nseg={recommended_nseg}. "
                "The explicit value is being honored.",
                SamplingAdequacyWarning,
            )

        self.cluster_ids = tuple(dict.fromkeys(self.segment_cluster_ids))
        self.cluster_indices = tuple(
            np.flatnonzero(np.asarray(self.segment_cluster_ids, dtype=object) == cluster_id)
            for cluster_id in self.cluster_ids
        )

        # Arrays
        self.a2full = np.zeros((self.tmax-self.tmin+1, self.ndim))
        self.s2full = np.zeros((self.tmax-self.tmin+1, self.ndim))

        self.a2 = np.zeros((self.tmax-self.tmin+1, self.nseg, self.ndim))
        self.s2 = np.zeros((self.tmax-self.tmin+1, self.nseg, self.ndim))

        self.s2var = np.zeros((self.tmax-self.tmin+1))
        self.s2var_members = np.zeros((self.tmax-self.tmin+1, self.nseg))
        self.q = np.zeros((self.tmax-self.tmin+1, self.nseg))

        # Lifecycle flags
        self._fitted = False
        self._analyzed = False

        # Results (populated by run_Dfit / analysis)
        self.Dseg = None
        self.Dstd = None
        self.Dsem_pred = None
        self.Dsem_emp = None
        self.Dperdim = None
        self.D = None
        self.Dempstd = None
        self.q_m = None
        self.q_std = None
        self.tc_selected = None  # selected tc in chosen time unit
        self.tc_selected_idx = None  # index into lag arrays
        self.tc_auto_unbounded = None  # unconstrained auto tc in chosen time unit
        self.tc_auto_unbounded_idx = None  # index into lag arrays for unconstrained auto tc
        self.auto_min_tc_used = None  # applied lower bound on the lag grid in chosen time unit
        self.analysis_result = None
        self.tc_suggestion = None
        self.cluster_D = None
        self.cluster_Dstd = None
        self.cluster_Dempstd = None
        self.cluster_Dsem_pred = None
        self.cluster_Dsem_emp = None
        self.cluster_Dperdim = None
        self.cluster_q_m = None
        self.cluster_q_std = None
        self.D_cluster_mean = None
        self.D_cluster_sd = None
        self.D_cluster_sem_pred = None
        self.D_cluster_sem_emp = None
        self.q_cluster_score = None
        self.q_cluster_max_deviation = None

    @staticmethod
    def load(filename: str) -> 'Dcov':
        """Load a serialized :class:`Dcov` instance from a pickle file.

        Parameters
        ----------
        filename : str
            Path to a ``.pkl`` file previously produced by
            :meth:`run_Dfit(save_model=True)`.

        Returns
        -------
        Dcov
            Deserialized estimator object.
        """
        with open(filename, 'rb') as f:
            obj = pickle.load(f)

        # A loaded model is assumed to have been fitted. This also ensures
        # backward compatibility with older pickle files that lack this flag.
        obj._fitted = True
        obj.tc_auto_unbounded = getattr(obj, 'tc_auto_unbounded', None)
        obj.tc_auto_unbounded_idx = getattr(obj, 'tc_auto_unbounded_idx', None)
        obj.auto_min_tc_used = getattr(obj, 'auto_min_tc_used', None)
        obj.analysis_result = getattr(obj, 'analysis_result', None)
        obj.tc_suggestion = getattr(obj, 'tc_suggestion', None)

        if not hasattr(obj, "segment_lengths"):
            obj.segment_lengths = np.full(obj.nseg, obj.nperseg + 1, dtype=int)
        if not hasattr(obj, "s2var_members"):
            obj.s2var_members = np.repeat(obj.s2var[:, np.newaxis], obj.nseg, axis=1)

        if not hasattr(obj, "segment_cluster_ids"):
            inferred_ids = None
            reader = getattr(obj, "reader", None)
            sources = getattr(reader, "sources", None)
            if sources and sum(getattr(source, "n_residues", 0) for source in sources) == obj.nseg:
                inferred_ids = tuple(
                    f"cluster_{source_index}"
                    for source_index, source in enumerate(sources)
                    for _ in range(source.n_residues)
                )
            if inferred_ids is None:
                inferred_ids = ("cluster_0",) * obj.nseg
            obj.segment_cluster_ids = inferred_ids
            warnings.warn(
                "Loaded a legacy model without cluster metadata. Cluster IDs were "
                "inferred where possible; otherwise all estimates were assigned to "
                "one cluster.",
                UserWarning,
            )

        obj.cluster_ids = tuple(dict.fromkeys(obj.segment_cluster_ids))
        obj.cluster_indices = tuple(
            np.flatnonzero(np.asarray(obj.segment_cluster_ids, dtype=object) == cluster_id)
            for cluster_id in obj.cluster_ids
        )
        for attribute in (
            "cluster_D",
            "cluster_Dstd",
            "cluster_Dempstd",
            "cluster_Dsem_pred",
            "cluster_Dsem_emp",
            "cluster_Dperdim",
            "cluster_q_m",
            "cluster_q_std",
            "D_cluster_mean",
            "D_cluster_sd",
            "D_cluster_sem_pred",
            "D_cluster_sem_emp",
            "q_cluster_score",
            "q_cluster_max_deviation",
        ):
            if not hasattr(obj, attribute):
                setattr(obj, attribute, None)
        return obj

    def _timestep_index(self, tc: float) -> int:
        """Convert an analysis lag time to the internal lag-index position.

        Parameters
        ----------
        tc : float
            Lag time in ``time_unit``.

        Returns
        -------
        int
            Zero-based index into arrays stored on the lag-time grid
            ``tmin..tmax``.

        Raises
        ------
        ValueError
            If ``tc`` is not a multiple of ``dt`` or lies outside the computed
            lag-time range.
        """
        if not math.isfinite(tc) or tc <= 0:
            raise ValueError("tc must be a positive finite lag time.")
        steps = (tc * self.time_scale) / self.dt
        steps_int = round(steps)
        if not math.isclose(steps, steps_int, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f'tc [{tc}] must be a multiple of dt [{self.dt}] within numerical tolerance')
        itc = int(steps_int) - self.tmin
        if itc < 0 or itc >= (self.tmax - self.tmin + 1):
            raise ValueError(f'tc [{tc}] is outside the computed timestep range')
        return itc

    def _ceil_timestep_index(self, tc: float) -> int:
        """Map a lower-bound lag time to the first available grid index.

        Parameters
        ----------
        tc : float
            Lower-bound lag time in ``time_unit``.

        Returns
        -------
        int
            Zero-based index of the first computed lag that is greater than or
            equal to ``tc``.

        Raises
        ------
        ValueError
            If the requested lower bound exceeds the computed lag grid.
        """
        if not math.isfinite(tc) or tc <= 0:
            raise ValueError("auto_min_tc must be a positive finite lag time.")
        steps = (tc * self.time_scale) / self.dt
        steps_int = math.ceil(steps - 1e-12)
        step = max(int(steps_int), self.tmin)
        itc = step - self.tmin
        if itc < 0 or itc >= (self.tmax - self.tmin + 1):
            raise ValueError(
                f"auto_min_tc [{tc}] exceeds the computed lag-time range up to "
                f"{self.tmax * self.dt / self.time_scale:.4g} {self.time_unit}"
            )
        return itc

    def _select_auto_tc(self, auto_min_tc: float | None = None) -> AutoTcSelection:
        """Select the lag time for ``analysis(tc="auto")``.

        Parameters
        ----------
        auto_min_tc : float or None, optional
            Optional lower bound in ``time_unit``. When provided, the auto
            search starts from the first lag on the computed grid that is
            greater than or equal to this value.

        Returns
        -------
        AutoTcSelection
            Resolved auto-selection metadata, including the unconstrained
            choice and the final bounded choice.
        """
        if self.m <= 2:
            raise ValueError("tc='auto' requires m >= 3 because Q is undefined for m=2.")
        if self.q_cluster_score is None or not np.any(np.isfinite(self.q_cluster_score)):
            raise ValueError("Automatic cutoff selection requires finite cluster Q statistics.")

        diff = self.q_cluster_score
        unbounded_idx = int(np.nanargmin(diff))
        selected_idx = unbounded_idx
        auto_min_tc_used = None

        if auto_min_tc is not None:
            min_idx = self._ceil_timestep_index(auto_min_tc)
            if not np.any(np.isfinite(diff[min_idx:])):
                raise ValueError("No finite cluster Q statistics are available above auto_min_tc.")
            selected_idx = min_idx + int(np.nanargmin(diff[min_idx:]))
            min_step = self.tmin + min_idx
            auto_min_tc_used = (min_step * self.dt) / self.time_scale

        selected_step = self.tmin + selected_idx
        selected_tc_ps = selected_step * self.dt
        selected_tc_disp = selected_tc_ps / self.time_scale

        unbounded_step = self.tmin + unbounded_idx
        unbounded_tc_disp = (unbounded_step * self.dt) / self.time_scale

        return AutoTcSelection(
            selected_idx=selected_idx,
            selected_step=selected_step,
            selected_tc_ps=selected_tc_ps,
            selected_tc_disp=selected_tc_disp,
            unbounded_idx=unbounded_idx,
            unbounded_step=unbounded_step,
            unbounded_tc_disp=unbounded_tc_disp,
            auto_min_tc_used=auto_min_tc_used,
            selected_score=float(self.q_cluster_score[selected_idx]),
            selected_max_deviation=float(self.q_cluster_max_deviation[selected_idx]),
        )

    def run_Dfit(self, save_model: bool = False):
        """Run GLS fitting over all configured lag steps and segments.

        Parameters
        ----------
        save_model : bool, default=False
            If ``True``, serialize the current estimator to ``{fout}.pkl``
            after fitting.

        Returns
        -------
        None
            This method updates instance attributes in place.

        Raises
        ------
        ValueError
            If the trajectory is too short for the requested ``m`` and lag
            range at runtime.

        Notes
        -----
        Side effects include:

        - Populating fit arrays (``a2``, ``s2``, ``q``, ``s2var``).
        - Updating convergence diagnostics (``non_converged_count``,
          ``percent_failed``).
        - Setting lifecycle flag ``_fitted=True``.
        - Optionally writing ``{fout}.pkl``.

        The implementation submits all independent ``(lag_step, segment_chunk)``
        combinations to the thread pool at once, so the pool can saturate all
        available workers across the full lag-time range rather than being
        limited to the number of segments per step.  In single-trajectory mode
        a two-phase approach is used: full-trajectory fits (one per lag step)
        are collected first because their results are needed as the Q-factor
        reference for the segment fits.
        """
        all_trajs = list(self.reader)  # load once; shape (n_trajs, n_frames, ndim)
        if self.multi:
            segment_data = all_trajs
        else:
            segment_data = list(np.array_split(all_trajs[0], self.nseg, axis=0))

        actual_lengths = np.asarray([len(segment) for segment in segment_data], dtype=int)
        if not np.array_equal(actual_lengths, self.segment_lengths):
            raise RuntimeError(
                "Internal segmentation lengths differ from the validated partition."
            )

        # Prevent BLAS/LAPACK from spawning internal threads that would
        # oversubscribe cores when combined with our thread pool.
        for env_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ.setdefault(env_var, '1')

        # Warm up numba-compiled kernels to avoid recompilation under the
        # global compilation lock during threaded execution.  We compile
        # for C-contiguous arrays (matching np.ascontiguousarray output).
        try:
            dummy = np.zeros(8)
            _ = math_utils.compute_MSD_1D_first_m(dummy, 3)
            _ = math_utils.calc_gls(5, 3, np.arange(3, dtype=np.float64), self.d2max, 2,
                                    c2=math_utils.setupc(2, 5), cm=math_utils.setupc(3, 5))
        except Exception:
            pass

        # Determine worker count (treat n_jobs=0 as serial)
        if self.n_jobs is None or self.n_jobs < 0:
            max_workers = None
        elif self.n_jobs == 0:
            max_workers = 1
        else:
            max_workers = self.n_jobs

        n_lag_steps = self.tmax - self.tmin + 1
        non_converged_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            # ------------------------------------------------------------------
            # Phase 1 (single-trajectory mode only):
            # Run the full-trajectory GLS fits for every lag step in parallel.
            # Results populate self.a2full / self.s2full and are later used as
            # the Q-factor reference in the segment fits.
            # ------------------------------------------------------------------
            if not self.multi:
                full_z = all_trajs[0]
                full_futures: dict[concurrent.futures.Future, int] = {
                    executor.submit(
                        analyze_full_traj_task,
                        full_z, step, self.m, self.dt, self.ndim, self.d2max, self.nitmax
                    ): t
                    for t, step in enumerate(range(self.tmin, self.tmax + 1))
                }

                full_iter = concurrent.futures.as_completed(full_futures)
                if self.progress:
                    full_iter = tqdm(full_iter, total=n_lag_steps, desc="full-traj fits")

                for fut in full_iter:
                    t = full_futures[fut]
                    res_a2, res_s2, conv_or_exc = fut.result()
                    if isinstance(conv_or_exc, Exception):
                        raise conv_or_exc
                    self.a2full[t] = res_a2
                    self.s2full[t] = res_s2
                    if not conv_or_exc:
                        non_converged_count += 1

            # ------------------------------------------------------------------
            # Phase 2: Segment fits — one future per (lag_step, chunk).
            # Each lag step's segments are split across workers so the
            # heaviest steps (step ≈ 1, largest per-segment work) are
            # shared.  The Numba-JIT'd compute in each chunk releases the
            # GIL, giving true thread-level parallelism.
            # ------------------------------------------------------------------
            n_workers_eff = max_workers if max_workers else (os.cpu_count() or 1)
            chunk_size = max(1, math.ceil(self.nseg / n_workers_eff))

            # future -> (t, step, s_start, s_end)
            seg_futures: dict[concurrent.futures.Future, tuple[int, int, int, int]] = {}

            for t, step in enumerate(range(self.tmin, self.tmax + 1)):
                n_values = sorted({int((length - 1) // step) for length in self.segment_lengths})
                covariance_helpers = {
                    n_value: (
                        math_utils.setupc(2, n_value),
                        math_utils.setupc(self.m, n_value),
                    )
                    for n_value in n_values
                }

                a2full_3D_val = float(np.sum(self.a2full[t])) if not self.multi else 0.0
                s2full_3D_val = float(np.sum(self.s2full[t])) if not self.multi else 0.0

                for s_start in range(0, self.nseg, chunk_size):
                    s_end = min(s_start + chunk_size, self.nseg)

                    chunk_data = segment_data[s_start:s_end]

                    fut = executor.submit(
                        analyze_chunk_task,
                        chunk_data, step, self.m, self.dt,
                        self.multi, self.ndim, self.d2max, self.nitmax,
                        covariance_helpers, a2full_3D_val, s2full_3D_val
                    )
                    seg_futures[fut] = (t, step, s_start, s_end)

            seg_iter = concurrent.futures.as_completed(seg_futures)
            if self.progress:
                seg_iter = tqdm(seg_iter, total=len(seg_futures), desc="segment fits")

            for fut in seg_iter:
                t, step, s_start, s_end = seg_futures[fut]
                chunk_results, error = fut.result()
                if error:
                    raise error
                for i, (res_a2, res_s2, q_val, res_converged) in enumerate(chunk_results):
                    s = s_start + i
                    self.a2[t, s] = res_a2
                    self.s2[t, s] = res_s2
                    self.q[t, s] = q_val
                    if not res_converged:
                        non_converged_count += 1

        # ------------------------------------------------------------------
        # Post-processing (serial): variance estimation and step normalisation.
        # These depend on the full a2[t] / s2[t] arrays being populated, so
        # they cannot be overlapped with the futures.
        # ------------------------------------------------------------------
        for t, step in enumerate(range(self.tmin, self.tmax + 1)):
            for s, segment_length in enumerate(self.segment_lengths):
                n_per_seg_step = int((segment_length - 1) // step)
                self.s2var_members[t, s] = math_utils.eval_vars(
                    n_per_seg_step,
                    self.m,
                    self.a2[t, s],
                    self.s2[t, s],
                    self.ndim,
                ) / (step ** 2)

            self.s2var[t] = float(np.mean(self.s2var_members[t]))

            self.a2[t] /= step
            self.s2[t] /= step
            if not self.multi:
                self.a2full[t] /= step
                self.s2full[t] /= step

        self.non_converged_count = non_converged_count
        self.total_cases = n_lag_steps * (self.nseg + (1 if not self.multi else 0))
        self.percent_failed = 0.0

        if non_converged_count > 0:
            self.percent_failed = (non_converged_count / self.total_cases) * 100
            print(
                f"WARNING: Optimizer did not converge in {non_converged_count} cases "
                f"({self.percent_failed:.1f}% of Total {self.total_cases}). "
                f"Falling back to M=2 for those cases."
            )

        self._fitted = True

        if save_model:
            with open(f'{self.fout}.pkl', 'wb') as f:
                pickle.dump(self, f)
                print(f"Model saved to {self.fout}.pkl")

    # Output and plotting
    def _prepare_lag_statistics(self) -> None:
        """Compute pooled and cluster-aware statistics over the full lag grid.

        Returns
        -------
        None
            Populates the lag-dependent pooled, per-cluster, and across-cluster
            arrays used by :meth:`analysis` and :meth:`suggest_tc`.

        Raises
        ------
        RuntimeError
            If called before :meth:`run_Dfit`.
        """
        if not self._fitted:
            raise RuntimeError("Call run_Dfit() before preparing lag statistics.")

        denominator = 2.0 * self.ndim * self.dt
        member_D = self.s2.sum(axis=2) / denominator
        member_var_D = np.maximum(self.s2var_members, 0.0) / (denominator ** 2)
        if np.any(self.s2var_members < 0):
            warnings.warn(
                "Negative predicted variance encountered; clamping to zero.",
                RuntimeWarning,
            )

        self.Dseg = np.mean(member_D, axis=1)
        self.Dstd = np.sqrt(np.mean(member_var_D, axis=1))
        self.Dsem_pred = np.sqrt(np.sum(member_var_D, axis=1)) / self.nseg
        if self.nseg > 1:
            self.Dempstd = np.std(member_D, axis=1, ddof=1)
            self.Dsem_emp = self.Dempstd / math.sqrt(self.nseg)
        else:
            self.Dempstd = np.full(member_D.shape[0], np.nan)
            self.Dsem_emp = np.full(member_D.shape[0], np.nan)

        if self.multi:
            self.D = self.Dseg
            self.Dperdim = np.mean(self.s2, axis=1) / (2.0 * self.dt)
        else:
            self.D = self.s2full.sum(axis=1) / denominator
            self.Dperdim = self.s2full / (2.0 * self.dt)

        if self.m > 2:
            self.q_m = np.mean(self.q, axis=1)
            if self.nseg > 1:
                self.q_std = np.std(self.q, axis=1, ddof=1)
            else:
                self.q_std = np.full(self.q.shape[0], np.nan)
        else:
            self.q_m = np.full(self.q.shape[0], np.nan)
            self.q_std = np.full(self.q.shape[0], np.nan)

        n_lags = member_D.shape[0]
        n_clusters = len(self.cluster_ids)
        self.cluster_D = np.zeros((n_lags, n_clusters))
        self.cluster_Dstd = np.zeros((n_lags, n_clusters))
        self.cluster_Dempstd = np.full((n_lags, n_clusters), np.nan)
        self.cluster_Dsem_pred = np.zeros((n_lags, n_clusters))
        self.cluster_Dsem_emp = np.full((n_lags, n_clusters), np.nan)
        self.cluster_Dperdim = np.zeros((n_lags, n_clusters, self.ndim))
        self.cluster_q_m = np.full((n_lags, n_clusters), np.nan)
        self.cluster_q_std = np.full((n_lags, n_clusters), np.nan)

        for cluster_index, member_indices in enumerate(self.cluster_indices):
            count = len(member_indices)
            cluster_member_D = member_D[:, member_indices]
            cluster_member_var = member_var_D[:, member_indices]
            self.cluster_D[:, cluster_index] = np.mean(cluster_member_D, axis=1)
            self.cluster_Dstd[:, cluster_index] = np.sqrt(
                np.mean(cluster_member_var, axis=1)
            )
            self.cluster_Dsem_pred[:, cluster_index] = (
                np.sqrt(np.sum(cluster_member_var, axis=1)) / count
            )
            self.cluster_Dperdim[:, cluster_index] = (
                np.mean(self.s2[:, member_indices, :], axis=1) / (2.0 * self.dt)
            )
            if count > 1:
                self.cluster_Dempstd[:, cluster_index] = np.std(
                    cluster_member_D, axis=1, ddof=1
                )
                self.cluster_Dsem_emp[:, cluster_index] = (
                    self.cluster_Dempstd[:, cluster_index] / math.sqrt(count)
                )
            if self.m > 2:
                self.cluster_q_m[:, cluster_index] = np.mean(
                    self.q[:, member_indices], axis=1
                )
                if count > 1:
                    self.cluster_q_std[:, cluster_index] = np.std(
                        self.q[:, member_indices], axis=1, ddof=1
                    )

        if not self.multi:
            self.cluster_D[:, 0] = self.D
            self.cluster_Dperdim[:, 0] = self.Dperdim

        self.D_cluster_mean = np.mean(self.cluster_D, axis=1)
        self.D_cluster_sd = (
            np.std(self.cluster_D, axis=1, ddof=1)
            if n_clusters > 1
            else np.full(n_lags, np.nan)
        )
        self.D_cluster_sem_pred = (
            np.sqrt(np.sum(self.cluster_Dsem_pred ** 2, axis=1)) / n_clusters
        )
        if np.all(np.isfinite(self.cluster_Dsem_emp), axis=1).any():
            self.D_cluster_sem_emp = np.where(
                np.all(np.isfinite(self.cluster_Dsem_emp), axis=1),
                np.sqrt(np.nansum(self.cluster_Dsem_emp ** 2, axis=1)) / n_clusters,
                np.nan,
            )
        else:
            self.D_cluster_sem_emp = np.full(n_lags, np.nan)

        if self.m > 2:
            cluster_deviation = np.abs(self.cluster_q_m - 0.5)
            self.q_cluster_score = np.mean(cluster_deviation, axis=1)
            self.q_cluster_max_deviation = np.max(cluster_deviation, axis=1)
        else:
            self.q_cluster_score = np.full(n_lags, np.nan)
            self.q_cluster_max_deviation = np.full(n_lags, np.nan)

    def _duration_to_steps(self, value: float, parameter_name: str) -> int:
        """Convert a physical duration to an exact positive lag-grid step count.

        Parameters
        ----------
        value : float
            Duration in ``time_unit``.
        parameter_name : str
            Public parameter name used in validation messages.

        Returns
        -------
        int
            Positive number of internal frame steps represented by ``value``.

        Raises
        ------
        TypeError
            If ``value`` is not numeric.
        ValueError
            If ``value`` is non-positive, non-finite, shorter than one frame,
            or not a multiple of ``dt``.
        """
        if (
            not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_))
        ):
            raise TypeError(f"{parameter_name} must be a positive finite number.")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{parameter_name} must be a positive finite number.")
        steps = (float(value) * self.time_scale) / self.dt
        steps_int = round(steps)
        if steps_int < 1:
            raise ValueError(
                f"{parameter_name} resolves to fewer than one frame step; "
                f"increase it to at least {self.dt / self.time_scale:.4g} "
                f"{self.time_unit}."
            )
        if not math.isclose(steps, steps_int, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{parameter_name} [{value}] must be a multiple of dt "
                f"[{self.dt / self.time_scale:.4g} {self.time_unit}] within "
                "numerical tolerance."
            )
        return int(steps_int)

    def _publication_candidate(
        self,
        candidate_idx: int,
        validation_steps: int,
        persistence_windows: int,
        blocks_per_window: int,
        relative_drift_tolerance: float,
        q_tolerance: float,
    ) -> TcCandidateDiagnostic:
        """Evaluate all clusters at one publication-cutoff candidate.

        Parameters
        ----------
        candidate_idx : int
            Candidate index on the stored lag grid.
        validation_steps : int
            Width of one validation window in internal lag steps.
        persistence_windows : int
            Number of consecutive windows to evaluate.
        blocks_per_window : int
            Number of contiguous blocks used to summarize each window.
        relative_drift_tolerance : float
            Maximum accepted relative block-mean range of ``D(t)``.
        q_tolerance : float
            Maximum accepted absolute block-mean Q deviation from 0.5.

        Returns
        -------
        TcCandidateDiagnostic
            Complete common and per-cluster diagnostics for the candidate.
        """
        cluster_diagnostics = []
        for cluster_index, cluster_id in enumerate(self.cluster_ids):
            relative_drifts = []
            q_deviations = []
            for window_index in range(persistence_windows):
                start = candidate_idx + window_index * validation_steps
                stop = start + validation_steps
                window_indices = np.arange(start, stop + 1, dtype=int)
                blocks = np.array_split(window_indices, blocks_per_window)
                d_block_means = np.array(
                    [np.mean(self.cluster_D[block, cluster_index]) for block in blocks]
                )
                q_block_means = np.array(
                    [np.mean(self.cluster_q_m[block, cluster_index]) for block in blocks]
                )

                d_range = float(np.ptp(d_block_means))
                d_level = abs(float(np.mean(d_block_means)))
                if d_level > np.finfo(float).tiny:
                    relative_drift = d_range / d_level
                else:
                    relative_drift = 0.0 if d_range == 0.0 else math.inf
                q_deviation = (
                    float(np.max(np.abs(q_block_means - 0.5)))
                    if np.all(np.isfinite(q_block_means))
                    else math.inf
                )
                relative_drifts.append(relative_drift)
                q_deviations.append(q_deviation)

            max_relative_drift = float(np.max(relative_drifts))
            max_q_deviation = float(np.max(q_deviations))
            diffusion_passes = (
                math.isfinite(max_relative_drift)
                and max_relative_drift <= relative_drift_tolerance
            )
            q_passes = math.isfinite(max_q_deviation) and max_q_deviation <= q_tolerance
            cluster_diagnostics.append(
                ClusterTcCandidateDiagnostic(
                    cluster_id=cluster_id,
                    diffusion=float(
                        self.cluster_D[candidate_idx, cluster_index] * self.diff_scale
                    ),
                    predicted_sem=float(
                        self.cluster_Dsem_pred[candidate_idx, cluster_index]
                        * self.diff_scale
                    ),
                    empirical_sem=self._scaled_optional(
                        self.cluster_Dsem_emp[candidate_idx, cluster_index]
                    ),
                    max_relative_drift=max_relative_drift,
                    max_q_deviation=max_q_deviation,
                    diffusion_passes=diffusion_passes,
                    q_passes=q_passes,
                    passes=diffusion_passes and q_passes,
                )
            )

        worst_normalized_violation = max(
            max(
                diagnostic.max_relative_drift / relative_drift_tolerance,
                diagnostic.max_q_deviation / q_tolerance,
            )
            for diagnostic in cluster_diagnostics
        )
        total_steps = validation_steps * persistence_windows
        return TcCandidateDiagnostic(
            tc=float((self.tmin + candidate_idx) * self.dt / self.time_scale),
            validation_end=float(
                (self.tmin + candidate_idx + total_steps) * self.dt / self.time_scale
            ),
            passes=all(diagnostic.passes for diagnostic in cluster_diagnostics),
            worst_normalized_violation=float(worst_normalized_violation),
            clusters=tuple(cluster_diagnostics),
        )

    def suggest_tc(
        self,
        validation_window: float,
        *,
        min_tc: float | None = None,
        relative_drift_tolerance: float = 0.05,
        q_tolerance: float = 0.10,
        persistence_windows: int = 2,
        blocks_per_window: int = 5,
        candidate_step: float | None = None,
        fout_prefix: str | None = None,
        make_plot: bool = True,
    ) -> TcSuggestion:
        """Suggest a reviewed common cutoff from sustained cluster diagnostics.

        Parameters
        ----------
        validation_window : float
            Physical width of each validation window in ``time_unit``. The
            value must be a multiple of ``dt`` and should represent a
            scientifically meaningful interval over which a plateau is
            expected to persist.
        min_tc : float or None, optional
            Lower search bound in ``time_unit``. ``None`` starts at ``tmin``;
            otherwise the bound is rounded up to the first stored lag.
        relative_drift_tolerance : float, default=0.05
            Maximum relative range among contiguous block means of ``D(t)`` in
            every validation window. The default permits 5% block-to-block
            drift and is a documented pragmatic criterion, not a universal
            statistical threshold.
        q_tolerance : float, default=0.10
            Maximum absolute deviation of every block-mean cluster Q value
            from 0.5. The default accepts the interval [0.4, 0.6].
        persistence_windows : int, default=2
            Number of consecutive validation windows that must pass for every
            cluster. Neighboring closed windows share only their boundary lag.
        blocks_per_window : int, default=5
            Number of contiguous blocks used to summarize each validation
            window. Block means reduce sensitivity to isolated lag-grid noise
            without treating adjacent lag points as independent replicates.
        candidate_step : float or None, optional
            Nominal spacing between evaluated candidates in ``time_unit``. The
            value must be a multiple of ``dt``. ``None`` uses one tenth of the
            validation window, rounded to at least one frame step. The final
            feasible candidate is also evaluated when it is off this spacing.
        fout_prefix : str or None, optional
            Output base for the diagnostic ``.dat``, ``.csv``, and plot files.
            ``None`` uses ``{fout}.tc_suggestion``.
        make_plot : bool, default=True
            Whether to write the two-panel cutoff-diagnostic plot.

        Returns
        -------
        TcSuggestion
            Dataclass containing the suggested common cutoff, individual
            cluster onsets, the closest failing candidate when applicable, and
            complete candidate diagnostics.

        Raises
        ------
        RuntimeError
            If called before :meth:`run_Dfit`.
        TypeError
            If integer or Boolean configuration values have invalid types.
        ValueError
            If ``m < 3``, thresholds or durations are invalid, or the lag range
            is too short for the requested persistence horizon.

        Notes
        -----
        This is a reproducible suggestion, not an automatic certification of a
        publication cutoff. Selection is based on practical equivalence bands,
        not p-values. Conditional cluster SEMs are reported for context but do
        not weight the point estimate or relax the plateau criterion. The
        returned numeric ``tc`` should be reviewed and then passed explicitly
        to :meth:`analysis`.
        """
        if not self._fitted:
            raise RuntimeError("Call run_Dfit() before suggest_tc().")
        if self.m < 3:
            raise ValueError("suggest_tc() requires m >= 3 because Q is undefined for m=2.")
        for name, value in (
            ("relative_drift_tolerance", relative_drift_tolerance),
            ("q_tolerance", q_tolerance),
        ):
            if (
                not isinstance(value, (int, float, np.integer, np.floating))
                or isinstance(value, (bool, np.bool_))
            ):
                raise TypeError(f"{name} must be a positive finite number.")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        if q_tolerance > 0.5:
            raise ValueError("q_tolerance must not exceed 0.5.")
        for name, value in (
            ("persistence_windows", persistence_windows),
            ("blocks_per_window", blocks_per_window),
        ):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be at least 1.")
        if blocks_per_window < 2:
            raise ValueError("blocks_per_window must be at least 2.")
        if not isinstance(make_plot, (bool, np.bool_)):
            raise TypeError("make_plot must be a Boolean.")

        validation_steps = self._duration_to_steps(validation_window, "validation_window")
        if validation_steps < blocks_per_window:
            raise ValueError(
                "validation_window is too short for blocks_per_window: "
                f"{validation_steps} lag steps cannot form {blocks_per_window} "
                "meaningful contiguous blocks."
            )
        if candidate_step is None:
            candidate_step_steps = max(1, int(round(validation_steps / 10.0)))
        else:
            candidate_step_steps = self._duration_to_steps(candidate_step, "candidate_step")
        candidate_step_used = candidate_step_steps * self.dt / self.time_scale

        if min_tc is None:
            min_idx = 0
        else:
            if (
                not isinstance(min_tc, (int, float, np.integer, np.floating))
                or isinstance(min_tc, (bool, np.bool_))
            ):
                raise TypeError("min_tc must be a positive finite number or None.")
            if not math.isfinite(float(min_tc)) or float(min_tc) <= 0:
                raise ValueError("min_tc must be a positive finite number or None.")
            min_steps = math.ceil((float(min_tc) * self.time_scale) / self.dt - 1e-12)
            min_idx = max(int(min_steps), self.tmin) - self.tmin
        min_tc_used = (self.tmin + min_idx) * self.dt / self.time_scale

        self._prepare_lag_statistics()
        total_validation_steps = validation_steps * persistence_windows
        max_start_idx = len(self.D) - 1 - total_validation_steps
        if min_idx < 0 or min_idx > max_start_idx:
            maximum_start = (
                (self.tmin + max_start_idx) * self.dt / self.time_scale
                if max_start_idx >= 0
                else None
            )
            suffix = (
                f"; the latest feasible start is {maximum_start:.4g} {self.time_unit}"
                if maximum_start is not None
                else ""
            )
            raise ValueError(
                "The computed lag range is too short for the requested min_tc, "
                "validation_window, and persistence_windows" + suffix + "."
            )

        candidate_indices = list(
            range(min_idx, max_start_idx + 1, candidate_step_steps)
        )
        if candidate_indices[-1] != max_start_idx:
            candidate_indices.append(max_start_idx)
        candidates = tuple(
            self._publication_candidate(
                candidate_idx,
                validation_steps,
                int(persistence_windows),
                int(blocks_per_window),
                float(relative_drift_tolerance),
                float(q_tolerance),
            )
            for candidate_idx in candidate_indices
        )

        cluster_onsets = []
        for cluster_index, cluster_id in enumerate(self.cluster_ids):
            onset = next(
                (
                    candidate.tc
                    for candidate in candidates
                    if candidate.clusters[cluster_index].passes
                ),
                None,
            )
            cluster_onsets.append(ClusterTcOnset(cluster_id=cluster_id, tc=onset))

        selected_candidate = None
        if all(onset.tc is not None for onset in cluster_onsets):
            latest_onset = max(onset.tc for onset in cluster_onsets if onset.tc is not None)
            selected_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.tc >= latest_onset and candidate.passes
                ),
                None,
            )
        closest_candidate = min(
            candidates,
            key=lambda candidate: (candidate.worst_normalized_violation, candidate.tc),
        )
        status = "suggested" if selected_candidate is not None else "no_common_plateau"
        self.tc_suggestion = TcSuggestion(
            tc=selected_candidate.tc if selected_candidate is not None else None,
            status=status,
            time_unit=self.time_unit,
            diffusion_unit=self.diffusion_unit,
            min_tc=float(min_tc_used),
            validation_window=float(validation_steps * self.dt / self.time_scale),
            persistence_windows=int(persistence_windows),
            blocks_per_window=int(blocks_per_window),
            candidate_step=float(candidate_step_used),
            relative_drift_tolerance=float(relative_drift_tolerance),
            q_tolerance=float(q_tolerance),
            cluster_onsets=tuple(cluster_onsets),
            selected_candidate=selected_candidate,
            closest_candidate=closest_candidate,
            candidates=candidates,
        )

        out_base = fout_prefix if fout_prefix is not None else f"{self.fout}.tc_suggestion"
        self._write_tc_suggestion_report(out_base, self.tc_suggestion)
        self._write_tc_suggestion_csv(out_base, self.tc_suggestion)
        if make_plot:
            self.plot_tc_suggestion(out_base, self.tc_suggestion)

        if selected_candidate is None:
            print(
                "No common publication cutoff passed all cluster criteria. "
                f"Closest diagnostic candidate: {closest_candidate.tc:.4g} "
                f"{self.time_unit} (worst normalized violation "
                f"{closest_candidate.worst_normalized_violation:.3g})."
            )
        else:
            print(
                f"Suggested common tc = {selected_candidate.tc:.4g} {self.time_unit}; "
                f"validated through {selected_candidate.validation_end:.4g} "
                f"{self.time_unit} in every cluster. Review diagnostics and rerun "
                "analysis() with this numeric cutoff."
            )
        return self.tc_suggestion

    def _write_tc_suggestion_report(
        self,
        out_base: str,
        suggestion: TcSuggestion,
    ) -> None:
        """Write the concise publication-cutoff suggestion report.

        Parameters
        ----------
        out_base : str
            Output path prefix.
        suggestion : TcSuggestion
            Suggestion and diagnostics to serialize.

        Returns
        -------
        None
            Writes ``{out_base}.dat``.
        """
        evaluated = suggestion.selected_candidate or suggestion.closest_candidate
        with open(f"{out_base}.dat", "w", encoding="utf-8") as handle:
            handle.write("PUBLICATION CUTOFF SUGGESTION\n")
            handle.write("================================\n")
            handle.write(
                "This is an auditable practical-equivalence suggestion, not an "
                "automatic certification of a publication cutoff.\n"
            )
            handle.write(f"Status: {suggestion.status}\n")
            if suggestion.tc is None:
                handle.write("Suggested common tc: NONE (no common plateau passed)\n")
            else:
                handle.write(
                    f"Suggested common tc: {suggestion.tc:.6g} {suggestion.time_unit}\n"
                )
            handle.write(
                f"Applied min_tc: {suggestion.min_tc:.6g} {suggestion.time_unit}\n"
                f"Validation window: {suggestion.validation_window:.6g} "
                f"{suggestion.time_unit}\n"
                f"Persistence windows: {suggestion.persistence_windows}\n"
                f"Blocks per window: {suggestion.blocks_per_window}\n"
                f"Candidate step: {suggestion.candidate_step:.6g} "
                f"{suggestion.time_unit}\n"
                f"Relative D drift tolerance: {suggestion.relative_drift_tolerance:.6g}\n"
                f"Q tolerance about 0.5: {suggestion.q_tolerance:.6g}\n"
                f"Evaluated candidates: {len(suggestion.candidates)}\n"
                f"Latest feasible candidate: {suggestion.candidates[-1].tc:.6g} "
                f"{suggestion.time_unit}\n"
            )
            handle.write("\nINDIVIDUAL CLUSTER ONSETS\n")
            for onset in suggestion.cluster_onsets:
                onset_text = (
                    f"{onset.tc:.6g} {suggestion.time_unit}"
                    if onset.tc is not None
                    else "NONE"
                )
                handle.write(f"{onset.cluster_id}: {onset_text}\n")
            if suggestion.tc is None:
                handle.write(
                    "\nCLOSEST FAILING CANDIDATE (DIAGNOSTIC ONLY; NOT SELECTED)\n"
                )
            else:
                handle.write("\nSELECTED COMMON CANDIDATE\n")
            handle.write(
                f"tc: {evaluated.tc:.6g} {suggestion.time_unit}\n"
                f"Validation horizon end: {evaluated.validation_end:.6g} "
                f"{suggestion.time_unit}\n"
                f"Worst normalized violation: "
                f"{evaluated.worst_normalized_violation:.6g}\n"
            )
            for diagnostic in evaluated.clusters:
                handle.write(
                    f"{diagnostic.cluster_id}: D={diagnostic.diffusion:.6e} "
                    f"{suggestion.diffusion_unit}; predicted conditional SEM="
                    f"{diagnostic.predicted_sem:.6e} {suggestion.diffusion_unit}; "
                    f"empirical conditional SEM="
                    f"{self._format_optional(diagnostic.empirical_sem, '.6e')} "
                    f"{suggestion.diffusion_unit}; "
                    f"max relative D drift={diagnostic.max_relative_drift:.6g}; "
                    f"max |Q-0.5|={diagnostic.max_q_deviation:.6g}; "
                    f"passes={diagnostic.passes}\n"
                )
            handle.write(
                "\nConditional SEMs provide within-box context only. They are not "
                "used as cluster weights and do not relax the plateau criteria.\n"
            )

    def _write_tc_suggestion_csv(
        self,
        out_base: str,
        suggestion: TcSuggestion,
    ) -> None:
        """Write complete publication-cutoff candidate diagnostics.

        Parameters
        ----------
        out_base : str
            Output path prefix.
        suggestion : TcSuggestion
            Suggestion and candidate diagnostics to serialize.

        Returns
        -------
        None
            Writes ``{out_base}.csv`` with one row per candidate and cluster.
        """
        fieldnames = [
            "status",
            "selected",
            "candidate_tc",
            "validation_end",
            "time_unit",
            "candidate_passes_all_clusters",
            "worst_normalized_violation",
            "cluster_id",
            "cluster_onset_tc",
            "diffusion",
            "diffusion_unit",
            "conditional_sem_predicted",
            "conditional_sem_empirical",
            "max_relative_drift",
            "relative_drift_tolerance",
            "diffusion_passes",
            "max_q_deviation",
            "q_tolerance",
            "q_passes",
            "cluster_passes",
            "min_tc",
            "validation_window",
            "persistence_windows",
            "blocks_per_window",
            "candidate_step",
        ]
        onset_by_cluster = {onset.cluster_id: onset.tc for onset in suggestion.cluster_onsets}
        with open(f"{out_base}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in suggestion.candidates:
                for diagnostic in candidate.clusters:
                    writer.writerow(
                        {
                            "status": suggestion.status,
                            "selected": suggestion.tc is not None
                            and math.isclose(candidate.tc, suggestion.tc),
                            "candidate_tc": candidate.tc,
                            "validation_end": candidate.validation_end,
                            "time_unit": suggestion.time_unit,
                            "candidate_passes_all_clusters": candidate.passes,
                            "worst_normalized_violation": candidate.worst_normalized_violation,
                            "cluster_id": diagnostic.cluster_id,
                            "cluster_onset_tc": onset_by_cluster[diagnostic.cluster_id]
                            if onset_by_cluster[diagnostic.cluster_id] is not None
                            else np.nan,
                            "diffusion": diagnostic.diffusion,
                            "diffusion_unit": suggestion.diffusion_unit,
                            "conditional_sem_predicted": diagnostic.predicted_sem,
                            "conditional_sem_empirical": diagnostic.empirical_sem
                            if diagnostic.empirical_sem is not None
                            else np.nan,
                            "max_relative_drift": diagnostic.max_relative_drift,
                            "relative_drift_tolerance": suggestion.relative_drift_tolerance,
                            "diffusion_passes": diagnostic.diffusion_passes,
                            "max_q_deviation": diagnostic.max_q_deviation,
                            "q_tolerance": suggestion.q_tolerance,
                            "q_passes": diagnostic.q_passes,
                            "cluster_passes": diagnostic.passes,
                            "min_tc": suggestion.min_tc,
                            "validation_window": suggestion.validation_window,
                            "persistence_windows": suggestion.persistence_windows,
                            "blocks_per_window": suggestion.blocks_per_window,
                            "candidate_step": suggestion.candidate_step,
                        }
                    )

    def analysis(self, tc: float | str = 10, fout_prefix: str | None = None,
                 auto_min_tc: float | None = None) -> AnalysisResult:
        """Compute pooled and cluster-aware statistics and write outputs.

        Parameters
        ----------
        tc : float or {'auto'}, default=10
            Common lag time for every cluster in ``time_unit``. A numeric value
            is recommended for publication. ``'auto'`` provides a diagnostic
            cluster-balanced Q suggestion and requires ``m >= 3``.
        fout_prefix : str or None, optional
            Custom output base. By default, use ``{fout}.tc_{tc}``.
        auto_min_tc : float or None, optional
            Lower bound for automatic cutoff selection in ``time_unit``.

        Returns
        -------
        AnalysisResult
            Selected-cutoff pooled, per-cluster, and across-cluster results.

        Raises
        ------
        RuntimeError
            If called before :meth:`run_Dfit`.
        ValueError
            If the cutoff is invalid or automatic selection is requested with
            ``m=2``.
        """
        if not self._fitted:
            raise RuntimeError("Call run_Dfit() before analysis().")
        self._prepare_lag_statistics()

        self.tc_auto_unbounded = None
        self.tc_auto_unbounded_idx = None
        self.auto_min_tc_used = None
        auto_selection = None

        if tc == 'auto':
            auto_selection = self._select_auto_tc(auto_min_tc=auto_min_tc)
            itc = auto_selection.selected_idx
            step = auto_selection.selected_step
            tc_ps = auto_selection.selected_tc_ps
            tc_disp = auto_selection.selected_tc_disp
            self.tc_auto_unbounded_idx = auto_selection.unbounded_idx
            self.tc_auto_unbounded = auto_selection.unbounded_tc_disp
            self.auto_min_tc_used = auto_selection.auto_min_tc_used
            print(
                f"Diagnostic auto tc = {tc_disp:.4g} {self.time_unit}; "
                f"mean |Q_cluster-0.5| = {auto_selection.selected_score:.4f}, "
                f"max deviation = {auto_selection.selected_max_deviation:.4f}."
            )
        else:
            if auto_min_tc is not None:
                raise ValueError("auto_min_tc can only be used when tc='auto'")
            if not isinstance(tc, (int, float, np.integer, np.floating)):
                raise ValueError("tc must be a numeric lag time or 'auto'.")
            itc = self._timestep_index(float(tc))
            step = self.tmin + itc
            tc_ps = step * self.dt
            tc_disp = tc_ps / self.time_scale

        self.tc_selected_idx = itc
        self.tc_selected = tc_disp
        out_base = fout_prefix if fout_prefix is not None else f"{self.fout}.tc_{tc_disp:.4g}"

        pooled_statistics = self._statistics_at(
            self.D,
            self.Dstd,
            self.Dempstd,
            self.Dsem_pred,
            self.Dsem_emp,
            self.q_m,
            self.q_std,
            itc,
        )
        cluster_statistics = tuple(
            ClusterStatistics(
                cluster_id=cluster_id,
                n_trajectories=len(self.cluster_indices[index]),
                statistics=self._statistics_at(
                    self.cluster_D[:, index],
                    self.cluster_Dstd[:, index],
                    self.cluster_Dempstd[:, index],
                    self.cluster_Dsem_pred[:, index],
                    self.cluster_Dsem_emp[:, index],
                    self.cluster_q_m[:, index],
                    self.cluster_q_std[:, index],
                    itc,
                ),
            )
            for index, cluster_id in enumerate(self.cluster_ids)
        )
        n_clusters = len(self.cluster_ids)
        across_statistics = AcrossClusterStatistics(
            n_clusters=n_clusters,
            mean=float(self.D_cluster_mean[itc] * self.diff_scale),
            sample_sd=self._scaled_optional(self.D_cluster_sd[itc]),
            propagated_predicted_sem=float(self.D_cluster_sem_pred[itc] * self.diff_scale),
            propagated_empirical_sem=self._scaled_optional(self.D_cluster_sem_emp[itc]),
        )
        self.analysis_result = AnalysisResult(
            tc=float(tc_disp),
            time_unit=self.time_unit,
            diffusion_unit=self.diffusion_unit,
            pooled=pooled_statistics,
            clusters=cluster_statistics,
            across_clusters=across_statistics,
        )

        self._write_analysis_report(
            out_base,
            self.analysis_result,
            auto_selection=auto_selection,
            requested_auto_min_tc=auto_min_tc,
        )
        self._write_analysis_csv(out_base)
        self._analyzed = True
        self.plot_results(tc_ps, out_base)
        return self.analysis_result

    def _scaled_optional(self, value: float) -> float | None:
        """Scale a finite internal diffusion value or return ``None``.

        Parameters
        ----------
        value : float
            Value in internal diffusion units.

        Returns
        -------
        float or None
            Scaled value, or ``None`` when ``value`` is not finite.
        """
        return float(value * self.diff_scale) if np.isfinite(value) else None

    def _statistics_at(
        self,
        diffusion,
        predicted_sd,
        empirical_sd,
        predicted_sem,
        empirical_sem,
        q_mean,
        q_sd,
        index: int,
    ) -> DiffusionStatistics:
        """Create a selected-lag statistics dataclass from result arrays.

        Parameters
        ----------
        diffusion, predicted_sd, empirical_sd, predicted_sem, empirical_sem : ndarray
            Lag-dependent diffusion and uncertainty arrays in internal units.
        q_mean, q_sd : ndarray
            Lag-dependent Q summary arrays.
        index : int
            Selected lag-grid index.

        Returns
        -------
        DiffusionStatistics
            Scaled selected-lag statistics.
        """
        return DiffusionStatistics(
            diffusion=float(diffusion[index] * self.diff_scale),
            predicted_sd=float(predicted_sd[index] * self.diff_scale),
            empirical_sd=self._scaled_optional(empirical_sd[index]),
            predicted_sem=float(predicted_sem[index] * self.diff_scale),
            empirical_sem=self._scaled_optional(empirical_sem[index]),
            q_mean=float(q_mean[index]) if np.isfinite(q_mean[index]) else None,
            q_sd=float(q_sd[index]) if np.isfinite(q_sd[index]) else None,
        )

    @staticmethod
    def _format_optional(value: float | None, format_spec: str = ".4e") -> str:
        """Format an optional numeric report value.

        Parameters
        ----------
        value : float or None
            Value to format.
        format_spec : str, default='.4e'
            Standard Python floating-point format specification.

        Returns
        -------
        str
            Formatted value or ``'N/A'``.
        """
        return "N/A" if value is None else format(value, format_spec)

    def _write_analysis_report(
        self,
        out_base: str,
        result: AnalysisResult,
        *,
        auto_selection: AutoTcSelection | None,
        requested_auto_min_tc: float | None,
    ) -> None:
        """Write the concise selected-cutoff text report.

        Parameters
        ----------
        out_base : str
            Output path without extension.
        result : AnalysisResult
            Selected-cutoff analysis result.
        auto_selection : AutoTcSelection or None
            Automatic cutoff metadata when diagnostic auto mode was used.
        requested_auto_min_tc : float or None
            User-requested lower cutoff bound.

        Returns
        -------
        None
            Writes ``{out_base}.dat``.
        """
        with open(f"{out_base}.dat", "w", encoding="utf-8") as handle:
            handle.write("DIFFUSION COEFFICIENT ESTIMATE\n")
            handle.write("INPUT:\n")
            handle.write(f"Number of dimensions: {self.ndim}\n")
            handle.write(f"Number of trajectory-level estimates: {self.nseg}\n")
            handle.write(f"Number of independent clusters: {len(self.cluster_ids)}\n")
            handle.write(f"Cluster IDs: {', '.join(self.cluster_ids)}\n")
            handle.write(f"Min/max lag time [steps]: {self.tmin}/{self.tmax}\n")
            handle.write(
                f"Min/max lag time [{self.time_unit}]: "
                f"{self.tmin * self.dt / self.time_scale:.6g}/"
                f"{self.tmax * self.dt / self.time_scale:.6g}\n"
            )
            handle.write(f"Parameter m (MSD points per lag step): {self.m}\n")
            handle.write(f"Total trajectory data points per dimension: {self.n + 1}\n")
            handle.write(
                "Data points per segment and dimension [min/max]: "
                f"{int(np.min(self.segment_lengths))}/{int(np.max(self.segment_lengths))}\n"
            )
            if hasattr(self, "non_converged_count"):
                handle.write(
                    f"Optimizer convergence failures: {self.non_converged_count} "
                    f"({self.percent_failed:.1f}% of {self.total_cases})\n"
                )

            if auto_selection is not None:
                handle.write("AUTO TC SELECTION:\n")
                handle.write(
                    f"Selected auto tc: {result.tc:.4g} {self.time_unit}\n"
                    f"Cluster-mean absolute Q deviation: {auto_selection.selected_score:.6g}\n"
                    f"Maximum cluster Q deviation: {auto_selection.selected_max_deviation:.6g}\n"
                )
                if requested_auto_min_tc is not None:
                    handle.write(
                        f"Requested auto_min_tc: {requested_auto_min_tc:.4g} {self.time_unit}\n"
                        f"Applied auto_min_tc on lag grid: "
                        f"{self.auto_min_tc_used:.4g} {self.time_unit}\n"
                    )
                    if self.tc_auto_unbounded_idx != self.tc_selected_idx:
                        handle.write(
                            f"Plain auto tc without lower bound: "
                            f"{self.tc_auto_unbounded:.4g} {self.time_unit}\n"
                        )

            pooled = result.pooled
            handle.write("POOLED TRAJECTORY-LEVEL RESULT (CONDITIONAL ON SAMPLED CLUSTERS):\n")
            handle.write(
                f"D: {pooled.diffusion:.6e} {self.diffusion_unit}\n"
                f"Predicted member SD: {pooled.predicted_sd:.6e} {self.diffusion_unit}\n"
                f"Empirical member SD: {self._format_optional(pooled.empirical_sd, '.6e')} "
                f"{self.diffusion_unit}\n"
                f"Predicted conditional SEM: {pooled.predicted_sem:.6e} {self.diffusion_unit}\n"
                f"Empirical conditional SEM: {self._format_optional(pooled.empirical_sem, '.6e')} "
                f"{self.diffusion_unit}\n"
                f"Q mean: {self._format_optional(pooled.q_mean, '.6f')}\n"
                f"Q sample SD: {self._format_optional(pooled.q_sd, '.6f')}\n"
            )

            across = result.across_clusters
            handle.write("ACROSS INDEPENDENT CLUSTERS (EQUAL WEIGHT):\n")
            handle.write(
                f"Cluster mean D: {across.mean:.6e} {self.diffusion_unit}\n"
                f"Between-cluster sample SD: {self._format_optional(across.sample_sd, '.6e')} "
                f"{self.diffusion_unit}\n"
                f"Propagated predicted within-cluster conditional SEM: "
                f"{across.propagated_predicted_sem:.6e} {self.diffusion_unit}\n"
                f"Propagated empirical within-cluster conditional SEM: "
                f"{self._format_optional(across.propagated_empirical_sem, '.6e')} "
                f"{self.diffusion_unit}\n"
            )
            handle.write(
                "The propagated conditional SEM and observed between-cluster SD are "
                "reported separately and are not added in quadrature.\n"
            )

            handle.write("INDIVIDUAL CLUSTERS:\n")
            for cluster in result.clusters:
                stats = cluster.statistics
                handle.write(
                    f"{cluster.cluster_id} [n={cluster.n_trajectories}]: "
                    f"D={stats.diffusion:.6e} {self.diffusion_unit}, "
                    f"predicted_member_SD={stats.predicted_sd:.6e} {self.diffusion_unit}, "
                    f"empirical_member_SD={self._format_optional(stats.empirical_sd, '.6e')} "
                    f"{self.diffusion_unit}, predicted_conditional_SEM="
                    f"{stats.predicted_sem:.6e} {self.diffusion_unit}, "
                    f"empirical_conditional_SEM="
                    f"{self._format_optional(stats.empirical_sem, '.6e')} "
                    f"{self.diffusion_unit}, Q={self._format_optional(stats.q_mean, '.6f')}\n"
                )
            handle.write(f"Complete lag-dependent data: {out_base}.csv\n")

    def _write_analysis_csv(self, out_base: str) -> None:
        """Write complete pooled, cluster, and across-cluster lag data.

        Parameters
        ----------
        out_base : str
            Output path without extension.

        Returns
        -------
        None
            Writes ``{out_base}.csv``.
        """
        dimension_columns = [f"D_dim_{dimension}" for dimension in range(self.ndim)]
        fieldnames = [
            "scope",
            "cluster_id",
            "n_trajectories",
            "n_clusters",
            "lag",
            "lag_unit",
            "diffusion",
            "diffusion_unit",
            "predicted_variance",
            "predicted_sd",
            "empirical_sd",
            "conditional_sem_predicted",
            "conditional_sem_empirical",
            "q_mean",
            "q_sd",
            "between_cluster_sample_sd",
            "propagated_within_cluster_sem_predicted",
            "propagated_within_cluster_sem_empirical",
            "q_cluster_mean_abs_deviation",
            "q_cluster_max_abs_deviation",
            *dimension_columns,
        ]

        def csv_number(value):
            """Return a finite float or CSV NaN.

            Parameters
            ----------
            value : float
                Numeric value to normalize for CSV output.

            Returns
            -------
            float
                Finite input value or ``NaN``.
            """
            return float(value) if np.isfinite(value) else np.nan

        with open(f"{out_base}.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for lag_index, step in enumerate(range(self.tmin, self.tmax + 1)):
                lag = step * self.dt / self.time_scale
                pooled_row = {
                    "scope": "pooled",
                    "cluster_id": "",
                    "n_trajectories": self.nseg,
                    "n_clusters": len(self.cluster_ids),
                    "lag": lag,
                    "lag_unit": self.time_unit,
                    "diffusion": self.D[lag_index] * self.diff_scale,
                    "diffusion_unit": self.diffusion_unit,
                    "predicted_variance": (self.Dstd[lag_index] * self.diff_scale) ** 2,
                    "predicted_sd": self.Dstd[lag_index] * self.diff_scale,
                    "empirical_sd": csv_number(self.Dempstd[lag_index] * self.diff_scale),
                    "conditional_sem_predicted": self.Dsem_pred[lag_index] * self.diff_scale,
                    "conditional_sem_empirical": csv_number(
                        self.Dsem_emp[lag_index] * self.diff_scale
                    ),
                    "q_mean": csv_number(self.q_m[lag_index]),
                    "q_sd": csv_number(self.q_std[lag_index]),
                    "between_cluster_sample_sd": np.nan,
                    "propagated_within_cluster_sem_predicted": np.nan,
                    "propagated_within_cluster_sem_empirical": np.nan,
                    "q_cluster_mean_abs_deviation": csv_number(
                        self.q_cluster_score[lag_index]
                    ),
                    "q_cluster_max_abs_deviation": csv_number(
                        self.q_cluster_max_deviation[lag_index]
                    ),
                }
                pooled_row.update(
                    {
                        column: self.Dperdim[lag_index, dimension] * self.diff_scale
                        for dimension, column in enumerate(dimension_columns)
                    }
                )
                writer.writerow(pooled_row)

                for cluster_index, cluster_id in enumerate(self.cluster_ids):
                    cluster_row = {
                        "scope": "cluster",
                        "cluster_id": cluster_id,
                        "n_trajectories": len(self.cluster_indices[cluster_index]),
                        "n_clusters": 1,
                        "lag": lag,
                        "lag_unit": self.time_unit,
                        "diffusion": self.cluster_D[lag_index, cluster_index] * self.diff_scale,
                        "diffusion_unit": self.diffusion_unit,
                        "predicted_variance": (
                            self.cluster_Dstd[lag_index, cluster_index] * self.diff_scale
                        ) ** 2,
                        "predicted_sd": (
                            self.cluster_Dstd[lag_index, cluster_index] * self.diff_scale
                        ),
                        "empirical_sd": csv_number(
                            self.cluster_Dempstd[lag_index, cluster_index] * self.diff_scale
                        ),
                        "conditional_sem_predicted": (
                            self.cluster_Dsem_pred[lag_index, cluster_index] * self.diff_scale
                        ),
                        "conditional_sem_empirical": csv_number(
                            self.cluster_Dsem_emp[lag_index, cluster_index] * self.diff_scale
                        ),
                        "q_mean": csv_number(self.cluster_q_m[lag_index, cluster_index]),
                        "q_sd": csv_number(self.cluster_q_std[lag_index, cluster_index]),
                        "between_cluster_sample_sd": np.nan,
                        "propagated_within_cluster_sem_predicted": np.nan,
                        "propagated_within_cluster_sem_empirical": np.nan,
                        "q_cluster_mean_abs_deviation": np.nan,
                        "q_cluster_max_abs_deviation": np.nan,
                    }
                    cluster_row.update(
                        {
                            column: (
                                self.cluster_Dperdim[lag_index, cluster_index, dimension]
                                * self.diff_scale
                            )
                            for dimension, column in enumerate(dimension_columns)
                        }
                    )
                    writer.writerow(cluster_row)

                across_per_dimension = np.mean(self.cluster_Dperdim[lag_index], axis=0)
                across_row = {
                    "scope": "across_clusters",
                    "cluster_id": "",
                    "n_trajectories": self.nseg,
                    "n_clusters": len(self.cluster_ids),
                    "lag": lag,
                    "lag_unit": self.time_unit,
                    "diffusion": self.D_cluster_mean[lag_index] * self.diff_scale,
                    "diffusion_unit": self.diffusion_unit,
                    "predicted_variance": np.nan,
                    "predicted_sd": np.nan,
                    "empirical_sd": np.nan,
                    "conditional_sem_predicted": np.nan,
                    "conditional_sem_empirical": np.nan,
                    "q_mean": np.nan,
                    "q_sd": np.nan,
                    "between_cluster_sample_sd": csv_number(
                        self.D_cluster_sd[lag_index] * self.diff_scale
                    ),
                    "propagated_within_cluster_sem_predicted": (
                        self.D_cluster_sem_pred[lag_index] * self.diff_scale
                    ),
                    "propagated_within_cluster_sem_empirical": csv_number(
                        self.D_cluster_sem_emp[lag_index] * self.diff_scale
                    ),
                    "q_cluster_mean_abs_deviation": csv_number(
                        self.q_cluster_score[lag_index]
                    ),
                    "q_cluster_max_abs_deviation": csv_number(
                        self.q_cluster_max_deviation[lag_index]
                    ),
                }
                across_row.update(
                    {
                        column: across_per_dimension[dimension] * self.diff_scale
                        for dimension, column in enumerate(dimension_columns)
                    }
                )
                writer.writerow(across_row)

    def plot_tc_suggestion(self, out_base: str, suggestion: TcSuggestion) -> None:
        """Create the publication-cutoff diagnostic figure.

        Parameters
        ----------
        out_base : str
            Output path prefix used by the plotting backend.
        suggestion : TcSuggestion
            Suggestion whose selected or closest candidate is highlighted.

        Returns
        -------
        None
            Writes ``{out_base}.{imgfmt}``.
        """
        plot_tc_suggestion_diagnostics(
            cluster_ids=self.cluster_ids,
            cluster_D=self.cluster_D,
            cluster_q_m=self.cluster_q_m,
            D_cluster_mean=self.D_cluster_mean,
            tmin=self.tmin,
            tmax=self.tmax,
            dt=self.dt,
            time_scale=self.time_scale,
            time_unit=self.time_unit,
            diff_scale=self.diff_scale,
            diffusion_unit=self.diffusion_unit,
            suggestion=suggestion,
            out_base=out_base,
            imgfmt=self.imgfmt,
        )

    def plot_results(self, tc, out_base):
        """Create the analysis figure for the selected lag time.

        Parameters
        ----------
        tc : float
            Selected lag time in ps used to draw the vertical marker.
        out_base : str
            Output path prefix used by the plotting backend.

        Returns
        -------
        None
            Writes the image file to disk.

        Notes
        -----
        This is a thin wrapper around :func:`plot_diffusion_results`.
        """
        tc_auto_unbounded_ps = None
        if self.tc_auto_unbounded is not None:
            tc_auto_unbounded_ps = self.tc_auto_unbounded * self.time_scale

        plot_diffusion_results(
            D=self.D, Dstd=self.Dstd, Dempstd=self.Dempstd,
            q_m=self.q_m, q_std=self.q_std, s2=self.s2,
            cluster_ids=self.cluster_ids, cluster_indices=self.cluster_indices,
            cluster_D=self.cluster_D, cluster_q_m=self.cluster_q_m,
            D_cluster_mean=self.D_cluster_mean, D_cluster_sd=self.D_cluster_sd,
            tmin=self.tmin, tmax=self.tmax, dt=self.dt,
            m=self.m, ndim=self.ndim, nseg=self.nseg,
            time_scale=self.time_scale, time_unit=self.time_unit,
            diff_scale=self.diff_scale, diffusion_unit=self.diffusion_unit,
            tc=tc, tc_auto_unbounded=tc_auto_unbounded_ps, tc_selected_idx=self.tc_selected_idx,
            out_base=out_base, imgfmt=self.imgfmt,
        )


    def finite_size_correction(self, T=300, eta=None, L=None, boxtype='cubic', tc=10):
        """Apply finite-size correction to the diffusion coefficient.

        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        eta : float
            Viscosity in Pa·s.
        L : float
            Edge length of cubic simulation box in nm.
        boxtype : str
            Box geometry (only ``'cubic'`` supported).
        tc : float
            Lag time from the analysis step (in ``time_unit``).

        Returns
        -------
        None
            Stores the corrected diffusion series in ``self.Dcor`` and prints
            the selected corrected value.

        Raises
        ------
        RuntimeError
            If called before :meth:`analysis`.
        ValueError
            If dimensionality/box type is unsupported or required parameters
            are missing.
        """
        if not self._analyzed:
            raise RuntimeError("Call analysis() before finite_size_correction().")
        itc = self._timestep_index(tc)
        if self.ndim != 3:
            raise ValueError("Currently only 3D correction implemented")
        if L is None:
            raise ValueError("Required parameter missing: L, box edge length")
        if eta is None:
            raise ValueError("Required parameter missing: eta, viscosity eta")
        if boxtype != 'cubic':
            raise ValueError("Correction only implemented for cubic simulation boxes")

        kbT = T * BOLTZMANN_K # J
        self.Dcor = self.D + kbT * XI_CUBIC * 1e15 / (6. * np.pi * eta * L) # nm^2 / ps
        Dcor_out = self.Dcor[itc] * self.diff_scale
        Dstd_out = self.Dstd[itc] * self.diff_scale
        tc_disp = (self.tmin + itc) * self.dt / self.time_scale
        print(
            f"Finite-size corrected pooled D at tc={tc_disp:.4g} {self.time_unit}: "
            f"{Dcor_out:.4e} {self.diffusion_unit}; predicted member SD "
            f"{Dstd_out:.4e} {self.diffusion_unit}"
        )

def plot_tc_suggestion_diagnostics(
    *,
    cluster_ids,
    cluster_D,
    cluster_q_m,
    D_cluster_mean,
    tmin,
    tmax,
    dt,
    time_scale,
    time_unit,
    diff_scale,
    diffusion_unit,
    suggestion,
    out_base,
    imgfmt,
):
    """Create the two-panel publication-cutoff diagnostic plot.

    Parameters
    ----------
    cluster_ids : tuple[str, ...]
        Cluster identifiers in plotting order.
    cluster_D : ndarray
        Per-cluster diffusion curves in internal units with shape
        ``(n_lag, n_cluster)``.
    cluster_q_m : ndarray
        Per-cluster mean Q curves with shape ``(n_lag, n_cluster)``.
    D_cluster_mean : ndarray
        Equal-weight cluster mean diffusion curve in internal units.
    tmin, tmax : int
        Minimum and maximum internal lag steps represented by the arrays.
    dt : float
        Internal frame timestep in ps.
    time_scale : float
        Conversion factor from ``time_unit`` to ps.
    time_unit : str
        Display unit for lag times.
    diff_scale : float
        Scale factor from internal diffusion units to ``diffusion_unit``.
    diffusion_unit : str
        Display unit for diffusion values.
    suggestion : TcSuggestion
        Suggestion and diagnostics to visualize.
    out_base : str
        Output filename prefix.
    imgfmt : {'pdf', 'png'}
        Output image format.

    Returns
    -------
    None
        Saves the figure to ``{out_base}.{imgfmt}``.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("paper", font_scale=0.7)
    fig, (ax_d, ax_q) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    xs = np.arange(tmin, tmax + 1, dtype=float) * dt / time_scale
    cluster_colors = sns.color_palette("colorblind", n_colors=max(len(cluster_ids), 1))
    for cluster_index, cluster_id in enumerate(cluster_ids):
        color = cluster_colors[cluster_index]
        ax_d.plot(
            xs,
            cluster_D[:, cluster_index] * diff_scale,
            color=color,
            linewidth=0.9,
            label=f"cluster {cluster_id}",
        )
        ax_q.plot(
            xs,
            cluster_q_m[:, cluster_index],
            color=color,
            linewidth=0.9,
            label=f"cluster {cluster_id}",
        )
    if len(cluster_ids) > 1:
        ax_d.plot(
            xs,
            D_cluster_mean * diff_scale,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label="equal-weight cluster mean",
        )

    evaluated = suggestion.selected_candidate or suggestion.closest_candidate
    selection_color = "tab:green" if suggestion.selected_candidate is not None else "tab:orange"
    selection_label = (
        "suggested common tc"
        if suggestion.selected_candidate is not None
        else "closest failing candidate"
    )
    for axis in (ax_d, ax_q):
        axis.axvspan(
            evaluated.tc,
            evaluated.validation_end,
            color=selection_color,
            alpha=0.08,
            label="validation horizon" if axis is ax_d else None,
        )
        axis.axvline(
            evaluated.tc,
            color=selection_color,
            linestyle="--",
            linewidth=1.2,
            label=selection_label if axis is ax_q else None,
        )

    for cluster_index, onset in enumerate(suggestion.cluster_onsets):
        if onset.tc is None:
            continue
        onset_step = int(round(onset.tc * time_scale / dt))
        onset_idx = onset_step - tmin
        if 0 <= onset_idx < len(xs):
            color = cluster_colors[cluster_index]
            ax_d.plot(
                onset.tc,
                cluster_D[onset_idx, cluster_index] * diff_scale,
                marker="o",
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=4,
            )
            ax_q.plot(
                onset.tc,
                cluster_q_m[onset_idx, cluster_index],
                marker="o",
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=4,
            )

    ax_q.axhspan(
        0.5 - suggestion.q_tolerance,
        0.5 + suggestion.q_tolerance,
        color="gray",
        alpha=0.12,
        label="Q acceptance band",
    )
    ax_q.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)
    ax_d.set_ylabel(fr"$D(t)$ [{diffusion_unit}]")
    ax_d.ticklabel_format(style="scientific", scilimits=(-3, 4))
    ax_d.legend(loc="best", ncol=2)
    ax_q.set_ylabel(r"cluster mean $Q(t)$")
    ax_q.set_xlabel(fr"lag time $t$ [{time_unit}]")
    ax_q.set_ylim(0, 1)
    ax_q.legend(loc="best", ncol=2)
    ax_d.set_xlim(xs[0], xs[-1])
    fig.suptitle(
        "Common cutoff suggestion: "
        + (
            f"{suggestion.tc:.4g} {time_unit}"
            if suggestion.tc is not None
            else "no common plateau"
        )
    )
    fig.tight_layout()
    fig.savefig(f"{out_base}.{imgfmt}", dpi=300)
    plt.close(fig)


def plot_diffusion_results(*, D, Dstd, Dempstd, q_m, q_std, s2,
                           cluster_ids, cluster_indices, cluster_D, cluster_q_m,
                           D_cluster_mean, D_cluster_sd,
                           tmin, tmax, dt, m, ndim, nseg,
                           time_scale, time_unit, diff_scale, diffusion_unit,
                           tc, tc_auto_unbounded, tc_selected_idx, out_base, imgfmt):
    """Create the three-panel diffusion analysis plot.

    Parameters
    ----------
    D : ndarray
        Diffusion coefficient series in internal units (nm^2/ps).
    Dstd : ndarray
        Predicted standard deviation of ``D`` in internal units.
    Dempstd : ndarray
        Empirical standard deviation of ``D`` in internal units.
    q_m : ndarray
        Mean quality factor per lag time.
    q_std : ndarray
        Standard deviation of quality factor per lag time.
    s2 : ndarray
        Segment-wise slope values with shape ``(n_lag, nseg, ndim)``.
    cluster_ids : tuple[str, ...]
        Cluster identifiers in plotting order.
    cluster_indices : tuple[ndarray, ...]
        Segment indices belonging to each cluster.
    cluster_D : ndarray
        Per-cluster diffusion estimates with shape ``(n_lag, n_cluster)``.
    cluster_q_m : ndarray
        Per-cluster Q means with shape ``(n_lag, n_cluster)``.
    D_cluster_mean : ndarray
        Equal-weight cluster mean diffusion series.
    D_cluster_sd : ndarray
        Sample standard deviation across clusters per lag.
    tmin : int
        Minimum lag step index included in the analysis grid.
    tmax : int
        Maximum lag step index included in the analysis grid.
    dt : float
        Internal timestep in ps.
    m : int
        Number of MSD points used per lag step.
    ndim : int
        Number of spatial dimensions.
    nseg : int
        Number of segments/molecules analyzed.
    time_scale : float
        Conversion factor from ``time_unit`` to ps.
    time_unit : str
        Display unit label for time axes.
    diff_scale : float
        Scale factor from nm^2/ps to requested output diffusion unit.
    diffusion_unit : str
        Display unit label for diffusion axes.
    tc : float
        Selected lag time in ps.
    tc_auto_unbounded : float or None
        Unconstrained auto-selected lag time in ps. When it differs from
        ``tc``, an additional comparison marker is drawn on the plot.
    tc_selected_idx : int or None
        Selected lag index in the lag grid, if already known.
    out_base : str
        Output filename prefix.
    imgfmt : str
        Output image format extension (``pdf`` or ``png``).

    Returns
    -------
    None
        Saves the figure to ``{out_base}.{imgfmt}``.

    Notes
    -----
    All inputs are plain arrays/scalars and the function does not depend on
    :class:`Dcov` internals.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context("paper", font_scale=0.5)
    fig = plt.figure(figsize=(6, 7.5))
    gs = fig.add_gridspec(3, 1, height_ratios=(3.0, 2.0, 2.0))
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0])
    xs = np.arange(tmin * dt, (tmax + 1) * dt, dt) / time_scale
    D_out = D * diff_scale
    Dstd_out = Dstd * diff_scale
    Dempstd_out = Dempstd * diff_scale
    show_auto_comparison = (
        tc_auto_unbounded is not None
        and not math.isclose(tc_auto_unbounded, tc, rel_tol=1e-9, abs_tol=1e-12)
    )

    ax0.plot(xs, D_out, color='C0', linewidth=0.8, label=r'$D$')
    ax0.plot(xs, D_out - Dstd_out, color='black', linestyle='dotted', linewidth=0.7,
             label='predicted member SD')
    ax0.plot(xs, D_out + Dstd_out, color='black', linestyle='dotted', linewidth=0.7)
    ax0.fill_between(xs, D_out - Dempstd_out, D_out + Dempstd_out,
                     color='C0', alpha=0.5, edgecolor='none', linewidth=0,
                     label='empirical member SD')
    cluster_colors = sns.color_palette("colorblind", n_colors=max(len(cluster_ids), 1))
    for cluster_index, cluster_id in enumerate(cluster_ids):
        ax0.plot(
            xs,
            cluster_D[:, cluster_index] * diff_scale,
            color=cluster_colors[cluster_index],
            linewidth=0.65,
            alpha=0.75,
            label=f"cluster {cluster_id}",
        )
    if len(cluster_ids) > 1:
        cluster_mean_out = D_cluster_mean * diff_scale
        cluster_sd_out = D_cluster_sd * diff_scale
        ax0.plot(
            xs,
            cluster_mean_out,
            color="tab:purple",
            linestyle="--",
            linewidth=1.1,
            label="equal-weight cluster mean",
        )
        ax0.fill_between(
            xs,
            cluster_mean_out - cluster_sd_out,
            cluster_mean_out + cluster_sd_out,
            color="tab:purple",
            alpha=0.1,
            edgecolor="none",
            label="between-cluster sample SD",
        )
    ax0.axvline(tc / time_scale, color='tab:red', linestyle='dashed')
    if show_auto_comparison:
        ax0.axvline(tc_auto_unbounded / time_scale, color='tab:orange', linestyle='dotted', linewidth=1.2)
    ax0.set(ylabel=fr'$D(t)$ [{diffusion_unit}]')
    ax0.set(xlim=(tmin * dt / time_scale, tmax * dt / time_scale))
    ax0.ticklabel_format(style='scientific', scilimits=(-3, 4))
    ax0.legend(ncol=2)
    ax0.set_title(f"MSD window per lag: t .. {m}\u00d7t [{time_unit}]")

    ax1.plot(xs, q_m, color='C0', linewidth=0.8, label='pooled Q mean')
    ax1.fill_between(xs, q_m - q_std, q_m + q_std,
                     color='C0', alpha=0.5, edgecolor='none', linewidth=0)
    if m > 2:
        for cluster_index, cluster_id in enumerate(cluster_ids):
            ax1.plot(
                xs,
                cluster_q_m[:, cluster_index],
                color=cluster_colors[cluster_index],
                linewidth=0.65,
                alpha=0.8,
                label=f"cluster {cluster_id}",
            )
    ax1.axhline(0.5, linestyle='dashed', color='gray', linewidth=1.2)
    ax1.axvline(
        tc / time_scale,
        color='tab:red',
        linestyle='dashed',
        label='selected tc',
    )
    if show_auto_comparison:
        ax1.axvline(
            tc_auto_unbounded / time_scale,
            color='tab:orange',
            linestyle='dotted',
            linewidth=1.2,
            label='plain auto tc',
        )
    ax1.set(ylabel=r'$Q(t)$')
    ax1.set(xlabel=fr'lag time $t$ [{time_unit}]')
    ax1.set(ylim=(0, 1))
    if m > 2:
        ax1.legend(loc='best', ncol=2)

    # Distribution of per-segment/molecule estimates at the selected tc
    itc = tc_selected_idx
    if itc is None:
        step = int(round(tc / dt))
        itc = step - tmin

    if 0 <= itc < len(D) and nseg > 0:
        D_seg_tc = s2[itc].sum(axis=1) / (2.0 * ndim * dt) * diff_scale
        violin_kwargs = dict(positions=[0], showmeans=False, showextrema=False, showmedians=False)
        try:
            parts = ax2.violinplot([D_seg_tc], orientation='horizontal', **violin_kwargs)
        except TypeError:
            parts = ax2.violinplot([D_seg_tc], vert=False, **violin_kwargs)
        for body in parts.get('bodies', []):
            body.set_facecolor('C0')
            body.set_alpha(0.5)
            body.set_edgecolor('none')
            body.set_linewidth(0.0)

        d_mean = float(np.mean(D_seg_tc))
        d_median = float(np.median(D_seg_tc))
        q05, q25, q75, q95 = np.percentile(D_seg_tc, [5, 25, 75, 95])

        ax2.axvline(D_out[itc], color='black', linestyle='solid', linewidth=1.2,
                    label=r'$D(t_c)$ (estimate)')
        ax2.axvline(d_mean, color='C0', linestyle='dashed', linewidth=1.2,
                    label=r'mean($D_i$)')
        ax2.axvline(d_median, color='C0', linestyle='solid', linewidth=1.2,
                    label=r'median($D_i$)')
        for cluster_index, (cluster_id, indices) in enumerate(zip(cluster_ids, cluster_indices)):
            cluster_values = D_seg_tc[indices]
            y_offset = np.full_like(cluster_values, 0.06 * (cluster_index - (len(cluster_ids) - 1) / 2))
            ax2.plot(
                cluster_values,
                y_offset,
                '|',
                color=cluster_colors[cluster_index],
                alpha=0.55,
                markersize=10,
            )
            ax2.plot(
                [cluster_D[itc, cluster_index] * diff_scale],
                [y_offset[0] if len(y_offset) else 0.0],
                marker='o',
                color=cluster_colors[cluster_index],
                markersize=3.5,
                label=f"{cluster_id} mean",
            )
        ax2.axvspan(q25, q75, color='C0', alpha=0.12, label='25\u201375%')
        ax2.axvspan(q05, q95, color='C0', alpha=0.05, label='5\u201395%')

        ax2.set(yticks=[], ylabel='Density of $D_i$')
        ax2.set_ylim(-0.6, 0.6)
        ax2.ticklabel_format(style='scientific', scilimits=(-3, 4))
        ax2.set(xlabel=fr'$D$ [{diffusion_unit}]')
        ax2.set_title(f"Per-segment $D$ at $t_c={tc / time_scale:.4g}$ {time_unit} (n={nseg})")
        ax2.legend(loc='best')
    else:
        ax2.axis('off')

    fig.tight_layout(h_pad=0.2)
    fig.savefig(f'{out_base}.{imgfmt}', dpi=300)
    plt.close(fig)
