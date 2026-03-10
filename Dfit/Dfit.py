# Code for the estimation of translational diffusion coefficients from simulation data
# version 1.0 (06/02/2020)
# Jakob Tomas Bullerjahn
# Soeren von Buelow (soeren.buelow@biophys.mpg.de)
# Gerhard Hummer

# Please read and cite the paper: 
# J. Bullerjahn, S. v. Buelow, G. Hummer: Optimal estimates of diffusion coefficients from molecular dynamics simulations, Journal of Chemical Physics 153, 024116 (2020).

import math
from collections.abc import Sequence
from pathlib import Path
import warnings
import concurrent.futures
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.special import gammainc

from . import math_utils
from .trajectory_reader import get_reader, TrajectoryReader

TrajectoryInput = str | Path | Sequence[str | Path] | None

XI_CUBIC = 2.837297
BOLTZMANN_K = 1.380649e-23 # J/K

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

    c = math_utils.setupc(m,n) # setups of cov matrix
    cov = math_utils.calc_cov(n,m,c,a2full_3D,s2full_3D)
    cinv = math_utils.inv_mat(cov)

    chi2 = math_utils.calc_chi2(m,a2_3D,s2_3D,msds_3D,cinv,ndim)

    if chi2 <= 0:
        q = 1.0
    else:
        q = 1-gammainc( (m-2)/2.,chi2/2.) # goodness-of-fit Q
    return q

def analyze_chunk_task(chunk_data, step, m, dt, nperseg, multi, ndim, d2max, nitmax, c2_pre, cm_pre, a2full_3D=0.0, s2full_3D=0.0):
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
    nperseg : int
        Number of steps per segment before striding.
    multi : bool
        ``True`` for multi-trajectory mode where each input trajectory is one
        segment/molecule, ``False`` for single-trajectory segmentation mode.
    ndim : int
        Number of spatial dimensions.
    d2max : float
        Convergence threshold for iterative GLS updates.
    nitmax : int
        Maximum number of GLS iterations.
    c2_pre : ndarray or None
        Optional precomputed covariance helper matrix for ``m=2``.
    cm_pre : ndarray or None
        Optional precomputed covariance helper matrix for ``m``.
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
            max_lag_steps = max(1, nperseg // m)
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
            z_dim = z_analyzed[:, d]
            msds[d] = math_utils.compute_MSD_1D_via_correlation(z_dim)[1:(m+1)]
            
            # Check if pre-calculated matrices match current n
            use_c2 = c2_pre
            use_cm = cm_pre
            
            res_a2[d], res_s2[d], conv = math_utils.calc_gls(n_per_seg_step, m, msds[d], d2max, nitmax, c2=use_c2, cm=use_cm)
            if not conv: res_converged = False
        
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
        z_dim = full_z[:, d][::step]
        n = len(z_dim) - 1
        msd = math_utils.compute_MSD_1D_via_correlation(z_dim)[1:(m + 1)]
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
    3. Call :meth:`analysis` to compute summary statistics and write outputs.
    4. Optionally call :meth:`finite_size_correction` (3D, cubic box only).

    Notes
    -----
    Internal diffusion units are nm^2/ps. Reported units are controlled through
    ``diffusion_unit``.
    """
    def __init__(self, fz: TrajectoryInput = None, universe=None, universes=None, selection=None,
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
        m : int, default=20
            Number of MSD values used in each lag-time GLS fit window.
        tmin : float or None, optional
            Minimum lag time in ``time_unit``. If ``None``, defaults to one
            frame step.
        tmax : float, default=100.0
            Maximum lag time in ``time_unit``.
        dt : float or None, optional
            Frame timestep in ``time_unit``. If ``None``, adopts the reader
            timestep (1.0 ps for text files, trajectory timestep for
            MDAnalysis).
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
        
        if dt is None:
            self.dt = self.reader.dt  # adopt reader's dt (already in ps)
        else:
            self.dt = dt * self.time_scale  # convert user-provided dt to ps
            if not math.isclose(self.dt, self.reader.dt, rel_tol=1e-5):
                warnings.warn(
                    f"User provided dt ({dt} {self.time_unit} = {self.dt} ps) differs from "
                    f"reader's dt ({self.reader.dt} ps). Using provided dt.",
                    UserWarning,
                )

        self.m = m
        
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
                if nseg < 1:
                    raise ValueError('nseg must be at least 1')
                if auto_nseg > 0 and nseg > auto_nseg:
                    print(f"Warning, too many segments chosen, falling back to nseg = {auto_nseg}")
                    self.nseg = auto_nseg
                else:
                    self.nseg = nseg
            
            self.nperseg = int(total_points / self.nseg) - 1
            if self.nperseg < 1:
                raise ValueError(f'nseg={self.nseg} yields segment length < 2 points; reduce nseg or use a longer trajectory')

        if self.m > self.nperseg:
            self.m = self.nperseg

        if self.m < 2:
            raise ValueError(
                f"m={self.m} after clamping to segment length ({self.nperseg}); "
                f"m must be >= 2. Use a longer trajectory or reduce nseg/tmax."
            )

        # In multi-traj mode, ensure tmax is feasible given segment length and m
        if self.multi:
            max_tmax_steps = max(1, self.nperseg // self.m)
            if self.tmax > max_tmax_steps:
                warnings.warn(
                    f"tmax ({self.tmax * self.dt / self.time_scale:.4g} {self.time_unit}) is too long for segment length; "
                    f"reducing to {max_tmax_steps * self.dt / self.time_scale:.4g} {self.time_unit}.",
                    UserWarning
                )
                self.tmax = max_tmax_steps
            if self.tmin > self.tmax:
                raise ValueError(
                    f"tmin ({self.tmin * self.dt / self.time_scale:.4g} {self.time_unit}) exceeds max feasible tmax "
                    f"({self.tmax * self.dt / self.time_scale:.4g} {self.time_unit}) for m={self.m} and trajectory length {self.nperseg+1}."
                )

        # Arrays
        self.a2full = np.zeros((self.tmax-self.tmin+1, self.ndim))
        self.s2full = np.zeros((self.tmax-self.tmin+1, self.ndim))

        self.a2 = np.zeros((self.tmax-self.tmin+1, self.nseg, self.ndim))
        self.s2 = np.zeros((self.tmax-self.tmin+1, self.nseg, self.ndim))

        self.s2var = np.zeros((self.tmax-self.tmin+1))
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
        steps = (tc * self.time_scale) / self.dt
        steps_int = round(steps)
        if not math.isclose(steps, steps_int, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f'tc [{tc}] must be a multiple of dt [{self.dt}] within numerical tolerance')
        itc = int(steps_int) - self.tmin
        if itc < 0 or itc >= (self.tmax - self.tmin + 1):
            raise ValueError(f'tc [{tc}] is outside the computed timestep range')
        return itc

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

        # Warm up numba-compiled kernels to avoid a slow first task
        try:
            dummy = np.zeros(8)
            _ = math_utils.compute_MSD_1D_via_correlation(dummy)
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

        n_workers_count = max_workers if max_workers else (os.cpu_count() or 1)
        # One chunk per worker, sized to spread segments evenly.
        # With many lag steps the pool is kept busy even with small chunk sizes.
        chunk_size = max(1, math.ceil(self.nseg / n_workers_count))

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
            # Phase 2: Segment fits for every (lag_step, chunk) combination.
            # All futures are submitted at once so the thread pool can schedule
            # them freely and run at peak parallelism.
            # ------------------------------------------------------------------
            # future -> (t, step, s_start, s_end)
            seg_futures: dict[concurrent.futures.Future, tuple[int, int, int, int]] = {}

            for t, step in enumerate(range(self.tmin, self.tmax + 1)):
                n_per_seg_step = int(self.nperseg / step)

                c2_pre = math_utils.setupc(2, n_per_seg_step) if n_per_seg_step >= 2 else None
                cm_pre = math_utils.setupc(self.m, n_per_seg_step) if n_per_seg_step >= self.m else None

                a2full_3D_val = float(np.sum(self.a2full[t])) if not self.multi else 0.0
                s2full_3D_val = float(np.sum(self.s2full[t])) if not self.multi else 0.0

                for s_start in range(0, self.nseg, chunk_size):
                    s_end = min(s_start + chunk_size, self.nseg)

                    if self.multi:
                        chunk_data = all_trajs[s_start:s_end]
                    else:
                        full_z = all_trajs[0]
                        chunk_data = [
                            full_z[s * (self.nperseg + 1):(s + 1) * (self.nperseg + 1)]
                            for s in range(s_start, s_end)
                        ]

                    fut = executor.submit(
                        analyze_chunk_task,
                        chunk_data, step, self.m, self.dt, self.nperseg,
                        self.multi, self.ndim, self.d2max, self.nitmax,
                        c2_pre, cm_pre, a2full_3D_val, s2full_3D_val
                    )
                    seg_futures[fut] = (t, step, s_start, s_end)

            seg_iter = concurrent.futures.as_completed(seg_futures)
            if self.progress:
                desc = "segment fits" if not self.multi else "lag steps"
                seg_iter = tqdm(seg_iter, total=len(seg_futures), desc=desc)

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
            n_per_seg_step = int(self.nperseg / step)
            a2m = np.mean(self.a2[t], axis=0)
            s2m = np.mean(self.s2[t], axis=0)
            self.s2var[t] = math_utils.eval_vars(n_per_seg_step, self.m, a2m, s2m, self.ndim)

            self.a2[t] /= step
            self.s2[t] /= step
            self.s2var[t] /= step ** 2
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
    def analysis(self, tc: float | str = 10, fout_prefix: str | None = None):
        """Compute diffusion coefficient statistics and write output files.

        Parameters
        ----------
        tc : float or 'auto'
            Lag time for the diffusion estimate (in ``time_unit``). Must be a
            multiple of dt.  Pass ``'auto'`` to select the lag where Q ≈ 0.5.
        fout_prefix : str, optional
            Custom base name for output files.  Default: ``{fout}.tc_{tc}``.

        Returns
        -------
        None
            This method updates analysis attributes in place.

        Raises
        ------
        RuntimeError
            If called before :meth:`run_Dfit`.
        ValueError
            If ``tc`` is invalid for the computed lag-time grid.

        Notes
        -----
        Side effects include:

        - Populating summary attributes (``D``, ``Dstd``, ``Dempstd``,
          ``Dsem_pred``, ``Dsem_emp``, ``q_m``, ``q_std``).
        - Recording selected lag time in ``tc_selected`` and
          ``tc_selected_idx``.
        - Writing ``.dat`` and plot output files.
        - Setting lifecycle flag ``_analyzed=True``.
        """
        if not self._fitted:
            raise RuntimeError("Call run_Dfit() before analysis().")

        # Calculate statistics
        Dseg = self.s2.sum(axis=2) # across dims
        self.Dseg = np.mean(Dseg, axis=1) / (2.*self.ndim*self.dt) # mean across segs, nm^2 / (dt * ps)

        if np.any(self.s2var < 0):
            warnings.warn("Negative variance encountered; clamping to zero.", RuntimeWarning)
        s2var_clamped = np.maximum(self.s2var, 0.0)
        self.Dstd = np.sqrt(s2var_clamped/ (2.*self.ndim*self.dt)**2) # nm^2 / (dt * ps)

        if self.multi: # no 'full' run available
            self.D = self.Dseg # nm^2 / (dt * ps)
            self.Dperdim = np.mean(self.s2,axis=1) / (2.*self.dt) # mean across segs
        else: # use full run
            self.D = self.s2full.sum(axis=1)/(2.*self.ndim*self.dt) # nm^2 / (dt * ps)
            self.Dperdim = self.s2full / (2.*self.dt)

        Dempstd = np.var(self.s2, axis=1) # across segments per dim
        Dempstd = np.sum(Dempstd, axis=1) # across dims
        if np.any(Dempstd < 0):
            warnings.warn("Negative empirical variance encountered; clamping to zero.", RuntimeWarning)
        Dempstd_clamped = np.maximum(Dempstd, 0.0)
        self.Dempstd = np.sqrt(Dempstd_clamped) / (2.*self.ndim*self.dt)
        self.q_m = np.mean(self.q, axis=1)
        self.q_std = np.std(self.q, axis=1)

        sem_denom = math.sqrt(max(int(self.nseg), 1))
        self.Dsem_pred = self.Dstd / sem_denom
        self.Dsem_emp = self.Dempstd / sem_denom

        # Scaled values for reporting
        D_out = self.D * self.diff_scale
        Dstd_out = self.Dstd * self.diff_scale
        Dempstd_out = self.Dempstd * self.diff_scale
        Dsem_pred_out = self.Dsem_pred * self.diff_scale
        Dsem_emp_out = self.Dsem_emp * self.diff_scale
        Dperdim_out = self.Dperdim * self.diff_scale

        if tc == 'auto':
            # Find index where q_m is closest to 0.5
            # self.q_m is array of shape (tmax-tmin+1,)
            # We want to minimize abs(q_m - 0.5)
            # Note: q_m indices correspond to steps tmin...tmax
            
            diff = np.abs(self.q_m - 0.5)
            idx_min = np.argmin(diff)
            itc = idx_min
            
            # Calculate actual tc value for reporting
            # itc is 0-indexed relative to self.tmin
            step = self.tmin + itc
            tc_ps = step * self.dt  # ps
            tc_disp = tc_ps / self.time_scale  # user unit
            
            print(f"Automatically selected tc = {tc_disp:.4g} {self.time_unit} (Q = {self.q_m[itc]:.4f})")
        else:
            itc = self._timestep_index(tc)
            step = self.tmin + itc
            tc_ps = step * self.dt  # ps
            tc_disp = tc_ps / self.time_scale  # user unit

        # store selected tc for later access
        self.tc_selected_idx = itc
        self.tc_selected = tc_disp
        
        # Determine output filename base
        if fout_prefix is not None:
            out_base = fout_prefix
        else:
            # Format tc for filename to avoid slashes or dots if possible or just standard format
            # Using fixed precision or string format if it was a string (though it's converted to float)
            # tc_disp is float.
            out_base = f"{self.fout}.tc_{tc_disp:.4g}"

        with open(f'{out_base}.dat','w') as g:
            g.write("DIFFUSION COEFFICIENT ESTIMATE\n")
            g.write("INPUT:\n")
            # g.write("Trajectory: {}\n".format(self.fz)) # fz might be None now
            g.write(f"Number of dimensions : {self.ndim}\n")
            g.write(f"Min/max lag time [steps]: {self.tmin}/{self.tmax}\n")
            g.write(f"Min/max lag time [{self.time_unit}]: {self.tmin*self.dt/self.time_scale}/{self.tmax*self.dt/self.time_scale}\n")
            g.write(f"Parameter m (MSD points per lag step): {self.m}\n")
            g.write(f"MSD window per lag step t: [{self.time_unit}] t to {self.m}*t (e.g., at t={self.tmin*self.dt/self.time_scale:.4g} {self.time_unit}, window spans up to {(self.m*self.tmin*self.dt)/self.time_scale:.4g} {self.time_unit})\n")
            
            if self.multi:
                g.write(f"Number of molecules: {self.nseg}\n")
            else:
                g.write(f"Number of segments: {self.nseg}\n")
                
            g.write(f"Total number of trajectory data points per dim.: {self.n+1}\n")
            g.write(f"Data points per segment and dim.: {self.nperseg+1}\n")
            
            if hasattr(self, 'non_converged_count'):
                 g.write(f"Optimizer convergence failures: {self.non_converged_count} ({self.percent_failed:.1f}% of {self.total_cases})\n")

            # Summary at chosen tc
            g.write(f"Your chosen diffusion coefficient at {tc_disp} {self.time_unit}: {D_out[itc]:.4e} {self.diffusion_unit}\n")
            g.write(f"Standard deviation at {tc_disp} {self.time_unit}: {Dstd_out[itc]:.4e} {self.diffusion_unit}\n")
            g.write(f"Empirical std at {tc_disp} {self.time_unit}: {Dempstd_out[itc]:.4e} {self.diffusion_unit}\n")
            g.write(f"SEM (predicted) at {tc_disp} {self.time_unit} [n={self.nseg}]: {Dsem_pred_out[itc]:.4e} {self.diffusion_unit}\n")
            g.write(f"SEM (empirical) at {tc_disp} {self.time_unit} [n={self.nseg}]: {Dsem_emp_out[itc]:.4e} {self.diffusion_unit}\n")
            g.write(f"Q-factor at {tc_disp} {self.time_unit}: {self.q_m[itc]:.4f}\n")
            if self.diffusion_unit != 'cm2/s':
                g.write(f"Your chosen diffusion coefficient at {tc_disp} {self.time_unit}: {self.D[itc]*0.01:.4e} cm^2/s\n")
            g.write("DIFFUSION COEFFICIENT OUTPUT SUMMARY:\n")
            g.write(f"t[{self.time_unit}] D[{self.diffusion_unit}] varD[{self.diffusion_unit}^2] Q\n")
            for t,step in enumerate(range(self.tmin, self.tmax+1)):
                g.write(f"{(step*self.dt)/self.time_scale:.4g} {D_out[t]:.5g} {(Dstd_out[t]**2):.5g} {self.q_m[t]:.5f}\n")
            if self.ndim > 1:
                g.write("\nDIFFUSION COEFFICIENT PER DIMENSION:\n")
                g.write(f"t[{self.time_unit}] Dx[{self.diffusion_unit}] Dy[{self.diffusion_unit}] ...\n")
                for step, Dt in zip(range(self.tmin, self.tmax+1), self.Dperdim):
                    g.write(f"{(step*self.dt)/self.time_scale:.4f} {Dperdim_out[step-self.tmin]}\n")
        
        self._analyzed = True
        self.plot_results(tc_ps, out_base)

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
        plot_diffusion_results(
            D=self.D, Dstd=self.Dstd, Dempstd=self.Dempstd,
            q_m=self.q_m, q_std=self.q_std, s2=self.s2,
            tmin=self.tmin, tmax=self.tmax, dt=self.dt,
            m=self.m, ndim=self.ndim, nseg=self.nseg,
            time_scale=self.time_scale, time_unit=self.time_unit,
            diff_scale=self.diff_scale, diffusion_unit=self.diffusion_unit,
            tc=tc, tc_selected_idx=self.tc_selected_idx,
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
        print(f"Finite-size corrected D at tc={tc_disp:.4g} {self.time_unit}: "
              f"{Dcor_out:.4e} {self.diffusion_unit} ± {Dstd_out:.4e} {self.diffusion_unit}")


def plot_diffusion_results(*, D, Dstd, Dempstd, q_m, q_std, s2,
                           tmin, tmax, dt, m, ndim, nseg,
                           time_scale, time_unit, diff_scale, diffusion_unit,
                           tc, tc_selected_idx, out_base, imgfmt):
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

    ax0.plot(xs, D_out, color='C0', linewidth=1.0, label=r'$D$')
    ax0.plot(xs, D_out - Dstd_out, color='black', linestyle='dotted', linewidth=0.7,
             label=r'$\delta \overline{D}^\mathrm{predicted}$')
    ax0.plot(xs, D_out + Dstd_out, color='black', linestyle='dotted', linewidth=0.7)
    ax0.fill_between(xs, D_out - Dempstd_out, D_out + Dempstd_out,
                     color='C0', alpha=0.5, edgecolor='none', linewidth=0,
                     label=r'$\delta \overline{D}^\mathrm{empirical}$')
    ax0.axvline(tc / time_scale, color='tab:red', linestyle='dashed')
    ax0.set(ylabel=fr'$D(t)$ [{diffusion_unit}]')
    ax0.set(xlim=(tmin * dt / time_scale, tmax * dt / time_scale))
    ax0.ticklabel_format(style='scientific', scilimits=(-3, 4))
    ax0.legend(ncol=2)
    ax0.set_title(f"MSD window per lag: t .. {m}\u00d7t [{time_unit}]")

    ax1.plot(xs, q_m, color='C0')
    ax1.fill_between(xs, q_m - q_std, q_m + q_std,
                     color='C0', alpha=0.5, edgecolor='none', linewidth=0)
    ax1.axhline(0.5, linestyle='dashed', color='gray', linewidth=1.2)
    ax1.axvline(tc / time_scale, color='tab:red', linestyle='dashed')
    ax1.set(ylabel=r'$Q(t)$')
    ax1.set(xlabel=fr'lag time $t$ [{time_unit}]')
    ax1.set(ylim=(0, 1))

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
        ax2.plot(D_seg_tc, np.zeros_like(D_seg_tc), '|', color='C0', alpha=0.3, markersize=10)
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
