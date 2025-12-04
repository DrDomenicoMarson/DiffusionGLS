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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import progressbar as bar
from scipy.special import gammainc

from . import math_utils
from .trajectory_reader import get_reader, TrajectoryReader

TrajectoryInput = str | Path | Sequence[str | Path] | None

XI_CUBIC = 2.837297
BOLTZMANN_K = 1.380649e-23 # J/K

def calc_q(n,m,a2_3D,s2_3D,msds_3D,a2full_3D,s2full_3D,ndim):
    """ Q per segment (Eq. 22). Use best a2 and s2 estimates from 
    full trajectory for cov and cinv. """

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

class Dcov():
    """
    Diffusion coefficient estimator using GLS on mean-squared displacement.
    
    Parameters of interest:
    - m (int): number of MSD points used per lag step (unitless). For a given lag step s,
      the fit uses MSD values at times s*dt, 2*s*dt, ..., m*s*dt. Larger m widens the
      lag-time window (m*s*dt) but requires longer segments and increases cost.
    """
    def __init__(self, fz: TrajectoryInput = None, universe=None, selection=None,
                 m: int = 20, tmin: float = None, tmax: float = 100.0, dt: float = 1.0,
                 d2max: float = 1e-10, nitmax: int = 100,
                 nseg: int | None = None, imgfmt: str = 'pdf', fout: str = 'D_analysis',
                 n_jobs: int = -1, normalize_lengths: bool = False, time_unit: str = 'ps'):

        # Initialize Reader
        self.reader: TrajectoryReader = get_reader(fz=fz, universe=universe, selection=selection, normalize_lengths=normalize_lengths)
        
        # Use reader properties
        self.ndim = self.reader.ndim
        self.n = self.reader.n_frames - 1 # N is steps, n_frames is points

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
        
        self.dt = dt * self.time_scale  # internal dt in ps
        if not math.isclose(self.dt, self.reader.dt, rel_tol=1e-5):
             warnings.warn(f"User provided dt ({dt} {self.time_unit}) differs from reader's dt ({self.reader.dt} ps). Using provided dt.", UserWarning)

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
                    raise ValueError('Timeseries too short! Reduce tmax or provide nseg >= 1')
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

        # Arrays
        self.a2full = np.zeros((self.tmax-self.tmin+1, self.ndim))
        self.s2full = np.zeros((self.tmax-self.tmin+1, self.ndim))

        self.a2 = np.zeros((self.tmax-self.tmin+1, self.nseg, self.ndim))
        self.s2 = np.zeros((self.tmax-self.tmin+1, self.nseg, self.ndim))

        self.s2var = np.zeros((self.tmax-self.tmin+1))
        self.q = np.zeros((self.tmax-self.tmin+1, self.nseg))

        # Results
        self.Dseg = None
        self.Dstd = None
        self.Dperdim = None
        self.D = None
        self.Dempstd = None
        self.q_m = None
        self.q_std = None
        self.tc_selected = None  # selected tc in chosen time unit
        self.tc_selected_idx = None  # index into lag arrays

    def _timestep_index(self, tc: float) -> int:
        steps = (tc * self.time_scale) / self.dt
        steps_int = round(steps)
        if not math.isclose(steps, steps_int, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f'tc [{tc}] must be a multiple of dt [{self.dt}] within numerical tolerance')
        itc = int(steps_int) - self.tmin
        if itc < 0 or itc >= (self.tmax - self.tmin + 1):
            raise ValueError(f'tc [{tc}] is outside the computed timestep range')
        return itc

    def run_Dfit(self):
        """ Main Function to calculate the stepsize sigma^2 and offset a^2. """
        
        # Pre-load data if it fits in memory or iterate?
        # The original code loaded everything. 
        # Our reader yields full trajectories.
        # To avoid re-reading for every 'step' in the outer loop, we should probably load it once
        # IF we can afford it.
        # The outer loop iterates over 'step' (lag time).
        # We need the full trajectory for every lag time calculation?
        # Actually, the original code sliced: z[::step].
        
        # Optimization: Read all data into a list of arrays once.
        # If memory is an issue, we would need to re-read from disk/MDA every time, which is slow.
        # Let's assume we load it.
        
        all_trajs = list(self.reader) # List of (N_frames, ndim) arrays

        # Warm up numba-compiled kernels once to avoid a slow first step
        try:
            dummy = np.zeros(8)
            _ = math_utils.compute_MSD_1D_via_correlation(dummy)
            _ = math_utils.calc_gls(5, 3, np.arange(3, dtype=np.float64), self.d2max, 2,
                                   c2=math_utils.setupc(2, 5), cm=math_utils.setupc(3, 5))
        except Exception:
            # If warm-up fails for any reason, continue; main computation will trigger JIT anyway.
            pass
        
        non_converged_count = 0

        # Determine worker count (treat n_jobs=0 as serial)
        if self.n_jobs is None or self.n_jobs < 0:
            max_workers = None
        elif self.n_jobs == 0:
            max_workers = 1
        else:
            max_workers = self.n_jobs

        # Use threads to avoid pickling large trajectory chunks; numba kernels release the GIL
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            with bar.ProgressBar(max_value=self.tmax-self.tmin+1) as progbar:
                for t, step in enumerate(range(self.tmin, self.tmax+1)):
                    
                    # 1. Full Trajectory Analysis (Only for Single Trajectory mode)
                    if not self.multi:
                        # In single mode, all_trajs has 1 element
                        full_z = all_trajs[0]
                        converged_all_dims = True
                        for d in range(self.ndim):
                            z_dim = full_z[:, d]
                            z_strided = z_dim[::step]
                            if len(z_strided) <= self.m:
                                 raise ValueError(f"Trajectory too short (length N={len(z_strided)}) to calculate m={self.m} MSD points at lag time step={step} (t={step*self.dt} ps). Please reduce tmax or m, or use a longer trajectory.")
                            n = len(z_strided) - 1
                            msd = math_utils.compute_MSD_1D_via_correlation(z_strided)[1:(self.m+1)]
                            self.a2full[t,d], self.s2full[t,d], converged = math_utils.calc_gls(n, self.m, msd, self.d2max, self.nitmax)
                            if not converged: converged_all_dims = False
                        
                        if not converged_all_dims: non_converged_count += 1
                        
                        a2full_3D = np.sum(self.a2full[t])
                        s2full_3D = np.sum(self.s2full[t])
                    
                    # 2. Segment Analysis
                    n_per_seg_step = int(self.nperseg / step)
                    
                    # Pre-calculate covariance matrices for this step
                    c2_pre = None
                    if n_per_seg_step >= 2:
                        c2_pre = math_utils.setupc(2, n_per_seg_step)
                    
                    cm_pre = None
                    if n_per_seg_step >= self.m:
                        cm_pre = math_utils.setupc(self.m, n_per_seg_step)
                    
                    # Prepare chunks (coarser to reduce overhead)
                    n_workers_count = max_workers if max_workers else (os.cpu_count() or 1)
                    chunk_size = max(1, math.ceil(self.nseg / n_workers_count))
                    
                    # Prepare full trajectory values if single mode
                    a2full_3D_val = 0.0
                    s2full_3D_val = 0.0
                    if not self.multi:
                        a2full_3D_val = np.sum(self.a2full[t])
                        s2full_3D_val = np.sum(self.s2full[t])
                    
                    future_to_range = {}
                    for s_start in range(0, self.nseg, chunk_size):
                        s_end = min(s_start + chunk_size, self.nseg)
                        
                        chunk_data = []
                        if self.multi:
                            chunk_data = all_trajs[s_start:s_end]
                        else:
                            # Slice the single trajectory
                            full_z = all_trajs[0]
                            for s in range(s_start, s_end):
                                zstart = s * (self.nperseg + 1)
                                zend = (s + 1) * (self.nperseg + 1)
                                chunk_data.append(full_z[zstart:zend])
                                
                        fut = executor.submit(analyze_chunk_task, chunk_data, step, self.m, self.dt, self.nperseg, self.multi, self.ndim, self.d2max, self.nitmax, c2_pre, cm_pre, a2full_3D_val, s2full_3D_val)
                        future_to_range[fut] = (s_start, s_end)
                    
                    for future in concurrent.futures.as_completed(future_to_range):
                        s_start, s_end = future_to_range[future]
                        try:
                            chunk_results, error = future.result()
                            if error:
                                raise error
                            
                            for i, (res_a2, res_s2, q_val, res_converged) in enumerate(chunk_results):
                                s = s_start + i
                                self.a2[t,s] = res_a2
                                self.s2[t,s] = res_s2
                                self.q[t,s] = q_val
                                if not res_converged: non_converged_count += 1
                        except Exception as exc:
                            raise exc

                    # 3. Averaging
                    a2m = np.mean(self.a2[t], axis=0)
                    s2m = np.mean(self.s2[t], axis=0)
                    
                    self.s2var[t] = math_utils.eval_vars(n_per_seg_step, self.m, a2m, s2m, self.ndim)
                    
                    # Normalize by step
                    self.a2[t] /= step
                    self.s2[t] /= step
                    self.s2var[t] /= step**2
                    if not self.multi:
                        self.a2full[t] /= step
                        self.s2full[t] /= step
                    
                    progbar.update(t+1)
        
        self.non_converged_count = non_converged_count
        self.total_cases = (self.tmax - self.tmin + 1) * (self.nseg + (1 if not self.multi else 0))
        self.percent_failed = 0.0
        
        if non_converged_count > 0:
            self.percent_failed = (non_converged_count / self.total_cases) * 100
            print(f"WARNING: Optimizer did not converge in {non_converged_count} cases ({self.percent_failed:.1f}% of Total {self.total_cases}). Falling back to M=2 for those cases.")

    # Output and plotting
    def analysis(self, tc: float | str = 10):
        # Calculate statistics first
        Dseg = self.s2.sum(axis=2) # across dims
        self.Dseg = np.mean(Dseg, axis=1) / (2.*self.ndim*self.dt) # mean across segs, nm^2 / (dt * ps)
        self.Dstd = np.sqrt(self.s2var/ (2.*self.ndim*self.dt)**2) # nm^2 / (dt * ps)

        if self.multi: # no 'full' run available
            self.D = self.Dseg # nm^2 / (dt * ps)
            self.Dperdim = np.mean(self.s2,axis=1) / (2.*self.dt) # mean across segs
        else: # use full run
            self.D = self.s2full.sum(axis=1)/(2.*self.ndim*self.dt) # nm^2 / (dt * ps)
            self.Dperdim = self.s2full / (2.*self.dt)

        Dempstd = np.var(self.s2, axis=1) # across segments per dim
        Dempstd = np.sum(Dempstd, axis=1) # across dims
        self.Dempstd = np.sqrt(Dempstd) / (2.*self.ndim*self.dt)
        self.q_m = np.mean(self.q, axis=1)
        self.q_std = np.std(self.q, axis=1)

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
            tc = step * self.dt
            tc_disp = tc / self.time_scale
            
            print(f"Automatically selected tc = {tc_disp:.4g} {self.time_unit} (Q = {self.q_m[itc]:.4f})")
        else:
            itc = self._timestep_index(tc)
            tc_disp = tc / self.time_scale

        # store selected tc for later access
        self.tc_selected_idx = itc
        self.tc_selected = tc_disp

        with open(f'{self.fout}.dat','w') as g:
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
            g.write(f"Your chosen diffusion coefficient at {tc_disp} {self.time_unit}: {self.D[itc]:.4e} nm^2/ps\n")
            g.write(f"Standard deviation at {tc_disp} {self.time_unit}: {self.Dstd[itc]:.4e} nm^2/ps\n")
            g.write(f"Empirical std at {tc_disp} {self.time_unit}: {self.Dempstd[itc]:.4e} nm^2/ps\n")
            g.write(f"Q-factor at {tc_disp} {self.time_unit}: {self.q_m[itc]:.4f}\n")
            g.write(f"Your chosen diffusion coefficient at {tc_disp} {self.time_unit}: {self.D[itc]*0.01:.4e} cm^2/s\n")
            g.write("DIFFUSION COEFFICIENT OUTPUT SUMMARY:\n")
            g.write(f"t[{self.time_unit}] D[nm^2/ps] varD[nm^4/ps^2] Q D[cm^2/s] varD[cm^4/s^2]\n")
            for t,step in enumerate(range(self.tmin, self.tmax+1)):
                g.write(f"{(step*self.dt)/self.time_scale:.4g} {self.D[t]:.5g} {self.Dstd[t]**2:.5g} {self.q_m[t]:.5f} {self.D[t]*0.01:.5g} {(self.Dstd[t]**2)*0.0001:.5g}\n")
            if self.ndim > 1:
                g.write("\nDIFFUSION COEFFICIENT PER DIMENSION:\n")
                g.write(f"t[{self.time_unit}] Dx[nm^2/ps] Dy[nm^2/ps] ...\n")
                for step, Dt in zip(range(self.tmin, self.tmax+1), self.Dperdim):
                    g.write(f"{(step*self.dt)/self.time_scale:.4f} {Dt}\n")
        
        self.plot_results(tc)

    def plot_results(self, tc):
        sns.set_context("paper", font_scale=0.5)
        fig, ax = plt.subplots(2,1,figsize=(6,6),sharex=True)
        xs = np.arange(self.tmin*self.dt,(self.tmax+1)*self.dt,self.dt) / self.time_scale
        ax[0].plot(xs,self.D,color='C0',label=r'$D$')
        # ax[0].plot(xs,Dseg,color='tab:orange', label= 'mean D segm')
        ax[0].plot(xs,self.D-self.Dstd,color='black',linestyle='dotted', label=r'$\delta \overline{D}^\mathrm{predicted}$')
        ax[0].plot(xs,self.D+self.Dstd,color='black',linestyle='dotted')
        ax[0].fill_between(xs,self.D-self.Dempstd,self.D+self.Dempstd,color='C0',alpha=0.5, label = r'$\delta \overline{D}^\mathrm{empirical}$')
        ax[0].axvline(tc/self.time_scale,color='tab:red',linestyle='dashed')
        ax[0].set(ylabel=r'$D(t)$ [nm$^2$ ps$^{-1}$]')
        ax[0].set(xlim=(self.tmin*self.dt/self.time_scale,self.tmax*self.dt/self.time_scale))
        ax[0].ticklabel_format(style='scientific',scilimits=(-3,4))
        ax[0].legend(ncol=2)
        ax[0].set_title(f"MSD window per lag: t .. {self.m}×t [{self.time_unit}]")
        ax[1].plot(xs,self.q_m,color='C0')
        ax[1].fill_between(xs,self.q_m-self.q_std,self.q_m+self.q_std,color='C0',alpha=0.5)
        ax[1].axhline(0.5,linestyle='dashed',color='gray',linewidth=1.2)
        ax[1].axvline(tc/self.time_scale,color='tab:red',linestyle='dashed')
        ax[1].set(ylabel=r'$Q(t)$')
        ax[1].set(xlabel=fr'lag time $t$ [{self.time_unit}]')
        ax[1].set(ylim=(0,1))
        fig.tight_layout(h_pad=0.1)
        fig.savefig(f'{self.fout}.{self.imgfmt}',dpi=300)
        plt.close(fig)

    def finite_size_correction(self, T=300, eta=None, L=None, boxtype='cubic', tc=10):
        """ T in Kelvin, eta in Pa*s, L in nm"""
        itc = self._timestep_index(tc)
        if self.ndim != 3:
            raise ValueError("Currently only 3D correction implemented")
        if L is None:
            raise ValueError("Required parameter missing: L, box edge length")
        if eta is None:
            raise ValueError("Required parameter missing: eta, viscosity eta")
        if boxtype != 'cubic':
            raise ValueError("Sorry, correction only implemented for cubic simulation boxes")

        kbT = T * BOLTZMANN_K # J
        self.Dcor = self.D + kbT * XI_CUBIC * 1e15 / (6. * np.pi * eta * L) # nm^2 / ps
        print(f"Finite-size corrected diffusion coefficient D_t for timestep {tc} ps: {self.Dcor[itc]:.4g} nm^2/ps with standard dev. {self.Dstd[itc]:.4g} nm^2/ps")
