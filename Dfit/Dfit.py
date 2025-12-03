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

import matplotlib.pyplot as plt
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

class Dcov():
    def __init__(self, fz: TrajectoryInput = None, universe=None, selection=None,
                 m: int = 20, tmin: float = None, tmax: float = 100.0, dt: float = 1.0,
                 d2max: float = 1e-10, nitmax: int = 100,
                 nseg: int | None = None, imgfmt: str = 'pdf', fout: str = 'D_analysis'):

        # Initialize Reader
        self.reader: TrajectoryReader = get_reader(fz=fz, universe=universe, selection=selection)
        
        # Use reader properties
        self.ndim = self.reader.ndim
        self.n = self.reader.n_frames - 1 # N is steps, n_frames is points
        
        self.dt = dt 
        if not math.isclose(self.dt, self.reader.dt, rel_tol=1e-5):
             warnings.warn(f"User provided dt ({self.dt}) differs from reader's dt ({self.reader.dt}). Using provided dt.", UserWarning)

        self.m = m
        
        # Convert tmin/tmax from ps to steps
        if tmin is None:
            self.tmin = 1
        else:
            self.tmin = int(round(tmin / self.dt))
            if self.tmin < 1:
                self.tmin = 1
                
        if tmax is None:
            self.tmax = int(round(100.0 / self.dt)) # Default 100 ps
        else:
            self.tmax = int(round(tmax / self.dt))
            
        self.d2max = d2max
        self.nitmax = nitmax

        if imgfmt not in ['pdf','png']:
            raise TypeError("Error! Choose 'pdf' or 'png' as output format.")
        self.imgfmt = imgfmt
        self.fout = fout

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

    def _timestep_index(self, tc: float) -> int:
        steps = tc / self.dt
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
        
        non_converged_count = 0
        
        with bar.ProgressBar(max_value=self.tmax-self.tmin+1) as progbar:
            for t, step in enumerate(range(self.tmin, self.tmax+1)):
                
                # 1. Full Trajectory Analysis (Only for Single Trajectory mode)
                if not self.multi:
                    # In single mode, all_trajs has 1 element
                    full_z = all_trajs[0]
                    for d in range(self.ndim):
                        z_dim = full_z[:, d]
                        z_strided = z_dim[::step]
                        if len(z_strided) <= self.m:
                             raise ValueError(f"Trajectory too short (length N={len(z_strided)}) to calculate m={self.m} MSD points at lag time step={step} (t={step*self.dt} ps). Please reduce tmax or m, or use a longer trajectory.")
                        n = len(z_strided) - 1
                        msd = math_utils.compute_MSD_1D_via_correlation(z_strided)[1:(self.m+1)]
                        self.a2full[t,d], self.s2full[t,d], converged = math_utils.calc_gls(n, self.m, msd, self.d2max, self.nitmax)
                        if not converged: non_converged_count += 1
                    
                    a2full_3D = np.sum(self.a2full[t])
                    s2full_3D = np.sum(self.s2full[t])
                
                # 2. Segment Analysis
                # If multi: iterate over molecules
                # If single: iterate over segments of the single trajectory
                
                n_per_seg_step = int(self.nperseg / step)
                
                for s in range(self.nseg):
                    msds = np.zeros((self.ndim, self.m))
                    
                    # Get the segment data
                    if self.multi:
                        # Segment s is molecule s
                        seg_z = all_trajs[s]
                        # Stride it
                        z_analyzed = seg_z[::step, :]
                    else:
                        # Segment s is a slice of the single trajectory
                        full_z = all_trajs[0]
                        zstart = s * (self.nperseg + 1)
                        zend = (s + 1) * (self.nperseg + 1)
                        z_analyzed = full_z[zstart:zend:step, :]

                    # Analyze per dimension
                    for d in range(self.ndim):
                        z_dim = z_analyzed[:, d]
                        msds[d] = math_utils.compute_MSD_1D_via_correlation(z_dim)[1:(self.m+1)]
                        self.a2[t,s,d], self.s2[t,s,d], converged = math_utils.calc_gls(n_per_seg_step, self.m, msds[d], self.d2max, self.nitmax)
                        if not converged: non_converged_count += 1
                    
                    msds_3D = np.sum(msds, axis=0)
                    a2_3D = np.sum(self.a2[t,s])
                    s2_3D = np.sum(self.s2[t,s])
                    
                    if self.multi:
                        self.q[t,s] = calc_q(n_per_seg_step, self.m, a2_3D, s2_3D, msds_3D, a2_3D, s2_3D, self.ndim)
                    else:
                        self.q[t,s] = calc_q(n_per_seg_step, self.m, a2_3D, s2_3D, msds_3D, a2full_3D, s2full_3D, self.ndim)

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
                
                progbar.update(t)
        
        if non_converged_count > 0:
            print(f"WARNING: Optimizer did not converge in {non_converged_count} cases. Falling back to M=2 for those cases.")

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
            
            print(f"Automatically selected tc = {tc:.4g} ps (Q = {self.q_m[itc]:.4f})")
        else:
            itc = self._timestep_index(tc)

        with open(f'{self.fout}.dat','w') as g:
            g.write("DIFFUSION COEFFICIENT ESTIMATE\n")
            g.write("INPUT:\n")
            # g.write("Trajectory: {}\n".format(self.fz)) # fz might be None now
            g.write("Number of dimensions : {}\n".format(self.ndim))
            g.write("Min/max timestep: {}/{}\n".format(self.tmin,self.tmax))
            g.write("Number of segments: {}\n".format(self.nseg))
            g.write("Total number of trajectory data points per dim.: {}\n".format(self.n+1))
            g.write("Data points per segment and dim.: {}\n".format(self.nperseg+1))

            g.write("Your chosen diffusion coefficient at {} ps: {:.4e} nm^2/ps\n".format(tc, self.D[itc]))
            g.write("Your chosen diffusion coefficient at {} ps: {:.4e} cm^2/s\n".format(tc, self.D[itc]*0.01))
            g.write("DIFFUSION COEFFICIENT OUTPUT SUMMARY:\n")
            g.write("t[ps] D[nm^2/ps] varD[nm^4/ps^2] Q D[cm^2/s] varD[cm^4/s^2]\n")
            for t,step in enumerate(range(self.tmin, self.tmax+1)):
                g.write("{:.4g} {:.5g} {:.5g} {:.5f} {:.5g} {:.5g}\n".format(
                    step*self.dt,
                    self.D[t],
                    self.Dstd[t]**2,
                    self.q_m[t],
                    self.D[t]*0.01,
                    (self.Dstd[t]**2)*0.0001
                ))
            if self.ndim > 1:
                g.write("\nDIFFUSION COEFFICIENT PER DIMENSION:\n")
                g.write("TIMESTEP Dx[nm^2/ps] Dy[nm^2/ps] ...\n")
                for t, Dt in zip( (range(self.tmin,self.tmax+1)), self.Dperdim):
                    g.write("{:.4f} {}\n".format(t, Dt))
        
        self.plot_results(tc)

    def plot_results(self, tc):
        fig, ax = plt.subplots(2,1,figsize=(6,6),sharex=True)
        xs = np.arange(self.tmin*self.dt,(self.tmax+1)*self.dt,self.dt)
        ax[0].plot(xs,self.D,color='C0',label=r'$D$')
        # ax[0].plot(xs,Dseg,color='tab:orange', label= 'mean D segm')
        ax[0].plot(xs,self.D-self.Dstd,color='black',linestyle='dotted', label=r'$\delta \overline{D}^\mathrm{predicted}$')
        ax[0].plot(xs,self.D+self.Dstd,color='black',linestyle='dotted')
        ax[0].fill_between(xs,self.D-self.Dempstd,self.D+self.Dempstd,color='C0',alpha=0.5, label = r'$\delta \overline{D}^\mathrm{empirical}$')
        ax[0].axvline(tc,color='tab:red',linestyle='dashed')
        ax[0].set(ylabel='diff. coeff. $D$ [nm$^2$ ps$^{-1}$]')
        ax[0].set(xlim=(self.tmin*self.dt,self.tmax*self.dt))
        ax[0].ticklabel_format(style='scientific',scilimits=(-3,4))
        ax[0].legend(ncol=2)
        ax[1].plot(xs,self.q_m,color='C0')
        ax[1].fill_between(xs,self.q_m-self.q_std,self.q_m+self.q_std,color='C0',alpha=0.5)
        ax[1].axhline(0.5,linestyle='dashed',color='gray',linewidth=1.2)
        ax[1].axvline(tc,color='tab:red',linestyle='dashed')
        ax[1].set(ylabel='quality factor Q')
        ax[1].set(xlabel='time step size [ps]')
        ax[1].set(ylim=(0,1))
        fig.tight_layout(h_pad=0.1)
        fig.savefig('{}.{}'.format(self.fout,self.imgfmt),dpi=300)
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
        print("Finite-size corrected diffusion coefficient D_t for timestep {} ps: {:.4g} nm^2/ps with standard dev. {:.4g} nm^2/ps".format(tc,self.Dcor[itc],self.Dstd[itc]))
