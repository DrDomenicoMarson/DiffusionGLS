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

import matplotlib.pyplot as plt
import numpy as np
import progressbar as bar
from scipy.special import gammainc

from . import math_utils

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
    def __init__(self, fz: TrajectoryInput = None,
                 m: int = 20, tmin: int = 1, tmax: int = 100, dt: float = 1.0,
                 d2max: float = 1e-10, nitmax: int = 100,
                 nseg: int | None = None, imgfmt: str = 'pdf', fout: str = 'D_analysis'):

        self.fz = fz
        if isinstance(self.fz, Sequence) and not isinstance(self.fz, (str, bytes, Path)):
            self.multi = True
            print('Analyzing trajectories of multiple molecules from the same simulation.')
        else:
            self.multi = False
            print('Analyzing single trajectory.')

        self.dt = dt # Trajecotory timestep in ps
        self.m = m
        self.tmin = tmin
        self.tmax = tmax
        self.d2max = d2max
        self.nitmax = nitmax

        if imgfmt not in ['pdf','png']:
            raise TypeError("Error! Choose 'pdf' or 'png' as output format.")
        self.imgfmt = imgfmt
        self.fout = fout

        if self.multi:
            self.zs = [np.loadtxt(f).T for f in self.fz]
            self.z = self.zs[0]  # This is only to determine ndim and n
        else:
            self.z = np.loadtxt(self.fz).T # read in timeseries (rows) for each dimensions (columns)

        if len(self.z.shape) > 1: # 2D data or more
            self.ndim = self.z.shape[0] # number of dimensions
            self.n = self.z.shape[1] - 1 # length of timeseries N+1
        else:
            self.ndim = 1
            self.n = self.z.shape[0] - 1 # length of timeseries N+1
        print(f'N = {self.n}')
        print(f'ndim = {self.ndim}')
        total_points = self.n + 1
        if self.multi:
            self.nseg = len(self.fz) # number of individual molecules
            self.nperseg = self.n # all molecules from trajectory with same length
        else:
            auto_nseg = int(total_points / (100. * self.tmax)) # number of segments
            if nseg is None:
                if auto_nseg < 1:
                    raise ValueError('Timeseries too short! Reduce tmax or provide nseg >= 1')
                self.nseg = auto_nseg
            else:
                if nseg < 1:
                    raise ValueError('nseg must be at least 1')
                if auto_nseg > 0 and nseg > auto_nseg: # too many segments chosen
                    print(f"Warning, too many segments chosen, falling back to nseg = {auto_nseg}")
                    self.nseg = auto_nseg
                else:
                    self.nseg = nseg
            self.nperseg = int(total_points / self.nseg) - 1 # length of segment timeseries Nperseg+1
            if self.nperseg < 1:
                raise ValueError(f'nseg={self.nseg} yields segment length < 2 points; reduce nseg or use a longer trajectory')
            self.a2full = np.zeros((self.tmax-self.tmin+1,self.ndim)) # full trajectory, per dim
            self.s2full = np.zeros((self.tmax-self.tmin+1,self.ndim)) # full trajectory, per dim

        if self.m > self.nperseg:
            self.m = self.nperseg # force self.m to not be larger than Nperseg

        self.a2 = np.zeros((self.tmax-self.tmin+1,self.nseg,self.ndim))  # per segment and dims
        self.s2 = np.zeros((self.tmax-self.tmin+1,self.nseg,self.ndim))  # per segment and dims

        self.s2var = np.zeros((self.tmax-self.tmin+1)) # mean across all segments and dims
        self.q = np.zeros((self.tmax-self.tmin+1,self.nseg)) # per segment, mean across dims

        # initialize som stuff that will be computed within the analysis method
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
        """ Main Function to calculate the stepsize sigma^2 and offset a^2 of a 
            random walk with noise.
        """
        with bar.ProgressBar(max_value=self.tmax-self.tmin+1) as progbar:
            for t,step in enumerate(range(self.tmin,self.tmax+1)):
                # Full trajectory s2, a2
                if not self.multi:
                    for d in range(self.ndim):
                        if self.ndim == 1:
                            z = self.z[::step] # full traj
                        else:
                            z = self.z[d,::step] # full traj
                        n = len(z) - 1
                        msd = math_utils.compute_MSD_1D_via_correlation(z)[1:(self.m+1)]
                        self.a2full[t,d], self.s2full[t,d] = math_utils.calc_gls(n,self.m,msd,self.d2max,self.nitmax)

                    a2full_3D = np.sum(self.a2full[t]) # sum across dims
                    s2full_3D = np.sum(self.s2full[t]) # sum across dims

                # segments
                n = int(self.nperseg / step) # per segment for given timestep
                # print('length N per seg = {}'.format(n))
                for s in range(self.nseg):
                    msds = np.zeros((self.ndim,self.m)) # dimensions
                    if self.multi: 
                        self.z = self.zs[s]
                        zstart = None
                        zend = None
                    else:
                        zstart = s * (self.nperseg+1)
                        zend = (s+1) * (self.nperseg+1)
                    for d in range(self.ndim):
                        if self.ndim == 1:
                            z = self.z[zstart:zend:step] # copy segment from trajectory
                        else:
                            z = self.z[d,zstart:zend:step] # copy segment from trajectory
                        msds[d] = math_utils.compute_MSD_1D_via_correlation(z)[1:(self.m+1)]
                        self.a2[t,s,d], self.s2[t,s,d] = math_utils.calc_gls(n,self.m,msds[d],self.d2max,self.nitmax)
                    msds_3D = np.sum(msds,axis=0) # --> MSD_3D per deltaT and segment
                    a2_3D = np.sum(self.a2[t,s]) # sum across dims
                    s2_3D = np.sum(self.s2[t,s]) # sum across dims

                    if self.multi:
                        self.q[t,s] = calc_q(n,self.m,a2_3D,s2_3D,msds_3D,a2_3D,s2_3D,self.ndim) # no 'full' trajectory to use
                    else:
                        self.q[t,s] = calc_q(n,self.m,a2_3D,s2_3D,msds_3D,a2full_3D,s2full_3D,self.ndim) # use a2full_3D and s2full_3D from 'full' trajectory for cinv

                a2m = np.mean(self.a2[t],axis=0) # mean across segments, per dim
                s2m = np.mean(self.s2[t],axis=0) # mean across segments, per dim

                self.s2var[t] = math_utils.eval_vars(n,self.m,a2m,s2m,self.ndim) # a2 and s2 are mean over segments, but still per dim (and for given timestp)
            
                self.a2[t] /= step
                self.s2[t] /= step
                self.s2var[t] /= step**2
                if not self.multi:
                    self.a2full[t] /= step
                    self.s2full[t] /= step
                progbar.update(t)

    # Output and plotting
    def analysis(self,tc=10):
        itc = self._timestep_index(tc)
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

        with open(f'{self.fout}.dat','w') as g:
            g.write("DIFFUSION COEFFICIENT ESTIMATE\n")
            g.write("INPUT:\n")
            g.write("Trajectory: {}\n".format(self.fz))
            g.write("Number of dimensions : {}\n".format(self.ndim))
            g.write("Min/max timestep: {}/{}\n".format(self.tmin,self.tmax))
            g.write("Number of segments: {}\n".format(self.nseg))
            g.write("Total number of trajectory data points per dim.: {}\n".format(self.n+1))
            g.write("Data points per segment and dim.: {}\n".format(self.nperseg+1))

            g.write("Your chosen diffusion coefficient at {} ps: {} nm^2/ps\n".format(tc,self.D[itc]))
            g.write("DIFFUSION COEFFICIENT OUTPUT SUMMARY:\n")
            g.write("t[ps] D[nm^2/ps] varD[nm^4/ps^2] Q\n")
            for t,step in enumerate(range(self.tmin, self.tmax+1)):
                g.write("{:.4g} {:.5g} {:.5g} {:.5f}\n".format(step*self.dt,self.D[t],self.Dstd[t]**2,self.q_m[t]))
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
