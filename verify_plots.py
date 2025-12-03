import numpy as np
import os
from Dfit.Dfit import Dcov
import matplotlib.pyplot as plt

def verify_plots():
    # Generate random walk
    n_steps = 100
    traj = np.cumsum(np.random.randn(n_steps + 1, 3), axis=0)
    fname = "verify_plot_traj.dat"
    np.savetxt(fname, traj)
    fout = "verify_plot_out"
    
    try:
        # Run Dfit
        dcov = Dcov(fz=fname, m=2, tmax=10.0, dt=1.0, nseg=1, fout=fout)
        dcov.run_Dfit()
        dcov.analysis()
        
        # Check if plot files exist
        if os.path.exists(f"{fout}.pdf"):
            print("[OK] PDF plot generated.")
        else:
            print("[FAIL] PDF plot missing.")
            
        # We can't easily check the visual content, but successful generation implies
        # seaborn import and plotting commands worked.
        
    finally:
        if os.path.exists(fname):
            os.remove(fname)
        if os.path.exists(f"{fout}.dat"):
            os.remove(f"{fout}.dat")
        if os.path.exists(f"{fout}.pdf"):
            os.remove(f"{fout}.pdf")
        if os.path.exists(f"{fout}.png"):
            os.remove(f"{fout}.png")

if __name__ == "__main__":
    verify_plots()
