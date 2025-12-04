
import numpy as np
import cProfile
import pstats
from Dfit.Dfit import Dcov
import os

def profile_run():
    print("--- Profiling Dfit ---")
    
    # Create a decent sized trajectory
    # N=10000, nseg=100.
    N = 10000
    traj = np.random.normal(0, 1, size=(N, 3))
    np.savetxt("profile_traj.dat", traj)
    
    dcov = Dcov(fz="profile_traj.dat", m=20, tmax=50.0, dt=1.0, nseg=100)
    
    # Profile run_Dfit
    profiler = cProfile.Profile()
    profiler.enable()
    dcov.run_Dfit()
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
    
    # Clean up
    if os.path.exists("profile_traj.dat"): os.remove("profile_traj.dat")
    if os.path.exists("D_analysis.dat"): os.remove("D_analysis.dat")

if __name__ == "__main__":
    profile_run()
