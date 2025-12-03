
import numpy as np
import time
from Dfit.Dfit import Dcov
import os

def benchmark():
    print("--- Benchmarking Parallelization ---")
    
    # Create multiple files to simulate molecules
    # 500 molecules, each 2000 steps.
    n_mols = 500
    n_steps = 2000
    files = []
    for i in range(n_mols):
        fname = f"bench_traj_{i}.dat"
        traj = np.random.normal(0, 1, size=(n_steps, 3))
        np.savetxt(fname, traj)
        files.append(fname)
    
    # Serial Run (n_jobs=1)
    start = time.time()
    dcov = Dcov(fz=files, m=20, tmax=50.0, dt=1.0, n_jobs=1)
    dcov.run_Dfit()
    end = time.time()
    print(f"Serial (n_jobs=1): {end - start:.4f} s")
    
    # Parallel Run (n_jobs=6)
    start = time.time()
    dcov = Dcov(fz=files, m=20, tmax=50.0, dt=1.0, n_jobs=6)
    dcov.run_Dfit()
    end = time.time()
    print(f"Parallel (n_jobs=6): {end - start:.4f} s")
    
    # Clean up
    for fname in files:
        if os.path.exists(fname): os.remove(fname)
    if os.path.exists("bench_list.dat"): os.remove("bench_list.dat")

if __name__ == "__main__":
    benchmark()
