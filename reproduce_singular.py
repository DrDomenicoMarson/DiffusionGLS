import numpy as np
import os
from Dfit.Dfit import Dcov

def reproduce():
    # User parameters
    dt = 5.0 # ps
    total_time = 200000.0 # ps (200 ns)
    n_steps = int(total_time / dt)
    
    # Generate stationary trajectory (all zeros)
    traj = np.zeros((n_steps + 1, 3))
    
    # Save to file
    fname = "singular_traj.dat"
    np.savetxt(fname, traj)
    
    try:
        # tmin=20000, tmax=100000, m=2
        # nseg=1 to force single segment analysis (avoiding auto-calc error for short traj)
        dcov = Dcov(fz=fname, m=2, tmin=20000, tmax=100000, dt=dt, nseg=1, fout="singular_out")
        dcov.run_Dfit()
        print("Run completed successfully (unexpected).")
    except Exception as e:
        print(f"Caught expected exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(fname):
            os.remove(fname)

if __name__ == "__main__":
    reproduce()
