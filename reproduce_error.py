
import numpy as np
import os
from Dfit.Dfit import Dcov

def reproduce_broadcast_error():
    print("--- Reproducing Broadcast Error ---")
    
    # Create a trajectory
    # We need len(z_strided) < m + 1
    # Let m=20. We need len < 21.
    # If step=1, N=20.
    # Let's use N=40, m=20.
    # If step=2, len = 20. Fail.
    
    N = 25
    traj = np.random.normal(0, 1, size=(N, 3))
    np.savetxt("test_error.dat", traj)
    
    # tmax=1.0 (1 step).
    
    try:
        # Reduce nitmax to force non-convergence
        dcov = Dcov(fz="test_error.dat", m=20, tmax=1.0, dt=1.0, nitmax=1)
        dcov.run_Dfit()
        print("Run completed.")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught UNEXPECTED error: {e}")

    # Clean up
    if os.path.exists("test_error.dat"): os.remove("test_error.dat")
    if os.path.exists("D_analysis.dat"): os.remove("D_analysis.dat")

if __name__ == "__main__":
    reproduce_broadcast_error()
