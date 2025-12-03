
import numpy as np
import os
from Dfit.Dfit import Dcov

def reproduce_segment_error():
    print("--- Reproducing Segment Broadcast Error ---")
    
    # User scenario:
    # Trajectory 200 ns, dt=5 ps -> 40,000 frames.
    # tmax=60000 ps -> 12,000 steps.
    # m=5.
    # nseg=1 (default).
    # nperseg = 40000.
    # step = 12000.
    # len = 40000 / 12000 = 3.33 -> 3 or 4 frames.
    # m=5. 3 <= 5. Fail.
    
    N = 40000
    traj = np.random.normal(0, 1, size=(N, 3))
    np.savetxt("test_seg_error.dat", traj)
    
    dt = 5.0
    tmax = 60000.0
    m = 5
    
    try:
        # tmin needs to be set too, user said 20000
        # Add nseg=2 to trigger segment error specifically
        dcov = Dcov(fz="test_seg_error.dat", m=m, tmin=20000.0, tmax=tmax, dt=dt, nseg=2)
        dcov.run_Dfit()
        print("FAILURE: Did not raise ValueError")
    except ValueError as e:
        print(f"Caught expected error: {e}")
        if "could not broadcast" in str(e):
             print("FAILURE: Still getting broadcast error")
        elif "Segment too short" in str(e):
             print("SUCCESS: Caught 'Segment too short' error")
        else:
             print(f"Caught other ValueError: {e}")
    except Exception as e:
        print(f"Caught UNEXPECTED error: {e}")

    # Clean up
    if os.path.exists("test_seg_error.dat"): os.remove("test_seg_error.dat")
    if os.path.exists("D_analysis.dat"): os.remove("D_analysis.dat")

if __name__ == "__main__":
    reproduce_segment_error()
