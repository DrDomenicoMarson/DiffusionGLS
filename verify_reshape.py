
import numpy as np
import os
from Dfit.trajectory_reader import NumpyTextReader

def test_loadtxt_behavior():
    print("--- Testing np.loadtxt behavior ---")
    # Create 1D file
    with open("test_1d.dat", "w") as f:
        f.write("1.0\n2.0\n3.0\n")
    
    data = np.loadtxt("test_1d.dat")
    print(f"1D file shape: {data.shape}")
    print(f"1D file ndim: {data.ndim}")
    
    # Create 3D file
    with open("test_3d.dat", "w") as f:
        f.write("1.0 0.0 0.0\n2.0 0.0 0.0\n3.0 0.0 0.0\n")
        
    data3 = np.loadtxt("test_3d.dat")
    print(f"3D file shape: {data3.shape}")

def test_mixed_dimensions():
    print("\n--- Testing Mixed Dimensions ---")
    # We already have test_3d.dat and test_1d.dat
    # Initialize reader with 3D file first (sets ndim=3)
    # Then try to iterate. The second file is 1D.
    
    reader = NumpyTextReader(["test_3d.dat", "test_1d.dat"])
    print(f"Reader ndim: {reader.ndim}")
    
    try:
        for i, traj in enumerate(reader):
            print(f"Traj {i} shape: {traj.shape}")
            # Simulate Dfit usage
            # Dfit expects (N, ndim)
            # If ndim=3, it accesses traj[:, 2]
            if reader.ndim == 3:
                _ = traj[:, 2]
                print(f"Traj {i} access OK")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught UNEXPECTED error: {e}")

    # Clean up
    if os.path.exists("test_1d.dat"): os.remove("test_1d.dat")
    if os.path.exists("test_3d.dat"): os.remove("test_3d.dat")

if __name__ == "__main__":
    test_loadtxt_behavior()
    test_mixed_dimensions()
