
import numpy as np
import os
from Dfit.Dfit import Dcov

def test_output_units():
    print("--- Testing Output Units ---")
    
    # Create a dummy trajectory
    traj = np.random.normal(0, 1, size=(1000, 3))
    np.savetxt("test_units.dat", traj)
    
    # Run Dfit
    dcov = Dcov(fz="test_units.dat", m=10, tmax=5, fout="test_units_out")
    dcov.run_Dfit()
    dcov.analysis(tc=2)
    
    # Check output file
    with open("test_units_out.dat", "r") as f:
        lines = f.readlines()
        
    # Find header line
    header_idx = -1
    for i, line in enumerate(lines):
        if "t[ps] D[nm^2/ps]" in line:
            header_idx = i
            break
            
    if header_idx == -1:
        print("FAILURE: Header line not found.")
        return
        
    print(f"Header found: {lines[header_idx].strip()}")
    
    # Check data lines
    data_lines = lines[header_idx+1:]
    for line in data_lines:
        if not line.strip(): continue
        if "DIFFUSION COEFFICIENT PER DIMENSION" in line: break
        
        parts = line.split()
        if len(parts) < 6:
            print(f"FAILURE: Not enough columns in line: {line.strip()}")
            continue
            
        d_nm = float(parts[1])
        var_nm = float(parts[2])
        d_cm = float(parts[4])
        var_cm = float(parts[5])
        
        # Verify conversion
        if not np.isclose(d_cm, d_nm * 0.01, rtol=1e-4):
             print(f"FAILURE: D conversion mismatch: {d_nm} -> {d_cm}")
        if not np.isclose(var_cm, var_nm * 0.0001, rtol=1e-4):
             print(f"FAILURE: varD conversion mismatch: {var_nm} -> {var_cm}")
             
    print("SUCCESS: Output units verified.")

    # Clean up
    if os.path.exists("test_units.dat"): os.remove("test_units.dat")
    if os.path.exists("test_units_out.dat"): os.remove("test_units_out.dat")
    if os.path.exists("test_units_out.pdf"): os.remove("test_units_out.pdf")

if __name__ == "__main__":
    test_output_units()
