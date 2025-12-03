
import Dfit
import MDAnalysis as mda
import os
import matplotlib.pyplot as plt

def analyze_system(name, tpr, xtc, selection, tmax=100, tc=10):
    print(f"--- Analyzing {name} ---")
    print(f"Topology: {tpr}")
    print(f"Trajectory: {xtc}")
    
    # Load Universe
    u = mda.Universe(tpr, xtc)
    print(f"System: {len(u.atoms)} atoms")
    
    # Initialize Dfit with MDAnalysis Universe
    # We use the selection to define what constitutes a "molecule" (segment)
    # The reader iterates over RESIDUES in the selection.
    res = Dfit.Dcov(universe=u, selection=selection, tmax=tmax, fout=f"D_analysis_{name}")
    
    # Run the fit
    res.run_Dfit()
    
    # Analyze and plot
    res.analysis(tc=tc)
    print(f"Analysis complete. Output saved to D_analysis_{name}.dat and .pdf\n")

if __name__ == "__main__":
    # Define paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    traj_dir = os.path.join(base_dir, "trajs")
    
    # 1. Analyze Water
    # Usually water residues are named SOL, WAT, or TIP3. 
    # Since we want to analyze diffusion of whole molecules, we select them.
    # The Dfit MDAnalysisReader treats each RESIDUE in the selection as a molecule.
    analyze_system(
        name="Water",
        tpr=os.path.join(traj_dir, "WAT_32.tpr"),
        xtc=os.path.join(traj_dir, "traj_WAT.xtc"),
        selection="all", # Assumes the box is pure water
        tmax=50, # Adjust based on trajectory length
        tc=5
    )

    # 2. Analyze Oxygen
    analyze_system(
        name="Oxygen",
        tpr=os.path.join(traj_dir, "O2_32.tpr"),
        xtc=os.path.join(traj_dir, "traj_O2.xtc"),
        selection="all", # Assumes the box is pure oxygen
        tmax=50,
        tc=5
    )
