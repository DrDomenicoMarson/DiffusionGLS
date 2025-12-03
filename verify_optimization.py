
import MDAnalysis as mda
import numpy as np
import warnings
from Dfit.trajectory_reader import MDAnalysisReader

def create_mock_universe(n_residues=10, atoms_per_residue=1, n_frames=5):
    # Create a universe with n_residues, each having atoms_per_residue
    n_atoms = n_residues * atoms_per_residue
    u = mda.Universe.empty(n_atoms, n_residues=n_residues, atom_resindex=np.repeat(np.arange(n_residues), atoms_per_residue), trajectory=True)
    u.trajectory.new_chunk(n_frames, dt=1.0)
    
    # Set positions
    for ts in u.trajectory:
        u.atoms.positions = np.random.random((n_atoms, 3))
        
    return u

def test_optimization():
    print("--- Testing Single Atom Optimization ---")
    
    # Case 1: Single atom residues
    print("Creating Universe with single-atom residues...")
    u_single = create_mock_universe(n_residues=5, atoms_per_residue=1)
    reader_single = MDAnalysisReader(u_single)
    
    print("Iterating...")
    # We want to capture stdout to check for the optimization message
    # But for now let's just run it and see if it works
    for traj in reader_single:
        pass
    print("Done iterating single-atom residues.")

    # Case 2: Multi atom residues
    print("\nCreating Universe with multi-atom residues...")
    u_multi = create_mock_universe(n_residues=5, atoms_per_residue=2)
    reader_multi = MDAnalysisReader(u_multi)
    
    print("Iterating...")
    for traj in reader_multi:
        pass
    print("Done iterating multi-atom residues.")

if __name__ == "__main__":
    test_optimization()
