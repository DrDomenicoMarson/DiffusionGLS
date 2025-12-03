
import numpy as np
from pathlib import Path
from collections.abc import Sequence, Iterator
import warnings

try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

class TrajectoryReader:
    """Base class/Interface for trajectory readers."""
    def __init__(self):
        self.n_frames = 0
        self.ndim = 3
        self.n_trajs = 0
        self.dt = 1.0 # Default, can be overridden

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yields trajectory arrays of shape (n_frames, ndim)."""
        raise NotImplementedError

class NumpyTextReader(TrajectoryReader):
    """Reads trajectories from text files (legacy format)."""
    def __init__(self, fz: str | Path | Sequence[str | Path]):
        super().__init__()
        self.files = []
        if isinstance(fz, (str, Path)):
            self.files = [fz]
        elif isinstance(fz, Sequence):
            self.files = list(fz)
        else:
            raise TypeError(f"Invalid input type for NumpyTextReader: {type(fz)}")
        
        self.n_trajs = len(self.files)
        
        # Read first file to determine properties
        if self.n_trajs > 0:
            first_traj = np.loadtxt(self.files[0])
            # Handle 1D vs Multi-D
            if len(first_traj.shape) == 1:
                self.n_frames = first_traj.shape[0]
                self.ndim = 1
            else:
                self.n_frames = first_traj.shape[0]
                self.ndim = first_traj.shape[1]

    def __iter__(self) -> Iterator[np.ndarray]:
        for f in self.files:
            data = np.loadtxt(f)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1) # Ensure (N, 1) for 1D, as np.loadtxt returns (N,)

            # Check consistency
            if data.shape[1] != self.ndim:
                raise ValueError(f"Trajectory file {f} has {data.shape[1]} dimensions, expected {self.ndim} based on first file.")
                
            yield data

class MDAnalysisReader(TrajectoryReader):
    """Reads trajectories using MDAnalysis."""
    def __init__(self, universe_or_group, selection_str: str = None):
        super().__init__()
        if not HAS_MDA:
            raise ImportError("MDAnalysis is not installed.")
        
        self.u = None
        self.ag = None
        
        # Determine if input is Universe or AtomGroup
        if isinstance(universe_or_group, mda.Universe):
            self.u = universe_or_group
            if selection_str:
                self.ag = self.u.select_atoms(selection_str)
            else:
                self.ag = self.u.atoms # Default to all atoms
        elif isinstance(universe_or_group, mda.AtomGroup):
            self.ag = universe_or_group
            self.u = self.ag.universe
        else:
             raise TypeError(f"Invalid input for MDAnalysisReader: {type(universe_or_group)}")

        self.n_frames = self.u.trajectory.n_frames
        self.dt = self.u.trajectory.dt # ps
        self.ndim = 3 # MD is always 3D
        
        # We treat each RESIDUE in the selection as a separate segment/molecule
        # This is a common assumption for diffusion (e.g. water, lipids, proteins)
        # If the selection is a single large molecule (like a protein), n_segments=1
        self.n_trajs = self.ag.n_residues
        
        if self.n_trajs == 0:
            warnings.warn("Selection contains 0 residues.")

    def __iter__(self) -> Iterator[np.ndarray]:
        # Pre-calculate center of mass for each residue for the whole trajectory
        # This can be memory intensive for huge systems. 
        # Optimization: Iterate frames and build trajectories per residue.
        
        # Strategy:
        # We need to yield (n_frames, 3) for each residue.
        # Since MDAnalysis iterates by FRAME, we have to pivot the data.
        # For memory efficiency with large trajectories, we might need a different approach,
        # but for now, let's load the COM trajectory for the selection.
        
        # 1. Create a list of arrays to hold data: [ (n_frames, 3), ... ]
        # This is still memory intensive. 
        # If n_segments * n_frames is too large, we are in trouble.
        # But Dfit needs the full time series for FFT.
        
        # Let's try a smarter way using MDAnalysis's built-in analysis if possible,
        # or just iterate frames and accumulate.
        
        trajs = np.zeros((self.n_trajs, self.n_frames, 3), dtype=np.float32)
        
        # Map residues to indices
        res_indices = {r.resindex: i for i, r in enumerate(self.ag.residues)}
        
        # Check for single-atom residues optimization
        if self.ag.n_atoms == self.n_trajs:
            print("Optimization: Single-atom residues detected. Using atom positions directly.")
            for i_frame, ts in enumerate(self.u.trajectory):
                trajs[:, i_frame, :] = self.ag.positions
        else:
            # Iterate trajectory once
            for i_frame, ts in enumerate(self.u.trajectory):
                # Calculate COM for each residue in the selection
                # This loop over residues might be slow in Python for 10k waters.
                # Faster: self.ag.center_of_mass(compound='residues')
                
                # Note: compound='residues' returns (n_residues, 3)
                # We need to make sure the order matches our iteration
                
                # MDAnalysis 2.0+ supports compound='residues'
                try:
                    coms = self.ag.center_of_mass(compound='residues')
                    trajs[:, i_frame, :] = coms
                except (TypeError, ValueError):
                    # Fallback for older MDA or if compound not supported
                    for i, res in enumerate(self.ag.residues):
                        trajs[i, i_frame, :] = res.atoms.center_of_mass()
        
        # Yield them
        for i in range(self.n_trajs):
            yield trajs[i]

def get_reader(fz=None, universe=None, selection=None) -> TrajectoryReader:
    """Factory function to get the appropriate reader."""
    if fz is not None and universe is not None:
        raise ValueError("Cannot provide both 'fz' and 'universe'.")
    if fz is not None and selection is not None:
        warnings.warn("'selection' is ignored when 'fz' is provided.")
    if universe is not None:
        return MDAnalysisReader(universe, selection)
    elif fz is not None:
        return NumpyTextReader(fz)
    else:
        raise ValueError("Must provide either 'fz' (files) or 'universe' (MDAnalysis).")
