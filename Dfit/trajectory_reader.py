
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
    """Abstract interface for trajectory backends used by :class:`Dfit.Dcov`.

    Notes
    -----
    Concrete subclasses must expose ``n_frames``, ``ndim``, ``n_trajs``, and
    ``dt`` attributes and implement ``__iter__`` yielding arrays of shape
    ``(n_frames, ndim)`` in nm.
    """
    def __init__(self):
        """Initialize shared trajectory metadata defaults.

        Returns
        -------
        None
        """
        self.n_frames = 0
        self.ndim = 3
        self.n_trajs = 0
        self.dt = 1.0 # Default, can be overridden

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield trajectory arrays.

        Returns
        -------
        Iterator[ndarray]
            Iterator over arrays of shape ``(n_frames, ndim)``.

        Raises
        ------
        NotImplementedError
            Always raised in the abstract base class.
        """
        raise NotImplementedError

class NumpyTextReader(TrajectoryReader):
    """Read one or more trajectories from whitespace-delimited text files.

    Each file is expected to contain coordinate values in nm, one frame per
    line and one spatial dimension per column.
    """
    def __init__(self, fz: str | Path | Sequence[str | Path], normalize_lengths: bool = False):
        """Load metadata from text trajectory files.

        Parameters
        ----------
        fz : str or Path or sequence[str | Path]
            Input file path(s). A single file maps to one trajectory, while a
            sequence maps to multiple trajectories.
        normalize_lengths : bool, default=False
            If ``True`` and multiple files have different lengths, truncate all
            trajectories to the shortest one.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``fz`` is not a supported path type/sequence.
        ValueError
            If no files are provided, if dimensions mismatch across files, or
            if lengths mismatch when ``normalize_lengths=False``.
        """
        super().__init__()
        self.files = []
        self.lengths = []
        self.normalize_lengths = normalize_lengths
        self.normalized = False
        if isinstance(fz, (str, Path)):
            self.files = [fz]
        elif isinstance(fz, Sequence):
            self.files = list(fz)
        else:
            raise TypeError(f"Invalid input type for NumpyTextReader: {type(fz)}")
        
        if len(self.files) == 0:
            raise ValueError("No trajectory files provided. Pass at least one file path.")
        
        self.n_trajs = len(self.files)
        
        # Read first file to determine properties
        if self.n_trajs > 0:
            first_traj = np.loadtxt(self.files[0])
            # Handle 0-d scalar, 1D, and multi-D arrays from loadtxt
            first_traj = np.atleast_1d(first_traj)
            if first_traj.ndim == 1:
                self.n_frames = first_traj.shape[0]
                self.ndim = 1
            else:
                self.n_frames = first_traj.shape[0]
                self.ndim = first_traj.shape[1]
            self.lengths.append(self.n_frames)

            # Validate remaining files have matching length and dims
            for f in self.files[1:]:
                data = np.loadtxt(f)
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                if data.shape[1] != self.ndim:
                    raise ValueError(f"Trajectory file {f} has {data.shape[1]} dimensions, expected {self.ndim} based on first file.")
                self.lengths.append(data.shape[0])

            unique_lengths = set(self.lengths)
            if len(unique_lengths) > 1:
                if not self.normalize_lengths:
                    raise ValueError(f"All input trajectories must have the same length. Found lengths: {sorted(unique_lengths)}. "
                                     f"Set normalize_lengths=True to truncate to the shortest trajectory.")
                self.normalized = True
                min_len = min(self.lengths)
                warnings.warn(f"normalize_lengths=True: truncating all {self.n_trajs} trajectories to {min_len} frames "
                              f"(original lengths: {sorted(unique_lengths)}).", UserWarning)
                self.n_frames = min_len

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield trajectories loaded from text files.

        Returns
        -------
        Iterator[ndarray]
            Arrays with shape ``(n_frames, ndim)``. In 1D, files are reshaped
            to ``(n_frames, 1)``. If normalization is enabled, yielded arrays
            are truncated to the common shortest length.

        Raises
        ------
        ValueError
            If a file has a different number of dimensions than expected.
        """
        for f in self.files:
            data = np.loadtxt(f)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1) # Ensure (N, 1) for 1D, as np.loadtxt returns (N,)

            # Check consistency
            if data.shape[1] != self.ndim:
                raise ValueError(f"Trajectory file {f} has {data.shape[1]} dimensions, expected {self.ndim} based on first file.")
            
            if self.normalized:
                yield data[:self.n_frames]
            else:
                yield data

class MDAnalysisReader(TrajectoryReader):
    """Read trajectories from MDAnalysis objects as per-residue COM tracks.

    Coordinates are converted from Angstrom (MDAnalysis default) to nm by
    multiplying by 0.1.
    """
    def __init__(self, universe_or_group, selection_str: str = None):
        """Initialize a reader from an MDAnalysis Universe or AtomGroup.

        Parameters
        ----------
        universe_or_group : MDAnalysis.Universe or MDAnalysis.AtomGroup
            Input object defining topology and trajectory.
        selection_str : str, optional
            Selection expression used when ``universe_or_group`` is a Universe.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If MDAnalysis is not installed.
        TypeError
            If ``universe_or_group`` is not a Universe or AtomGroup.

        Notes
        -----
        Each residue in the final selection is treated as one trajectory.
        """
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
        """Yield per-residue trajectories over all frames.

        Returns
        -------
        Iterator[ndarray]
            One ``(n_frames, 3)`` array per residue trajectory in nm.

        Notes
        -----
        This implementation materializes an in-memory array with shape
        ``(n_trajs, n_frames, 3)``, which can be memory intensive for large
        systems.
        """
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
                trajs[:, i_frame, :] = self.ag.positions * 0.1 # Convert Angstrom to nm
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
                    trajs[:, i_frame, :] = coms * 0.1 # Convert Angstrom to nm
                except (TypeError, ValueError):
                    # Fallback for older MDA or if compound not supported
                    # This is slow
                    for i, res in enumerate(self.ag.residues):
                        trajs[i, i_frame, :] = res.atoms.center_of_mass() * 0.1 # Convert Angstrom to nm
        
        # Yield them
        for i in range(self.n_trajs):
            yield trajs[i]

def get_reader(fz=None, universe=None, selection=None, normalize_lengths: bool = False) -> TrajectoryReader:
    """Create a trajectory reader from text files or MDAnalysis input.

    Parameters
    ----------
    fz : str or Path or sequence[str | Path], optional
        Text trajectory input path(s).
    universe : MDAnalysis.Universe or MDAnalysis.AtomGroup, optional
        MDAnalysis input object.
    selection : str, optional
        MDAnalysis selection string. Ignored when ``fz`` is provided.
    normalize_lengths : bool, default=False
        Enable truncation to shortest length for multi-file text input.

    Returns
    -------
    TrajectoryReader
        Concrete reader instance matching the provided input mode.

    Raises
    ------
    ValueError
        If both input modes are provided, or if neither is provided.
    """
    if fz is not None and universe is not None:
        raise ValueError("Cannot provide both 'fz' and 'universe'.")
    if fz is not None and selection is not None:
        warnings.warn("'selection' is ignored when 'fz' is provided.")
    if universe is not None:
        if normalize_lengths:
            warnings.warn("'normalize_lengths' is ignored for MDAnalysis inputs.", UserWarning)
        return MDAnalysisReader(universe, selection)
    elif fz is not None:
        return NumpyTextReader(fz, normalize_lengths=normalize_lengths)
    else:
        raise ValueError("Must provide either 'fz' (files) or 'universe' (MDAnalysis).")
