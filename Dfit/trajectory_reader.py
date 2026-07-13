
import numpy as np
from pathlib import Path
from collections.abc import Sequence, Iterator
from dataclasses import dataclass
import math
import warnings

try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False


@dataclass(frozen=True)
class MDATrajectorySource:
    """Container for one MDAnalysis trajectory source and its selection.

    Parameters
    ----------
    universe : object
        MDAnalysis universe associated with the selected atoms.
    atomgroup : object
        Atom selection used to define residue trajectories.
    n_residues : int
        Number of residues contributing trajectories from this source.
    cluster_id : str
        Cluster identifier inherited by every residue trajectory from this
        source.
    """
    universe: object
    atomgroup: object
    n_residues: int
    cluster_id: str


def _normalize_cluster_ids(
    cluster_ids,
    expected_length: int,
    *,
    default_ids: Sequence[str],
) -> tuple[str, ...]:
    """Validate and normalize user-provided cluster identifiers.

    Parameters
    ----------
    cluster_ids : sequence[object] or None
        Optional identifiers aligned with input trajectory sources. Repeated
        values intentionally combine sources into one cluster.
    expected_length : int
        Required number of identifiers.
    default_ids : sequence[str]
        Identifiers to use when ``cluster_ids`` is ``None``.

    Returns
    -------
    tuple[str, ...]
        Normalized string identifiers preserving input order.

    Raises
    ------
    TypeError
        If ``cluster_ids`` is a string or is not a sequence.
    ValueError
        If its length does not match ``expected_length`` or an identifier is
        empty.
    """
    if cluster_ids is None:
        values = list(default_ids)
    else:
        if isinstance(cluster_ids, (str, bytes)) or not isinstance(cluster_ids, Sequence):
            raise TypeError("cluster_ids must be a non-string sequence aligned with the inputs.")
        values = list(cluster_ids)

    if len(values) != expected_length:
        raise ValueError(
            f"cluster_ids must contain {expected_length} value(s), got {len(values)}."
        )

    normalized = tuple(str(value) for value in values)
    if any(not value.strip() for value in normalized):
        raise ValueError("cluster_ids cannot contain empty identifiers.")
    return normalized

class TrajectoryReader:
    """Abstract interface for trajectory backends used by :class:`Dfit.Dcov`.

    Notes
    -----
    Concrete subclasses must expose ``n_frames``, ``ndim``, ``n_trajs``, and
    ``dt`` attributes, plus one ``trajectory_cluster_ids`` entry per yielded
    trajectory, and implement ``__iter__`` yielding arrays of shape
    ``(n_frames, ndim)`` in nm. ``dt`` is ``None`` when the source has no time
    metadata.
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
        self.dt = None
        self.trajectory_cluster_ids: tuple[str, ...] = ()

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


def _resolve_mda_source(universe_or_group, selection_str: str | None):
    """Resolve a Universe/AtomGroup input into a universe and atom selection.

    Parameters
    ----------
    universe_or_group : MDAnalysis.Universe or MDAnalysis.AtomGroup
        Input object defining topology and trajectory.
    selection_str : str or None
        Selection expression applied only when input is a Universe.

    Returns
    -------
    tuple[object, object]
        ``(universe, atomgroup)`` pair to analyze.

    Raises
    ------
    TypeError
        If ``universe_or_group`` is neither Universe nor AtomGroup.
    """
    if isinstance(universe_or_group, mda.Universe):
        universe = universe_or_group
        if selection_str:
            atomgroup = universe.select_atoms(selection_str)
        else:
            atomgroup = universe.atoms
    elif isinstance(universe_or_group, mda.AtomGroup):
        atomgroup = universe_or_group
        universe = atomgroup.universe
    else:
        raise TypeError(f"Invalid MDAnalysis input type: {type(universe_or_group)}")
    return universe, atomgroup


def _compute_residue_com_trajectories(
    universe,
    atomgroup,
    n_frames: int,
    n_trajs: int,
) -> np.ndarray:
    """Compute per-residue COM trajectories for one MDAnalysis selection.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe providing frame iteration.
    atomgroup : MDAnalysis.AtomGroup
        Selected atoms used to compute per-residue COM trajectories.
    n_frames : int
        Number of trajectory frames.
    n_trajs : int
        Number of residue trajectories to produce.

    Returns
    -------
    ndarray
        Array of shape ``(n_trajs, n_frames, 3)`` in nm.
    """
    trajs = np.zeros((n_trajs, n_frames, 3), dtype=np.float32)

    # Optimization for 1-atom residues (e.g., monoatomic molecules):
    # per-residue COM is exactly atom positions.
    if atomgroup.n_atoms == n_trajs:
        print("Optimization: Single-atom residues detected. Using atom positions directly.")
        for i_frame, _ in enumerate(universe.trajectory):
            trajs[:, i_frame, :] = atomgroup.positions * 0.1  # Angstrom -> nm
        return trajs

    for i_frame, _ in enumerate(universe.trajectory):
        try:
            coms = atomgroup.center_of_mass(compound='residues')
            trajs[:, i_frame, :] = coms * 0.1  # Angstrom -> nm
        except (TypeError, ValueError):
            # Compatibility fallback for older MDAnalysis APIs.
            for i, res in enumerate(atomgroup.residues):
                trajs[i, i_frame, :] = res.atoms.center_of_mass() * 0.1  # Angstrom -> nm
    return trajs

class NumpyTextReader(TrajectoryReader):
    """Read one or more trajectories from whitespace-delimited text files.

    Each file is expected to contain coordinate values in nm, one frame per
    line and one spatial dimension per column.
    """
    def __init__(
        self,
        fz: str | Path | Sequence[str | Path],
        normalize_lengths: bool = False,
        cluster_ids=None,
    ):
        """Load metadata from text trajectory files.

        Parameters
        ----------
        fz : str or Path or sequence[str | Path]
            Input file path(s). A single file maps to one trajectory, while a
            sequence maps to multiple trajectories.
        normalize_lengths : bool, default=False
            If ``True`` and multiple files have different lengths, truncate all
            trajectories to the shortest one.
        cluster_ids : sequence[object] or None, optional
            Cluster identifier for each input file. Repeated identifiers group
            files into one cluster. When omitted, all text files belong to one
            cluster.

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
        self.trajectory_cluster_ids = _normalize_cluster_ids(
            cluster_ids,
            self.n_trajs,
            default_ids=["cluster_0"] * self.n_trajs,
        )
        
        # Read first file to determine properties
        if self.n_trajs > 0:
            first_traj = np.loadtxt(self.files[0], ndmin=2)
            self.n_frames = first_traj.shape[0]
            self.ndim = first_traj.shape[1]
            self.lengths.append(self.n_frames)

            # Validate remaining files have matching length and dims
            for f in self.files[1:]:
                data = np.loadtxt(f, ndmin=2)
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
            data = np.loadtxt(f, ndmin=2)

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
    def __init__(self, universe_or_group, selection_str: str = None, cluster_ids=None):
        """Initialize a reader from an MDAnalysis Universe or AtomGroup.

        Parameters
        ----------
        universe_or_group : MDAnalysis.Universe or MDAnalysis.AtomGroup
            Input object defining topology and trajectory.
        selection_str : str, optional
            Selection expression used when ``universe_or_group`` is a Universe.
        cluster_ids : sequence[object] or None, optional
            One optional identifier for this MDAnalysis source.

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

        self.u, self.ag = _resolve_mda_source(universe_or_group, selection_str)

        self.n_frames = self.u.trajectory.n_frames
        self.dt = self.u.trajectory.dt # ps
        self.ndim = 3 # MD is always 3D
        
        # We treat each RESIDUE in the selection as a separate segment/molecule
        # This is a common assumption for diffusion (e.g. water, lipids, proteins)
        # If the selection is a single large molecule (like a protein), n_segments=1
        self.n_trajs = self.ag.n_residues
        source_cluster_id = _normalize_cluster_ids(
            cluster_ids,
            1,
            default_ids=["cluster_0"],
        )[0]
        self.trajectory_cluster_ids = (source_cluster_id,) * self.n_trajs

        if self.n_trajs == 0:
            raise ValueError("Selection contains 0 residues; choose a non-empty MDAnalysis selection.")

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
        trajs = _compute_residue_com_trajectories(self.u, self.ag, self.n_frames, self.n_trajs)

        for i in range(self.n_trajs):
            yield trajs[i]


class MDAnalysisMultiReader(TrajectoryReader):
    """Pool multiple MDAnalysis trajectories as one multi-molecule dataset.

    Each input Universe/AtomGroup contributes residue COM trajectories. All
    inputs must share the same frame count and timestep.
    """
    def __init__(
        self,
        universes_or_groups: Sequence,
        selection_str: str | list[str] | None = None,
        cluster_ids=None,
    ):
        """Initialize pooled MDAnalysis reader from multiple inputs.

        Parameters
        ----------
        universes_or_groups : sequence[MDAnalysis.Universe | MDAnalysis.AtomGroup]
            Inputs to pool.
        selection_str : str or list[str] or None, optional
            Selection expression(s). If a single string, the same selection is
            applied to every Universe input. If a list, it must have the same
            length as ``universes_or_groups`` and each entry is applied to the
            corresponding input. AtomGroup inputs ignore the selection string.
        cluster_ids : sequence[object] or None, optional
            Identifier for each Universe/AtomGroup source. Repeated identifiers
            combine sources into one cluster. By default, every source is a
            separate cluster.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If MDAnalysis is not installed.
        TypeError
            If ``universes_or_groups`` is not a non-string sequence or contains
            unsupported entries.
        ValueError
            If the sequence is empty, if a selection list length does not match
            ``universes_or_groups``, or if inputs have mismatched frame count
            or timestep.
        """
        super().__init__()
        if not HAS_MDA:
            raise ImportError("MDAnalysis is not installed.")

        if isinstance(universes_or_groups, (str, Path)) or not isinstance(universes_or_groups, Sequence):
            raise TypeError(
                "universes must be a non-string sequence of MDAnalysis Universe/AtomGroup objects."
            )

        inputs = list(universes_or_groups)
        if len(inputs) == 0:
            raise ValueError("No MDAnalysis universes provided. Pass at least one Universe/AtomGroup.")

        source_cluster_ids = _normalize_cluster_ids(
            cluster_ids,
            len(inputs),
            default_ids=[f"cluster_{i}" for i in range(len(inputs))],
        )

        # Resolve selection_str into a per-input list.
        if isinstance(selection_str, list):
            if len(selection_str) != len(inputs):
                raise ValueError(
                    f"When 'selection' is a list it must have the same length as 'universes'. "
                    f"Got {len(selection_str)} selection(s) for {len(inputs)} universe(s)."
                )
            selections = selection_str
        else:
            # Single string (or None) – broadcast to all inputs.
            selections = [selection_str] * len(inputs)

        self.sources: list[MDATrajectorySource] = []
        ref_n_frames = None
        ref_dt = None

        for i, item in enumerate(inputs):
            universe, atomgroup = _resolve_mda_source(item, selections[i])
            n_frames = universe.trajectory.n_frames
            dt = universe.trajectory.dt

            if ref_n_frames is None:
                ref_n_frames = n_frames
            elif n_frames != ref_n_frames:
                raise ValueError(
                    f"All MDAnalysis universes must have the same number of frames. "
                    f"Found {n_frames} at index {i}, expected {ref_n_frames}."
                )

            if ref_dt is None:
                ref_dt = dt
            elif not math.isclose(dt, ref_dt, rel_tol=1e-5, abs_tol=1e-12):
                raise ValueError(
                    f"All MDAnalysis universes must have the same dt (ps). "
                    f"Found {dt} at index {i}, expected {ref_dt}."
                )

            n_residues = atomgroup.n_residues
            if n_residues == 0:
                raise ValueError(
                    f"Selection contains 0 residues for MDAnalysis input index {i}; "
                    "choose a non-empty selection."
                )

            self.sources.append(
                MDATrajectorySource(
                    universe=universe,
                    atomgroup=atomgroup,
                    n_residues=n_residues,
                    cluster_id=source_cluster_ids[i],
                )
            )

        self.n_frames = int(ref_n_frames)
        self.dt = float(ref_dt)
        self.ndim = 3
        self.n_trajs = int(sum(source.n_residues for source in self.sources))
        self.trajectory_cluster_ids = tuple(
            source.cluster_id
            for source in self.sources
            for _ in range(source.n_residues)
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield pooled per-residue trajectories over all input universes.

        Returns
        -------
        Iterator[ndarray]
            One ``(n_frames, 3)`` array per residue trajectory in nm.
        """
        for source in self.sources:
            trajs = _compute_residue_com_trajectories(
                source.universe, source.atomgroup, self.n_frames, source.n_residues
            )
            for i in range(source.n_residues):
                yield trajs[i]


def get_reader(
    fz=None,
    universe=None,
    universes=None,
    selection=None,
    normalize_lengths: bool = False,
    cluster_ids=None,
) -> TrajectoryReader:
    """Create a trajectory reader from text files or MDAnalysis input.

    Parameters
    ----------
    fz : str or Path or sequence[str | Path], optional
        Text trajectory input path(s).
    universe : MDAnalysis.Universe or MDAnalysis.AtomGroup, optional
        MDAnalysis input object.
    universes : sequence[MDAnalysis.Universe | MDAnalysis.AtomGroup], optional
        Multiple MDAnalysis inputs to pool into one multi-molecule analysis.
        All inputs must have matching frame count and dt.
    selection : str or list[str], optional
        MDAnalysis selection string. Ignored when ``fz`` is provided. For
        ``universes``, pass a single string to apply the same selection to
        every Universe, or a list of strings (one per Universe) to apply
        different selections to each.
    normalize_lengths : bool, default=False
        Enable truncation to shortest length for multi-file text input.
    cluster_ids : sequence[object] or None, optional
        Cluster identifiers aligned with text files, a single MDAnalysis input,
        or the entries in ``universes``.

    Returns
    -------
    TrajectoryReader
        Concrete reader instance matching the provided input mode.

    Raises
    ------
    ValueError
        If multiple input modes are provided, or if none is provided.
    """
    provided_modes = sum(v is not None for v in (fz, universe, universes))
    if provided_modes > 1:
        raise ValueError("Provide exactly one input mode: 'fz', 'universe', or 'universes'.")
    if provided_modes == 0:
        raise ValueError("Must provide one input mode: 'fz', 'universe', or 'universes'.")

    if fz is not None and selection is not None:
        warnings.warn("'selection' is ignored when 'fz' is provided.")

    if universe is not None:
        if normalize_lengths:
            warnings.warn("'normalize_lengths' is ignored for MDAnalysis inputs.", UserWarning)
        return MDAnalysisReader(universe, selection, cluster_ids=cluster_ids)
    if universes is not None:
        if normalize_lengths:
            warnings.warn("'normalize_lengths' is ignored for MDAnalysis inputs.", UserWarning)
        return MDAnalysisMultiReader(universes, selection, cluster_ids=cluster_ids)
    if fz is not None:
        return NumpyTextReader(
            fz,
            normalize_lengths=normalize_lengths,
            cluster_ids=cluster_ids,
        )

    # Defensive fallback; should not be reachable due to provided_modes checks.
    raise ValueError("Unable to determine trajectory reader input mode.")
