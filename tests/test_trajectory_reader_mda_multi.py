import types

import pytest

import Dfit.trajectory_reader as trajectory_reader


class _FakeTrajectory:
    """Minimal MDAnalysis-like trajectory object for metadata tests."""

    def __init__(self, n_frames: int, dt: float):
        self.n_frames = n_frames
        self.dt = dt


class _FakeAtomGroup:
    """Minimal MDAnalysis-like AtomGroup exposing residue metadata."""

    def __init__(self, universe, n_residues: int):
        self.universe = universe
        self.n_residues = n_residues


class _FakeUniverse:
    """Minimal MDAnalysis-like Universe exposing selection and trajectory metadata."""

    def __init__(self, n_frames: int, dt: float, n_residues: int):
        self.trajectory = _FakeTrajectory(n_frames=n_frames, dt=dt)
        self.atoms = _FakeAtomGroup(self, n_residues=n_residues)

    def select_atoms(self, _selection):
        return self.atoms


def _install_fake_mda(monkeypatch):
    """Patch trajectory_reader globals with fake MDAnalysis types."""

    fake_mda = types.SimpleNamespace(Universe=_FakeUniverse, AtomGroup=_FakeAtomGroup)
    monkeypatch.setattr(trajectory_reader, "mda", fake_mda, raising=False)
    monkeypatch.setattr(trajectory_reader, "HAS_MDA", True)


def test_mda_multi_reader_metadata(monkeypatch):
    """Pooled reader should aggregate residue counts across inputs."""

    _install_fake_mda(monkeypatch)
    u1 = _FakeUniverse(n_frames=101, dt=2.0, n_residues=32)
    u2 = _FakeUniverse(n_frames=101, dt=2.0, n_residues=20)

    reader = trajectory_reader.MDAnalysisMultiReader([u1, u2], selection_str="all")

    assert reader.n_frames == 101
    assert reader.dt == 2.0
    assert reader.ndim == 3
    assert reader.n_trajs == 52


def test_mda_multi_reader_mismatched_frames(monkeypatch):
    """Inputs with different frame counts should be rejected."""

    _install_fake_mda(monkeypatch)
    u1 = _FakeUniverse(n_frames=101, dt=2.0, n_residues=32)
    u2 = _FakeUniverse(n_frames=100, dt=2.0, n_residues=32)

    with pytest.raises(ValueError, match="same number of frames"):
        trajectory_reader.MDAnalysisMultiReader([u1, u2], selection_str="all")


def test_mda_multi_reader_mismatched_dt(monkeypatch):
    """Inputs with different dt should be rejected."""

    _install_fake_mda(monkeypatch)
    u1 = _FakeUniverse(n_frames=101, dt=2.0, n_residues=32)
    u2 = _FakeUniverse(n_frames=101, dt=1.0, n_residues=32)

    with pytest.raises(ValueError, match="same dt"):
        trajectory_reader.MDAnalysisMultiReader([u1, u2], selection_str="all")


def test_mda_multi_reader_list_selections(monkeypatch):
    """A list of selection strings (one per universe) should be accepted."""

    _install_fake_mda(monkeypatch)
    u1 = _FakeUniverse(n_frames=101, dt=2.0, n_residues=32)
    u2 = _FakeUniverse(n_frames=101, dt=2.0, n_residues=20)

    reader = trajectory_reader.MDAnalysisMultiReader(
        [u1, u2], selection_str=["resname LIG", "resname SOL"]
    )

    assert reader.n_trajs == 52


def test_mda_multi_reader_list_selections_wrong_length(monkeypatch):
    """A selection list whose length differs from the universes list should raise ValueError."""

    _install_fake_mda(monkeypatch)
    u1 = _FakeUniverse(n_frames=101, dt=2.0, n_residues=32)
    u2 = _FakeUniverse(n_frames=101, dt=2.0, n_residues=20)

    with pytest.raises(ValueError, match="same length"):
        trajectory_reader.MDAnalysisMultiReader(
            [u1, u2], selection_str=["resname LIG"]
        )

