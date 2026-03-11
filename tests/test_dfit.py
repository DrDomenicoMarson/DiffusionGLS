
import pytest
import numpy as np
import os
from Dfit.Dfit import Dcov, XI_CUBIC, BOLTZMANN_K
from Dfit.trajectory_reader import NumpyTextReader
from conftest import generate_random_walk


def _set_q_profile(dcov: Dcov, q_profile: np.ndarray) -> None:
    """Override segment Q values with a deterministic mean profile for tests.

    Parameters
    ----------
    dcov : Dcov
        Fitted diffusion estimator whose ``q`` array will be overwritten.
    q_profile : ndarray
        One-dimensional array of length ``tmax - tmin + 1`` containing the
        desired mean ``Q`` value at each lag.
    """
    q_profile = np.asarray(q_profile, dtype=float)
    expected_shape = (dcov.tmax - dcov.tmin + 1,)
    if q_profile.shape != expected_shape:
        raise ValueError(f"q_profile must have shape {expected_shape}, got {q_profile.shape}")
    dcov.q[:] = q_profile[:, np.newaxis]


def test_reader_text(random_walk_file):
    reader = NumpyTextReader(random_walk_file)
    assert reader.n_trajs == 1
    assert reader.ndim == 3
    
    # Iterate
    trajs = list(reader)
    assert len(trajs) == 1
    assert trajs[0].shape[0] == 5001

def test_dcov_initialization(random_walk_file, tmp_path):
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0, fout=str(tmp_path / 'D_analysis'))
    assert dcov.ndim == 3
    assert dcov.n == 5000
    assert dcov.dt == 1.0

def test_run_dfit(random_walk_file, tmp_path):
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0, nseg=5, fout=str(tmp_path / 'D_analysis'))
    dcov.run_Dfit()
    dcov.analysis(tc=10)
    
    # Check if results are populated
    assert dcov.D is not None
    assert dcov.Dstd is not None
    assert dcov.Dsem_pred is not None
    assert dcov.Dsem_emp is not None
    assert dcov.q_m is not None
    
    # Check shapes
    assert len(dcov.D) == 20 # tmax - tmin + 1 (default tmin=1)
    assert dcov.Dsem_pred.shape == dcov.Dstd.shape
    assert dcov.Dsem_emp.shape == dcov.Dempstd.shape
    assert np.allclose(dcov.Dsem_pred, dcov.Dstd / np.sqrt(dcov.nseg))
    assert np.allclose(dcov.Dsem_emp, dcov.Dempstd / np.sqrt(dcov.nseg))
    
    # Basic sanity check: D should be positive
    assert np.all(dcov.D > 0)

def test_finite_size_correction(random_walk_file, tmp_path):
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0, fout=str(tmp_path / 'D_analysis'))
    dcov.run_Dfit()
    dcov.analysis(tc=10)
    
    # Mock values
    T = 300
    eta = 0.001 # Pa*s
    L = 10.0 # nm
    tc = 10
    
    # Capture stdout to check print
    dcov.finite_size_correction(T=T, eta=eta, L=L, tc=tc)
    
    # Check if Dcor is calculated
    assert hasattr(dcov, 'Dcor')
    
    # Manual calculation check
    itc = dcov._timestep_index(tc)
    kbT = T * BOLTZMANN_K
    expected_correction = kbT * XI_CUBIC * 1e15 / (6. * np.pi * eta * L)
    expected_Dcor = dcov.D[itc] + expected_correction
    
    assert np.isclose(dcov.Dcor[itc], expected_Dcor)

def test_timestep_index(random_walk_file, tmp_path):
    with pytest.warns(UserWarning, match="differs from"):
        dcov = Dcov(fz=random_walk_file, dt=0.5, tmin=1.0, tmax=10.0, fout=str(tmp_path / 'D_analysis'))
    
    # Valid tc
    idx = dcov._timestep_index(2.0) # 2.0 / 0.5 = 4 steps. tmin is 2 steps (idx 0). So 4 steps is idx 2?
    # Wait, tmin is in steps. 
    # tmin=2 means start at step 2.
    # tc=2.0 means step 4.
    # itc = 4 - 2 = 2.
    assert idx == 2
    
    # Invalid tc (not multiple)
    with pytest.raises(ValueError, match="multiple of dt"):
        dcov._timestep_index(2.1)
        
    # Invalid tc (out of range)
    with pytest.raises(ValueError, match="outside"):
        dcov._timestep_index(1000.0)

def test_auto_tc(random_walk_file, tmp_path):
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0, fout=str(tmp_path / 'D_analysis'))
    dcov.run_Dfit()
    
    # Run with auto
    dcov.analysis(tc='auto')
    
    # Check if output file exists
    tc_selected = dcov.tc_selected
    expected_path = f"{tmp_path / 'D_analysis'}.tc_{tc_selected:.4g}.dat"
    assert os.path.exists(expected_path)
    assert dcov.tc_auto_unbounded == pytest.approx(tc_selected)
    assert dcov.tc_auto_unbounded_idx == dcov.tc_selected_idx
    assert dcov.auto_min_tc_used is None
    
    # We can't easily assert WHICH tc was chosen without parsing stdout or checking internals,
    # but we can check that it didn't crash and produced output.
    # Ideally we would check if the chosen Q is close to 0.5, but with random walk data it might vary.
    # Let's just ensure it runs.


def test_auto_tc_with_lower_bound_rounds_up(random_walk_file, tmp_path):
    with pytest.warns(UserWarning, match="differs from"):
        dcov = Dcov(
            fz=random_walk_file,
            dt=0.5,
            m=5,
            tmax=5.0,
            fout=str(tmp_path / 'D_analysis_bound'),
        )
    dcov.run_Dfit()

    q_profile = np.full(dcov.tmax - dcov.tmin + 1, 0.9)
    q_profile[1] = 0.49
    q_profile[4] = 0.48
    _set_q_profile(dcov, q_profile)

    dcov.analysis(tc='auto', auto_min_tc=2.1)

    assert dcov.tc_auto_unbounded == pytest.approx(1.0)
    assert dcov.tc_auto_unbounded_idx == 1
    assert dcov.tc_selected == pytest.approx(2.5)
    assert dcov.tc_selected_idx == 4
    assert dcov.auto_min_tc_used == pytest.approx(2.5)

    expected_path = f"{tmp_path / 'D_analysis_bound'}.tc_{dcov.tc_selected:.4g}.dat"
    assert os.path.exists(expected_path)

    with open(expected_path, 'r', encoding='utf-8') as handle:
        text = handle.read()

    assert "AUTO TC SELECTION:" in text
    assert "Requested auto_min_tc: 2.1 ps" in text
    assert "Applied auto_min_tc on lag grid: 2.5 ps" in text
    assert "Plain auto tc without lower bound: 1 ps" in text


def test_auto_tc_with_excessive_lower_bound_raises(random_walk_file, tmp_path):
    dcov = Dcov(fz=random_walk_file, m=5, tmax=10.0, fout=str(tmp_path / 'D_analysis'))
    dcov.run_Dfit()

    with pytest.raises(ValueError, match="auto_min_tc .* exceeds the computed lag-time range"):
        dcov.analysis(tc='auto', auto_min_tc=100.0)


def test_auto_min_tc_requires_auto_mode(random_walk_file, tmp_path):
    dcov = Dcov(fz=random_walk_file, m=5, tmax=10.0, fout=str(tmp_path / 'D_analysis'))
    dcov.run_Dfit()

    with pytest.raises(ValueError, match="auto_min_tc can only be used when tc='auto'"):
        dcov.analysis(tc=5.0, auto_min_tc=2.0)

def test_mismatched_lengths_error(tmp_path):
    traj1 = generate_random_walk(n_steps=100, dim=3)
    traj2 = generate_random_walk(n_steps=120, dim=3)
    file1 = tmp_path / "traj1.dat"
    file2 = tmp_path / "traj2.dat"
    np.savetxt(file1, traj1)
    np.savetxt(file2, traj2)

    with pytest.raises(ValueError, match="length"):
        Dcov(fz=[str(file1), str(file2)])

def test_mismatched_lengths_normalize(tmp_path):
    n1 = 200
    n2 = 150
    traj1 = generate_random_walk(n_steps=n1, dim=3)
    traj2 = generate_random_walk(n_steps=n2, dim=3)
    file1 = tmp_path / "traj1.dat"
    file2 = tmp_path / "traj2.dat"
    np.savetxt(file1, traj1)
    np.savetxt(file2, traj2)

    min_len = min(n1, n2) + 1  # frames = steps + 1
    with pytest.warns(UserWarning, match="normalize_lengths"):
        dcov = Dcov(fz=[str(file1), str(file2)], normalize_lengths=True, m=10, tmax=10.0, fout=str(tmp_path / 'D_analysis_norm'))
    assert dcov.reader.n_frames == min_len
    assert dcov.n == min_len - 1

    dcov.run_Dfit()
    dcov.analysis(tc=10)

def test_short_trajectory_error(tmp_path):
    # Create a short trajectory
    # N=40, m=20, tmax=5.0 (5 steps).
    # step=5. len = 40/5 = 8. 8 <= 20. Should fail.
    traj = generate_random_walk(n_steps=40, dim=3)
    file_path = tmp_path / "short_traj.dat"
    np.savetxt(file_path, traj)
    
    with pytest.raises(ValueError, match="Trajectory too short"):
        # tmax=5.0 means 5 steps (dt=1.0)
        # Pass nseg=1 to bypass "Timeseries too short" check in init
        dcov = Dcov(fz=str(file_path), m=20, tmax=5.0, dt=1.0, nseg=1, fout=str(tmp_path / 'D_analysis'))
        dcov.run_Dfit()

def test_short_segment_error(tmp_path):
    # Create a trajectory that is long enough overall, but segments are too short
    # N=100. nseg=2. Segment length ~50.
    # tmax=20.0 (20 steps). step=20.
    # Segment len strided = 50 / 20 = 2.5 -> 2 or 3.
    # m=5. 3 <= 5. Should fail.
    traj = generate_random_walk(n_steps=100, dim=3)
    file_path = tmp_path / "short_seg_traj.dat"
    np.savetxt(file_path, traj)
    
    with pytest.raises(ValueError, match="Segment too short"):
        dcov = Dcov(fz=str(file_path), m=5, tmax=20.0, dt=1.0, nseg=2, fout=str(tmp_path / 'D_analysis'))
        dcov.run_Dfit()

def test_convergence_warning(tmp_path, capsys):
    # Force non-convergence by setting nitmax=1
    # N=50, m=20, tmax=1.0 (1 step).
    traj = generate_random_walk(n_steps=50, dim=3)
    file_path = tmp_path / "conv_traj.dat"
    np.savetxt(file_path, traj)
    
    dcov = Dcov(fz=str(file_path), m=20, tmax=1.0, dt=1.0, nitmax=1, nseg=1, fout=str(tmp_path / 'D_analysis'))
    dcov.run_Dfit()
    
    captured = capsys.readouterr()
    assert "WARNING: Optimizer did not converge in 2 cases (100.0% of Total 2)" in captured.out
    assert "Falling back to M=2" in captured.out

def test_n_jobs(random_walk_file, tmp_path):
    # Test running with specific n_jobs
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0, n_jobs=1, fout=str(tmp_path / 'D_analysis_1'))
    dcov.run_Dfit()
    dcov.analysis(tc=10)
    assert os.path.exists(f"{tmp_path / 'D_analysis_1'}.tc_{10.0:.4g}.dat")
    
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0, n_jobs=2, fout=str(tmp_path / 'D_analysis_2'))
    dcov.run_Dfit()
    dcov.analysis(tc=10)
    assert os.path.exists(f"{tmp_path / 'D_analysis_2'}.tc_{10.0:.4g}.dat")

    # n_jobs=0 should be treated as serial (1 worker)
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0, n_jobs=0, fout=str(tmp_path / 'D_analysis_0'))
    dcov.run_Dfit()
    dcov.analysis(tc=10)
    assert os.path.exists(f"{tmp_path / 'D_analysis_0'}.tc_{10.0:.4g}.dat")

def test_time_unit_ns(random_walk_file, tmp_path):
    # Data have dt=1 ps; using time_unit=ns means inputs are in ns
    dcov = Dcov(fz=random_walk_file, dt=0.001, tmax=0.02, m=5, time_unit='ns', fout=str(tmp_path / 'D_analysis_ns'))
    dcov.run_Dfit()
    dcov.analysis(tc=0.01)
    assert os.path.exists(f"{tmp_path / 'D_analysis_ns'}.tc_{0.01:.4g}.dat")
    # tmax=0.02 ns -> 20 ps -> 20 lag steps (tmin=1)
    assert len(dcov.D) == 20

def test_multi_tmax_clamp(tmp_path):
    # Two short trajectories; m and tmax too large should trigger clamp in multi mode
    traj1 = generate_random_walk(n_steps=50, dim=3)
    traj2 = generate_random_walk(n_steps=50, dim=3)
    file1 = tmp_path / "traj1.dat"
    file2 = tmp_path / "traj2.dat"
    np.savetxt(file1, traj1)
    np.savetxt(file2, traj2)

    with pytest.warns(UserWarning, match="tmax"):
        dcov = Dcov(fz=[str(file1), str(file2)], m=10, tmax=100.0, dt=1.0, fout=str(tmp_path / 'D_analysis_clamp'))
    # tmax should be clamped to nperseg // m
    assert dcov.tmax == dcov.nperseg // dcov.m


# --- P1 / P2 validation tests ---

def test_dt_auto_adopt(random_walk_file, tmp_path):
    """When dt is not provided, it should adopt reader.dt (1.0 for text files)."""
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0, fout=str(tmp_path / 'D_analysis'))
    assert dcov.dt == 1.0  # NumpyTextReader defaults to 1.0 ps


def test_dt_explicit_mismatch_warns(random_walk_file, tmp_path):
    """Explicit dt differing from reader.dt should warn."""
    with pytest.warns(UserWarning, match="differs from"):
        Dcov(fz=random_walk_file, dt=0.5, tmax=20.0, fout=str(tmp_path / 'D_analysis'))


def test_m_too_small_error(tmp_path):
    """m clamped below 2 should raise ValueError, not crash later."""
    # Create a very short trajectory: 2 frames -> nperseg=1
    traj = generate_random_walk(n_steps=1, dim=3)  # 2 frames
    file_path = tmp_path / "tiny_traj.dat"
    np.savetxt(file_path, traj)

    with pytest.raises(ValueError, match="m must be >= 2"):
        Dcov(fz=str(file_path), m=20, tmax=1.0, dt=1.0, nseg=1,
             fout=str(tmp_path / 'D_analysis'))


def test_analysis_before_run_error(random_walk_file, tmp_path):
    """analysis() before run_Dfit() should raise RuntimeError."""
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0, fout=str(tmp_path / 'D_analysis'))
    with pytest.raises(RuntimeError, match="run_Dfit"):
        dcov.analysis(tc=10)


def test_fsc_before_analysis_error(random_walk_file, tmp_path):
    """finite_size_correction() before analysis() should raise RuntimeError."""
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0, fout=str(tmp_path / 'D_analysis'))
    dcov.run_Dfit()
    with pytest.raises(RuntimeError, match="analysis"):
        dcov.finite_size_correction(T=300, eta=0.001, L=10.0, tc=10)


def test_empty_files_error(tmp_path):
    """Empty file list should raise ValueError early."""
    with pytest.raises(ValueError, match="No trajectory files"):
        Dcov(fz=[], fout=str(tmp_path / 'D_analysis'))


def test_input_mode_conflict_fz_and_universes(random_walk_file, tmp_path):
    """Providing fz together with universes should raise ValueError."""
    with pytest.raises(ValueError, match="exactly one input mode"):
        Dcov(
            fz=random_walk_file,
            universes=[object()],
            fout=str(tmp_path / 'D_analysis'),
        )


def test_input_mode_conflict_universe_and_universes(tmp_path):
    """Providing universe together with universes should raise ValueError."""
    with pytest.raises(ValueError, match="exactly one input mode"):
        Dcov(
            universe=object(),
            universes=[object()],
            fout=str(tmp_path / 'D_analysis'),
        )


def test_single_frame_error(tmp_path):
    """Trajectory with only 1 frame should raise ValueError early."""
    # np.loadtxt on a single-row 3D file reads shape (3,) which becomes 3 frames×1D.
    # So write a single value to get exactly 1 frame in 1D.
    file_path = tmp_path / "one_frame.dat"
    file_path.write_text("0.0\n")

    with pytest.raises(ValueError, match="too short"):
        Dcov(fz=str(file_path), fout=str(tmp_path / 'D_analysis'))


def test_parallel_consistent_single(random_walk_file, tmp_path):
    """Serial and parallel runs must produce numerically identical results (single-traj mode)."""
    kwargs = dict(fz=random_walk_file, m=10, tmax=20.0, nseg=5, progress=False)

    dcov_serial = Dcov(**kwargs, n_jobs=0, fout=str(tmp_path / 'D_serial'))
    dcov_serial.run_Dfit()
    dcov_serial.analysis(tc=10)

    dcov_par = Dcov(**kwargs, n_jobs=4, fout=str(tmp_path / 'D_par'))
    dcov_par.run_Dfit()
    dcov_par.analysis(tc=10)

    assert np.allclose(dcov_serial.D, dcov_par.D), "D arrays differ between serial and parallel runs"
    assert np.allclose(dcov_serial.a2, dcov_par.a2), "a2 arrays differ"
    assert np.allclose(dcov_serial.s2, dcov_par.s2), "s2 arrays differ"
    assert np.allclose(dcov_serial.q, dcov_par.q),   "q arrays differ"


def test_parallel_consistent_multi(tmp_path):
    """Serial and parallel runs must produce numerically identical results (multi-traj mode)."""
    trajs = [generate_random_walk(n_steps=500, dim=3) for _ in range(8)]
    files = []
    for i, traj in enumerate(trajs):
        p = tmp_path / f"traj{i}.dat"
        np.savetxt(p, traj)
        files.append(str(p))

    kwargs = dict(fz=files, m=10, tmax=20.0, progress=False)

    dcov_serial = Dcov(**kwargs, n_jobs=0, fout=str(tmp_path / 'D_serial'))
    dcov_serial.run_Dfit()
    dcov_serial.analysis(tc=10)

    dcov_par = Dcov(**kwargs, n_jobs=4, fout=str(tmp_path / 'D_par'))
    dcov_par.run_Dfit()
    dcov_par.analysis(tc=10)

    assert np.allclose(dcov_serial.D, dcov_par.D), "D arrays differ between serial and parallel runs"
    assert np.allclose(dcov_serial.a2, dcov_par.a2), "a2 arrays differ"
    assert np.allclose(dcov_serial.s2, dcov_par.s2), "s2 arrays differ"
