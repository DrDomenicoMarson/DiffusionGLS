
import pytest
import numpy as np
import os
from Dfit.Dfit import Dcov, XI_CUBIC, BOLTZMANN_K
from Dfit.trajectory_reader import NumpyTextReader

# Mock data generation
def generate_random_walk(n_steps, dim=3, diffusion_coeff=1.0, dt=1.0):
    # MSD = 2 * dim * D * t
    # variance of step = 2 * D * dt
    step_std = np.sqrt(2 * diffusion_coeff * dt)
    steps = np.random.normal(0, step_std, size=(n_steps, dim))
    trajectory = np.cumsum(steps, axis=0)
    # Add initial position 0
    trajectory = np.vstack([np.zeros((1, dim)), trajectory])
    return trajectory

@pytest.fixture
def random_walk_file(tmp_path):
    # Create a temporary trajectory file
    traj = generate_random_walk(n_steps=5000, dim=3, diffusion_coeff=0.1, dt=1.0)
    file_path = tmp_path / "test_traj.dat"
    np.savetxt(file_path, traj)
    return str(file_path)

def test_reader_text(random_walk_file):
    reader = NumpyTextReader(random_walk_file)
    assert reader.n_trajs == 1
    assert reader.ndim == 3
    
    # Iterate
    trajs = list(reader)
    assert len(trajs) == 1
    assert trajs[0].shape[0] == 5001

def test_dcov_initialization(random_walk_file):
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0)
    assert dcov.ndim == 3
    assert dcov.n == 5000
    assert dcov.dt == 1.0

def test_run_dfit(random_walk_file):
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0, nseg=5)
    dcov.run_Dfit()
    dcov.analysis(tc=10)
    
    # Check if results are populated
    assert dcov.D is not None
    assert dcov.Dstd is not None
    assert dcov.q_m is not None
    
    # Check shapes
    assert len(dcov.D) == 20 # tmax - tmin + 1 (default tmin=1)
    
    # Basic sanity check: D should be positive
    assert np.all(dcov.D > 0)

def test_finite_size_correction(random_walk_file):
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0)
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

def test_timestep_index(random_walk_file):
    dcov = Dcov(fz=random_walk_file, dt=0.5, tmin=1.0, tmax=10.0)
    
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

def test_auto_tc(random_walk_file):
    dcov = Dcov(fz=random_walk_file, m=10, tmax=20.0)
    dcov.run_Dfit()
    
    # Run with auto
    dcov.analysis(tc='auto')
    
    # Check if output file exists
    assert os.path.exists('D_analysis.dat')
    
    # We can't easily assert WHICH tc was chosen without parsing stdout or checking internals,
    # but we can check that it didn't crash and produced output.
    # Ideally we would check if the chosen Q is close to 0.5, but with random walk data it might vary.
    # Let's just ensure it runs.

def test_short_trajectory_error(tmp_path):
    # Create a short trajectory
    # N=40, m=20, tmax=5.0 (5 steps).
    # step=5. len = 40/5 = 8. 8 <= 20. Should fail.
    traj = generate_random_walk(n_steps=40, dim=3)
    file_path = tmp_path / "short_traj.dat"
    np.savetxt(file_path, traj)
    
    with pytest.raises(ValueError, match="Trajectory too short"):
        # tmax=5.0 means 5 steps (dt=1.0)
        dcov = Dcov(fz=str(file_path), m=20, tmax=5.0, dt=1.0)
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
        dcov = Dcov(fz=str(file_path), m=5, tmax=20.0, dt=1.0, nseg=2)
        dcov.run_Dfit()

def test_convergence_warning(tmp_path, capsys):
    # Force non-convergence by setting nitmax=1
    # N=50, m=20, tmax=1.0 (1 step).
    traj = generate_random_walk(n_steps=50, dim=3)
    file_path = tmp_path / "conv_traj.dat"
    np.savetxt(file_path, traj)
    
    dcov = Dcov(fz=str(file_path), m=20, tmax=1.0, dt=1.0, nitmax=1)
    dcov.run_Dfit()
    
    captured = capsys.readouterr()
    assert "WARNING: Optimizer did not converge" in captured.out
    assert "Falling back to M=2" in captured.out
