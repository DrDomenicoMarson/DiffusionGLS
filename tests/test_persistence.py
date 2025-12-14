import pytest
import numpy as np
import os
import pickle
from Dfit.Dfit import Dcov

# Mock data generation (reused from test_dfit.py)
def generate_random_walk(n_steps, dim=3, diffusion_coeff=1.0, dt=1.0):
    step_std = np.sqrt(2 * diffusion_coeff * dt)
    steps = np.random.normal(0, step_std, size=(n_steps, dim))
    trajectory = np.cumsum(steps, axis=0)
    trajectory = np.vstack([np.zeros((1, dim)), trajectory])
    return trajectory

@pytest.fixture
def random_walk_file(tmp_path):
    traj = generate_random_walk(n_steps=1000, dim=3, diffusion_coeff=0.1, dt=1.0)
    file_path = tmp_path / "test_traj.dat"
    np.savetxt(file_path, traj)
    return str(file_path)

def test_save_load_cycle(random_walk_file, tmp_path):
    fout = str(tmp_path / 'D_analysis_save_load')
    dcov = Dcov(fz=random_walk_file, m=5, tmax=10.0, fout=fout)
    dcov.run_Dfit(save_model=True)
    
    # Check if pickle file exists
    pkl_path = f"{fout}.pkl"
    assert os.path.exists(pkl_path)
    
    # Load model
    dcov_loaded = Dcov.load(pkl_path)
    
    # Compare simple attributes
    assert dcov_loaded.m == dcov.m
    assert dcov_loaded.dt == dcov.dt
    
    # Compare calculated arrays
    assert np.allclose(dcov_loaded.a2, dcov.a2)
    assert np.allclose(dcov_loaded.s2, dcov.s2)
    assert np.allclose(dcov_loaded.s2var, dcov.s2var)
    assert np.allclose(dcov_loaded.q, dcov.q)

def test_flexible_analysis_output(random_walk_file, tmp_path):
    fout = str(tmp_path / 'D_analysis_flex')
    dcov = Dcov(fz=random_walk_file, m=5, tmax=10.0, fout=fout)
    dcov.run_Dfit()
    
    # 1. Test custom prefix
    custom_prefix = str(tmp_path / 'my_custom_output')
    dcov.analysis(tc=5.0, fout_prefix=custom_prefix)
    
    assert os.path.exists(f"{custom_prefix}.dat")
    assert os.path.exists(f"{custom_prefix}.pdf")
    
    # 2. Test default naming (append tc)
    dcov.analysis(tc=5.0)
    # tc=5.0 should result in output_5.dat (since 5.0 formatted with .4g is 5)
    # Let's verify what the code actually produces.
    # tc_disp = 5.0. .4g -> '5' probably? 
    # If 5.0, it might be '5'.
    
    # Let's check generally for files startin with dcov.fout
    # We expect D_analysis_flex_5.dat or D_analysis_flex_5.0.dat
    
    # Explicitly check for what we expect standard formatting to do for 5.0
    expected_suffix = f"{5.0:.4g}" # likely '5'
    expected_base = f"{fout}.tc_{expected_suffix}"
    
    assert os.path.exists(f"{expected_base}.dat")
    assert os.path.exists(f"{expected_base}.pdf")

def test_auto_tc_naming(random_walk_file, tmp_path):
    fout = str(tmp_path / 'D_analysis_auto')
    dcov = Dcov(fz=random_walk_file, m=5, tmax=10.0, fout=fout)
    dcov.run_Dfit()
    
    dcov.analysis(tc='auto')
    
    # We need to find what tc was selected to verify the file
    tc_selected = dcov.tc_selected
    expected_suffix = f"{tc_selected:.4g}"
    expected_path = f"{fout}.tc_{expected_suffix}.dat"
    
    assert os.path.exists(expected_path)
