import pytest
import numpy as np
import os
import pickle
from Dfit.Dfit import Dcov


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


def test_save_load_cycle(random_walk_file, tmp_path):
    fout = str(tmp_path / 'D_analysis_save_load')
    dcov = Dcov(fz=random_walk_file, dt=1.0, m=5, tmax=10.0, fout=fout)
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
    dcov = Dcov(fz=random_walk_file, dt=1.0, m=5, tmax=10.0, fout=fout)
    dcov.run_Dfit()
    
    # 1. Test custom prefix
    custom_prefix = str(tmp_path / 'my_custom_output')
    dcov.analysis(tc=5.0, fout_prefix=custom_prefix)
    
    assert os.path.exists(f"{custom_prefix}.dat")
    assert os.path.exists(f"{custom_prefix}.csv")
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
    dcov = Dcov(fz=random_walk_file, dt=1.0, m=5, tmax=10.0, fout=fout)
    dcov.run_Dfit()
    
    dcov.analysis(tc='auto')
    
    # We need to find what tc was selected to verify the file
    tc_selected = dcov.tc_selected
    expected_suffix = f"{tc_selected:.4g}"
    expected_path = f"{fout}.tc_{expected_suffix}.dat"
    
    assert os.path.exists(expected_path)


def test_auto_tc_metadata_persists(random_walk_file, tmp_path):
    fout = str(tmp_path / 'D_analysis_auto_bound')
    dcov = Dcov(fz=random_walk_file, dt=0.5, m=5, tmax=5.0, fout=fout)
    dcov.run_Dfit()

    q_profile = np.full(dcov.tmax - dcov.tmin + 1, 0.9)
    q_profile[1] = 0.49
    q_profile[4] = 0.48
    _set_q_profile(dcov, q_profile)

    dcov.analysis(tc='auto', auto_min_tc=2.1)

    pkl_path = f"{fout}.analysis.pkl"
    with open(pkl_path, 'wb') as handle:
        pickle.dump(dcov, handle)

    dcov_loaded = Dcov.load(pkl_path)
    assert dcov_loaded.tc_selected == pytest.approx(dcov.tc_selected)
    assert dcov_loaded.tc_selected_idx == dcov.tc_selected_idx
    assert dcov_loaded.tc_auto_unbounded == pytest.approx(dcov.tc_auto_unbounded)
    assert dcov_loaded.tc_auto_unbounded_idx == dcov.tc_auto_unbounded_idx
    assert dcov_loaded.auto_min_tc_used == pytest.approx(dcov.auto_min_tc_used)


def test_legacy_pickle_without_cluster_metadata_is_conservative(random_walk_file, tmp_path):
    """Legacy fitted models fall back to one cluster rather than pseudoreplicas."""
    dcov = Dcov(
        fz=random_walk_file,
        dt=1.0,
        m=5,
        tmax=5.0,
        fout=str(tmp_path / "legacy"),
        progress=False,
    )
    dcov.run_Dfit()
    for attribute in ("segment_cluster_ids", "segment_lengths", "s2var_members"):
        delattr(dcov, attribute)

    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as handle:
        pickle.dump(dcov, handle)

    with pytest.warns(UserWarning, match="legacy model without cluster metadata"):
        loaded = Dcov.load(str(path))

    assert loaded.cluster_ids == ("cluster_0",)
    assert loaded.segment_cluster_ids == ("cluster_0",) * loaded.nseg
    assert loaded.s2var_members.shape == (len(loaded.s2var), loaded.nseg)
