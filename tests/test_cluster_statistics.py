import csv

import numpy as np
import pytest

from Dfit import AnalysisResult, Dcov, SamplingAdequacyWarning
from Dfit import math_utils
from conftest import generate_random_walk


def _make_stub_dcov(tmp_path, cluster_ids, *, m=3):
    """Create a multi-trajectory estimator suitable for deterministic analysis tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used for input and output files.
    cluster_ids : sequence[str]
        Cluster identifier for every generated text trajectory.
    m : int, default=3
        Number of MSD points configured on the estimator.

    Returns
    -------
    Dcov
        Initialized estimator whose fitted arrays can be populated directly.
    """
    files = []
    for index in range(len(cluster_ids)):
        path = tmp_path / f"trajectory_{index}.dat"
        np.savetxt(path, np.zeros((30, 2)))
        files.append(str(path))
    return Dcov(
        fz=files,
        cluster_ids=cluster_ids,
        dt=1.0,
        m=m,
        tmax=2.0,
        progress=False,
        n_jobs=0,
        fout=str(tmp_path / "cluster_analysis"),
    )


def _install_deterministic_fit(dcov, diffusion_values, member_variances, q_values=None):
    """Populate fitted arrays with deterministic diffusion and variance values.

    Parameters
    ----------
    dcov : Dcov
        Estimator to mark as fitted.
    diffusion_values : sequence[float]
        Per-member diffusion values in internal units.
    member_variances : sequence[float]
        Per-member predicted diffusion variances in internal units squared.
    q_values : ndarray or None, optional
        Optional Q array with shape ``(n_lags, n_members)``.

    Returns
    -------
    None
        Mutates only the supplied test estimator.
    """
    diffusion_values = np.asarray(diffusion_values, dtype=float)
    member_variances = np.asarray(member_variances, dtype=float)
    denominator = 2.0 * dcov.ndim * dcov.dt
    # Deliberately anticorrelate component slopes. The total remains exactly
    # denominator * D_i, so empirical uncertainty must be calculated from the
    # complete D_i rather than summed marginal variances.
    component_0 = 3.0 * diffusion_values
    component_1 = denominator * diffusion_values - component_0
    slopes = np.column_stack((component_0, component_1))
    dcov.s2[:] = slopes[np.newaxis, :, :]
    dcov.a2[:] = 0.0
    dcov.s2var_members[:] = (
        member_variances[np.newaxis, :] * denominator ** 2
    )
    dcov.s2var[:] = np.mean(dcov.s2var_members, axis=1)
    if q_values is None:
        dcov.q[:] = 0.5
    else:
        dcov.q[:] = np.asarray(q_values, dtype=float)
    dcov._fitted = True


def test_cluster_statistics_and_error_decomposition(tmp_path):
    """Pooled, cluster, and propagated uncertainties follow their definitions."""
    cluster_ids = ["box_A"] * 2 + ["box_B"] * 3 + ["box_C"] * 4
    diffusion_values = np.array([1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 13.0])
    member_variances = np.arange(1.0, 10.0) * 0.01
    dcov = _make_stub_dcov(tmp_path, cluster_ids)
    _install_deterministic_fit(dcov, diffusion_values, member_variances)

    result = dcov.analysis(tc=1.0, fout_prefix=str(tmp_path / "result"))

    assert isinstance(result, AnalysisResult)
    assert result.pooled.diffusion == pytest.approx(np.mean(diffusion_values) * 0.01)
    assert result.pooled.empirical_sd == pytest.approx(
        np.std(diffusion_values, ddof=1) * 0.01
    )
    assert result.pooled.predicted_sd == pytest.approx(
        np.sqrt(np.mean(member_variances)) * 0.01
    )
    assert result.pooled.predicted_sem == pytest.approx(
        np.sqrt(np.sum(member_variances)) / len(member_variances) * 0.01
    )

    cluster_means = np.array([2.0, 5.0, 10.0])
    assert [cluster.statistics.diffusion for cluster in result.clusters] == pytest.approx(
        cluster_means * 0.01
    )
    assert result.across_clusters.mean == pytest.approx(np.mean(cluster_means) * 0.01)
    assert result.across_clusters.sample_sd == pytest.approx(
        np.std(cluster_means, ddof=1) * 0.01
    )

    cluster_predicted_sems = []
    cluster_empirical_sems = []
    for indices in dcov.cluster_indices:
        cluster_predicted_sems.append(
            np.sqrt(np.sum(member_variances[indices])) / len(indices)
        )
        cluster_empirical_sems.append(
            np.std(diffusion_values[indices], ddof=1) / np.sqrt(len(indices))
        )
    assert result.across_clusters.propagated_predicted_sem == pytest.approx(
        np.sqrt(np.sum(np.square(cluster_predicted_sems))) / 3.0 * 0.01
    )
    assert result.across_clusters.propagated_empirical_sem == pytest.approx(
        np.sqrt(np.sum(np.square(cluster_empirical_sems))) / 3.0 * 0.01
    )

    with open(tmp_path / "result.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["scope"] for row in rows} == {"pooled", "cluster", "across_clusters"}
    assert {row["cluster_id"] for row in rows if row["scope"] == "cluster"} == {
        "box_A",
        "box_B",
        "box_C",
    }
    report_text = (tmp_path / "result.dat").read_text(encoding="utf-8")
    assert "POOLED TRAJECTORY-LEVEL RESULT (CONDITIONAL ON SAMPLED CLUSTERS)" in report_text
    assert "ACROSS INDEPENDENT CLUSTERS (EQUAL WEIGHT)" in report_text
    assert "not added in quadrature" in report_text
    assert all(cluster_id in report_text for cluster_id in ("box_A", "box_B", "box_C"))


def test_cluster_balanced_auto_tc_does_not_use_member_weighting(tmp_path):
    """Diagnostic auto selection gives every cluster equal influence."""
    cluster_ids = ["small_A", "small_B"] + ["large_C"] * 7
    dcov = _make_stub_dcov(tmp_path, cluster_ids)
    q_values = np.array(
        [
            [0.1, 0.1, *([0.6] * 7)],
            [0.62] * 9,
        ]
    )
    _install_deterministic_fit(
        dcov,
        diffusion_values=np.ones(9),
        member_variances=np.full(9, 0.01),
        q_values=q_values,
    )

    dcov.analysis(tc="auto", fout_prefix=str(tmp_path / "auto"))

    assert dcov.tc_selected == pytest.approx(2.0)
    assert abs(np.mean(q_values[0]) - 0.5) < abs(np.mean(q_values[1]) - 0.5)
    assert dcov.q_cluster_score[1] < dcov.q_cluster_score[0]


def test_m2_numeric_analysis_has_no_q_and_auto_rejects(tmp_path):
    """The exactly determined m=2 fit cannot provide a Q-based cutoff."""
    dcov = _make_stub_dcov(tmp_path, ["box_A", "box_A"], m=2)
    _install_deterministic_fit(
        dcov,
        diffusion_values=[1.0, 2.0],
        member_variances=[0.1, 0.1],
    )

    result = dcov.analysis(tc=1.0, fout_prefix=str(tmp_path / "m2"))
    assert result.pooled.q_mean is None
    assert result.pooled.q_sd is None

    with pytest.raises(ValueError, match="requires m >= 3"):
        dcov.analysis(tc="auto", fout_prefix=str(tmp_path / "m2_auto"))


def test_explicit_segmentation_is_honored_and_uses_every_frame(tmp_path):
    """A feasible explicit nseg warns when weak but is never replaced."""
    trajectory = generate_random_walk(n_steps=5000, dim=3)
    path = tmp_path / "long_trajectory.dat"
    np.savetxt(path, trajectory)

    with pytest.warns(SamplingAdequacyWarning, match="being honored"):
        dcov = Dcov(fz=str(path), dt=1.0, m=10, tmax=20.0, nseg=5)

    assert dcov.nseg == 5
    assert np.sum(dcov.segment_lengths) == len(trajectory)
    assert np.ptp(dcov.segment_lengths) <= 1

    automatic = Dcov(fz=str(path), dt=1.0, m=10, tmax=20.0, nseg=None)
    assert automatic.nseg == 2


def test_covariance_pseudoinverse_is_scale_relative():
    """Covariance inversion scales correctly at very small magnitudes."""
    covariance = np.array([[2.0e-15, 0.5e-15], [0.5e-15, 1.0e-15]])
    inverse = math_utils.inv_mat(covariance)
    scaled_inverse = math_utils.inv_mat(covariance * 1.0e8)

    assert np.all(np.isfinite(inverse))
    assert np.allclose(inverse, inverse.T)
    assert np.allclose(scaled_inverse, inverse / 1.0e8, rtol=1e-10, atol=1e-10)


def test_degenerate_zero_diffusion_fit_remains_finite():
    """A zero covariance must not turn the GLS update into NaN values."""
    a2, s2, converged = math_utils.calc_gls(
        20,
        3,
        np.zeros(3),
        1e-10,
        10,
    )

    assert converged
    assert a2 == pytest.approx(0.0)
    assert s2 == pytest.approx(0.0)
    assert math_utils.eval_vars(20, 3, np.zeros(1), np.zeros(1), 1) == pytest.approx(0.0)
