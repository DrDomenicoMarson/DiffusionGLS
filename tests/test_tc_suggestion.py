import csv
import pickle

import numpy as np
import pytest

from Dfit import Dcov, TcSuggestion


def _make_profile_dcov(tmp_path, cluster_ids, *, m=3):
    """Create a fitted-array-compatible estimator for cutoff tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used for generated text trajectories.
    cluster_ids : sequence[str]
        Cluster identifier for every member trajectory.
    m : int, default=3
        Number of MSD points configured for each GLS fit.

    Returns
    -------
    Dcov
        Initialized estimator with a 1--12 ps lag grid.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = []
    for index in range(len(cluster_ids)):
        path = tmp_path / f"tc_member_{index}.dat"
        np.savetxt(path, np.zeros((40, 2)))
        files.append(path)
    return Dcov(
        fz=files,
        cluster_ids=cluster_ids,
        dt=1.0,
        m=m,
        tmax=12.0,
        fout=str(tmp_path / "tc_analysis"),
        imgfmt="png",
        progress=False,
        n_jobs=0,
    )


def _install_cluster_profiles(dcov, diffusion_profiles, q_profiles):
    """Populate deterministic cluster ``D(t)`` and ``Q(t)`` profiles.

    Parameters
    ----------
    dcov : Dcov
        Estimator to mark as fitted.
    diffusion_profiles : mapping[str, array-like]
        Internal diffusion profile for every cluster ID.
    q_profiles : mapping[str, array-like]
        Mean Q profile for every cluster ID.

    Returns
    -------
    None
        Mutates only the supplied estimator's fitted arrays.
    """
    denominator = 2.0 * dcov.ndim * dcov.dt
    for member_index, cluster_id in enumerate(dcov.segment_cluster_ids):
        diffusion = np.asarray(diffusion_profiles[cluster_id], dtype=float)
        q_values = np.asarray(q_profiles[cluster_id], dtype=float)
        dcov.s2[:, member_index, :] = 0.0
        dcov.s2[:, member_index, 0] = denominator * diffusion
        dcov.q[:, member_index] = q_values
    dcov.s2var_members[:] = 0.01 * denominator ** 2
    dcov.s2var[:] = np.mean(dcov.s2var_members, axis=1)
    dcov._fitted = True


def _passing_profiles():
    """Return profiles with cluster plateau onsets at 3, 5, and 1 ps.

    Returns
    -------
    tuple[dict[str, ndarray], dict[str, ndarray]]
        Diffusion and Q profiles keyed by cluster ID.
    """
    diffusion = {
        "box_A": np.array([1.30, 1.15, *([1.00] * 10)]),
        "box_B": np.array([1.50, 1.40, 1.30, 1.20, *([0.90] * 8)]),
        "box_C": np.full(12, 1.10),
    }
    q_values = {
        "box_A": np.array([0.10, 0.20, *([0.50] * 10)]),
        "box_B": np.array([0.10, 0.15, 0.20, 0.20, *([0.50] * 8)]),
        "box_C": np.full(12, 0.50),
    }
    return diffusion, q_values


def test_publication_tc_suggestion_requires_all_clusters_and_writes_outputs(tmp_path):
    """The common suggestion is the first sustained all-cluster candidate."""
    dcov = _make_profile_dcov(
        tmp_path,
        ["box_A", "box_A", "box_B", "box_C", "box_C", "box_C"],
    )
    diffusion, q_values = _passing_profiles()
    _install_cluster_profiles(dcov, diffusion, q_values)
    out_base = tmp_path / "suggestion"

    suggestion = dcov.suggest_tc(
        validation_window=2.0,
        min_tc=1.0,
        relative_drift_tolerance=0.05,
        q_tolerance=0.10,
        persistence_windows=2,
        blocks_per_window=2,
        candidate_step=1.0,
        fout_prefix=str(out_base),
    )

    assert isinstance(suggestion, TcSuggestion)
    assert suggestion.status == "suggested"
    assert suggestion.tc == pytest.approx(5.0)
    assert suggestion.selected_candidate.validation_end == pytest.approx(9.0)
    assert {onset.cluster_id: onset.tc for onset in suggestion.cluster_onsets} == {
        "box_A": pytest.approx(3.0),
        "box_B": pytest.approx(5.0),
        "box_C": pytest.approx(1.0),
    }
    assert all(cluster.passes for cluster in suggestion.selected_candidate.clusters)
    assert suggestion.selected_candidate.clusters[1].empirical_sem is None
    assert suggestion.selected_candidate.clusters[0].empirical_sem is not None
    assert suggestion.candidate_step == pytest.approx(1.0)

    report = out_base.with_suffix(".dat").read_text(encoding="utf-8")
    assert "Suggested common tc: 5 ps" in report
    assert "Conditional SEMs provide within-box context only" in report
    assert out_base.with_suffix(".png").exists()
    with out_base.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["cluster_id"] for row in rows} == {"box_A", "box_B", "box_C"}
    assert sum(row["selected"] == "True" for row in rows) == 3
    assert all("max_relative_drift" in row for row in rows)

    result = dcov.analysis(tc=suggestion.tc, fout_prefix=str(tmp_path / "final"))
    assert result.tc == pytest.approx(5.0)


def test_publication_tc_suggestion_does_not_promote_closest_failing_candidate(tmp_path):
    """A persistently drifting cluster produces ``tc=None`` rather than a fallback."""
    dcov = _make_profile_dcov(tmp_path, ["box_A", "box_B", "box_C"])
    diffusion, q_values = _passing_profiles()
    diffusion["box_C"] = np.linspace(0.5, 2.0, 12)
    _install_cluster_profiles(dcov, diffusion, q_values)
    out_base = tmp_path / "no_plateau"

    suggestion = dcov.suggest_tc(
        validation_window=2.0,
        relative_drift_tolerance=0.01,
        q_tolerance=0.10,
        persistence_windows=2,
        blocks_per_window=2,
        candidate_step=1.0,
        fout_prefix=str(out_base),
        make_plot=False,
    )

    assert suggestion.status == "no_common_plateau"
    assert suggestion.tc is None
    assert suggestion.selected_candidate is None
    assert suggestion.closest_candidate.passes is False
    assert dict((onset.cluster_id, onset.tc) for onset in suggestion.cluster_onsets)[
        "box_C"
    ] is None
    report = out_base.with_suffix(".dat").read_text(encoding="utf-8")
    assert "Suggested common tc: NONE" in report
    assert "CLOSEST FAILING CANDIDATE (DIAGNOSTIC ONLY; NOT SELECTED)" in report
    with out_base.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert not any(row["selected"] == "True" for row in rows)


def test_publication_tc_suggestion_rejects_isolated_q_crossing(tmp_path):
    """One Q value near 0.5 cannot replace sustained block-level adequacy."""
    dcov = _make_profile_dcov(tmp_path, ["box_A", "box_A"])
    diffusion = {"box_A": np.ones(12)}
    q_values = {"box_A": np.array([0.2, 0.2, 0.5, *([0.2] * 9)])}
    _install_cluster_profiles(dcov, diffusion, q_values)

    suggestion = dcov.suggest_tc(
        validation_window=2.0,
        q_tolerance=0.10,
        persistence_windows=2,
        blocks_per_window=2,
        candidate_step=1.0,
        fout_prefix=str(tmp_path / "isolated_q"),
        make_plot=False,
    )

    assert suggestion.tc is None
    assert suggestion.cluster_onsets[0].tc is None
    assert all(not candidate.clusters[0].q_passes for candidate in suggestion.candidates)


def test_publication_tc_suggestion_survives_pickle_reload(tmp_path):
    """A fitted 0.4-era-style pickle has enough arrays for the new selector."""
    dcov = _make_profile_dcov(tmp_path, ["box_A", "box_B", "box_C"])
    diffusion, q_values = _passing_profiles()
    _install_cluster_profiles(dcov, diffusion, q_values)
    del dcov.tc_suggestion
    pickle_path = tmp_path / "fitted.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(dcov, handle)

    loaded = Dcov.load(str(pickle_path))
    suggestion = loaded.suggest_tc(
        validation_window=2.0,
        persistence_windows=2,
        blocks_per_window=2,
        candidate_step=1.0,
        fout_prefix=str(tmp_path / "loaded_suggestion"),
        make_plot=False,
    )

    assert suggestion.tc == pytest.approx(5.0)
    assert loaded.tc_suggestion is suggestion


def test_publication_tc_suggestion_validates_statistical_contract(tmp_path):
    """Invalid Q, duration, persistence, and m settings fail explicitly."""
    dcov = _make_profile_dcov(tmp_path, ["box_A", "box_B", "box_C"])
    diffusion, q_values = _passing_profiles()
    _install_cluster_profiles(dcov, diffusion, q_values)

    with pytest.raises(ValueError, match="q_tolerance must not exceed"):
        dcov.suggest_tc(validation_window=2.0, q_tolerance=0.6)
    with pytest.raises(ValueError, match="multiple of dt"):
        dcov.suggest_tc(validation_window=2.5)
    with pytest.raises(ValueError, match="too short for blocks_per_window"):
        dcov.suggest_tc(validation_window=2.0, blocks_per_window=3)
    with pytest.raises(ValueError, match="lag range is too short"):
        dcov.suggest_tc(
            validation_window=4.0,
            persistence_windows=3,
            blocks_per_window=2,
        )

    m2 = _make_profile_dcov(tmp_path / "m2", ["box_A", "box_B"], m=2)
    m2_diffusion = {"box_A": np.ones(12), "box_B": np.ones(12)}
    m2_q = {"box_A": np.full(12, np.nan), "box_B": np.full(12, np.nan)}
    _install_cluster_profiles(m2, m2_diffusion, m2_q)
    with pytest.raises(ValueError, match="requires m >= 3"):
        m2.suggest_tc(validation_window=2.0, blocks_per_window=2)


def test_three_box_smoke_fit_suggestion_and_numeric_analysis(tmp_path):
    """Three independently fitted boxes complete the full public workflow."""
    rng = np.random.default_rng(20260713)
    paths = []
    for box_index in range(3):
        increments = rng.normal(size=(240, 2))
        trajectory = np.vstack((np.zeros((1, 2)), np.cumsum(increments, axis=0)))
        path = tmp_path / f"smoke_box_{box_index}.dat"
        np.savetxt(path, trajectory)
        paths.append(path)

    dcov = Dcov(
        fz=paths,
        cluster_ids=["box_1", "box_2", "box_3"],
        dt=1.0,
        m=3,
        tmax=4.0,
        fout=str(tmp_path / "smoke"),
        progress=False,
        n_jobs=0,
    )
    dcov.run_Dfit()
    suggestion = dcov.suggest_tc(
        validation_window=2.0,
        relative_drift_tolerance=1.0e6,
        q_tolerance=0.5,
        persistence_windows=1,
        blocks_per_window=2,
        candidate_step=1.0,
        fout_prefix=str(tmp_path / "smoke_suggestion"),
        make_plot=False,
    )

    assert suggestion.tc is not None
    result = dcov.analysis(tc=suggestion.tc, fout_prefix=str(tmp_path / "smoke_final"))
    assert result.across_clusters.n_clusters == 3
    assert len(result.clusters) == 3
