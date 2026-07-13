"""Suggest, review, and apply a common publication cutoff from a fitted pickle.

Example
-------
Run a 10 ns validation window starting at 50 ns::

    python Example/run_publication_tc_suggestion.py D_analysis.pkl \
        --validation-window 10 --min-tc 50

The time unit is the ``time_unit`` stored in the fitted model. If no common
plateau passes, the script writes diagnostics and deliberately skips the final
numeric analysis.
"""

import argparse
from collections.abc import Sequence

import Dfit


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the fitted-pickle example.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for publication-cutoff suggestion settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Suggest a sustained common tc across all clusters in a fitted "
            "Dfit pickle, then run analysis with the numeric suggestion."
        )
    )
    parser.add_argument("model", help="Pickle written by Dcov.run_Dfit(save_model=True).")
    parser.add_argument(
        "--validation-window",
        type=float,
        required=True,
        help="Width of each validation window in the model's time_unit.",
    )
    parser.add_argument(
        "--min-tc",
        type=float,
        default=None,
        help="Optional lower cutoff-search bound in the model's time_unit.",
    )
    parser.add_argument(
        "--relative-drift-tolerance",
        type=float,
        default=0.05,
        help="Maximum block-mean relative D drift (default: 0.05).",
    )
    parser.add_argument(
        "--q-tolerance",
        type=float,
        default=0.10,
        help="Maximum block-mean absolute Q deviation from 0.5 (default: 0.10).",
    )
    parser.add_argument(
        "--persistence-windows",
        type=int,
        default=2,
        help="Number of consecutive validation windows required (default: 2).",
    )
    parser.add_argument(
        "--blocks-per-window",
        type=int,
        default=5,
        help="Number of contiguous blocks per validation window (default: 5).",
    )
    parser.add_argument(
        "--candidate-step",
        type=float,
        default=None,
        help="Optional candidate spacing in the model's time_unit.",
    )
    parser.add_argument(
        "--suggestion-output",
        default=None,
        help="Optional cutoff-diagnostic output prefix.",
    )
    parser.add_argument(
        "--analysis-output",
        default=None,
        help="Optional final numeric-analysis output prefix.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the cutoff-diagnostic plot; text and CSV are still written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run cutoff suggestion and, only on success, numeric analysis.

    Parameters
    ----------
    argv : sequence[str] or None, optional
        Arguments excluding the executable name. ``None`` uses
        ``sys.argv[1:]`` through :mod:`argparse`.

    Returns
    -------
    int
        Zero when a common cutoff is suggested and analyzed; two when no
        common plateau passes.
    """
    args = build_parser().parse_args(argv)
    model = Dfit.Dcov.load(args.model)
    suggestion = model.suggest_tc(
        validation_window=args.validation_window,
        min_tc=args.min_tc,
        relative_drift_tolerance=args.relative_drift_tolerance,
        q_tolerance=args.q_tolerance,
        persistence_windows=args.persistence_windows,
        blocks_per_window=args.blocks_per_window,
        candidate_step=args.candidate_step,
        fout_prefix=args.suggestion_output,
        make_plot=not args.no_plot,
    )
    if suggestion.tc is None:
        print("No common cutoff passed. Final analysis was not run.")
        return 2

    result = model.analysis(tc=suggestion.tc, fout_prefix=args.analysis_output)
    print(
        f"Final numeric analysis used tc={result.tc:.6g} {result.time_unit}; "
        f"equal-weight cluster mean D={result.across_clusters.mean:.6e} "
        f"{result.diffusion_unit}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
