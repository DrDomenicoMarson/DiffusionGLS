
import Dfit
import MDAnalysis as mda
import os
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MDAInputSpec:
    """Topology/trajectory pair used to build one MDAnalysis Universe.

    Parameters
    ----------
    tpr : str
        Path to topology file readable by MDAnalysis.
    xtc : str
        Path to trajectory file readable by MDAnalysis.
    """
    tpr: str
    xtc: str

def analyze_system(name, tpr, xtc, selection, tmax=100, tc=10):
    """Run end-to-end diffusion analysis for one MD system.

    Parameters
    ----------
    name : str
        Label used in console messages and output file names.
    tpr : str
        Path to topology file readable by MDAnalysis.
    xtc : str
        Path to trajectory file readable by MDAnalysis.
    selection : str
        Atom selection used to define molecules/residues for diffusion
        analysis.
    tmax : float, default=100
        Maximum lag time passed to :class:`Dfit.Dcov`, in ps.
    tc : float, default=10
        Lag time used in :meth:`Dfit.Dcov.analysis`, in ps.

    Returns
    -------
    None

    Notes
    -----
    Side effects:

    - Creates ``D_analysis_{name}.dat``, ``D_analysis_{name}.csv``, and the
      configured plot in the ``Example`` directory.
    - Prints progress and summary messages to stdout.
    """
    print(f"--- Analyzing {name} ---")
    print(f"Topology: {tpr}")
    print(f"Trajectory: {xtc}")
    
    # Load Universe
    u = mda.Universe(tpr, xtc)
    print(f"System: {len(u.atoms)} atoms")
    
    # Determine output path relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f"D_analysis_{name}")
    
    # Initialize Dfit with MDAnalysis Universe
    # We use the selection to define what constitutes a "molecule" (segment)
    # The reader iterates over RESIDUES in the selection.
    res = Dfit.Dcov(universe=u, selection=selection, tmax=tmax, fout=output_path)
    
    # Run the fit
    res.run_Dfit()
    
    # Analyze and plot
    res.analysis(tc=tc)
    print(f"Analysis complete. Output saved to D_analysis_{name}.dat, .csv, and .pdf\n")


def analyze_pooled_system(
    name: str,
    systems: Sequence[MDAInputSpec],
    selection: str,
    tmax: float = 100,
    tc: float | str = "auto",
):
    """Run cluster-aware diffusion analysis across independent replicas.

    Parameters
    ----------
    name : str
        Label used in console messages and output file names.
    systems : sequence[MDAInputSpec]
        Topology/trajectory inputs used to build Universes. Each Universe is an
        independent cluster by default. All trajectories must have matching
        frame count and timestep.
    selection : str
        Shared atom selection applied to each Universe.
    tmax : float, default=100
        Maximum lag time passed to :class:`Dfit.Dcov`, in ps.
    tc : float or 'auto', default='auto'
        Lag time used in :meth:`Dfit.Dcov.analysis`, in ps, or ``'auto'``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``systems`` is empty.
    """
    if len(systems) == 0:
        raise ValueError("At least one system is required for pooled analysis.")

    print(f"--- Pooled analysis: {name} ---")
    universes = []
    for i, spec in enumerate(systems):
        print(f"Replica {i}: topology={spec.tpr}, trajectory={spec.xtc}")
        universes.append(mda.Universe(spec.tpr, spec.xtc))
    print(f"Loaded {len(universes)} universes.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f"D_analysis_{name}")

    res = Dfit.Dcov(universes=universes, selection=selection, tmax=tmax, fout=output_path)
    res.run_Dfit()
    res.analysis(tc=tc)
    print(
        f"Cluster-aware analysis complete. Output saved to "
        f"D_analysis_{name}.dat, .csv, and .pdf\n"
    )


if __name__ == "__main__":
    # Define paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    traj_dir = os.path.join(base_dir, "trajs")
    
    # 1. Analyze Water
    # Usually water residues are named SOL, WAT, or TIP3. 
    # Since we want to analyze diffusion of whole molecules, we select them.
    # The Dfit MDAnalysisReader treats each RESIDUE in the selection as a molecule.
    analyze_system(
        name="Water",
        tpr=os.path.join(traj_dir, "WAT_32.tpr"),
        xtc=os.path.join(traj_dir, "traj_WAT.xtc"),
        selection="all", # Assumes the box is pure water
        tmax=50, # Adjust based on trajectory length
        tc=5
    )

    # 2. Analyze Oxygen
    analyze_system(
        name="Oxygen",
        tpr=os.path.join(traj_dir, "O2_32.tpr"),
        xtc=os.path.join(traj_dir, "traj_O2.xtc"),
        selection="all", # Assumes the box is pure oxygen
        tmax=50,
        tc=5
    )

    # Example pooled-replica usage (edit paths to your own replicas):
    # replicas = [
    #     MDAInputSpec(tpr="replica1.tpr", xtc="replica1.xtc"),
    #     MDAInputSpec(tpr="replica2.tpr", xtc="replica2.xtc"),
    # ]
    # analyze_pooled_system(name="Water_pooled", systems=replicas, selection="all", tmax=50, tc="auto")
