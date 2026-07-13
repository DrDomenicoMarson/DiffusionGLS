# General outline
- we are writing a paper about the development of a tailored FF for PET and PEF (called QMD-FF), employing a new QM-based tool.
- the QM-based tool provides FF parameters for each polymer
- each polymer is composed by 100 monomer (repeating units)
- each box is composed by 32 polymer chains
- the paper will cover the development of these forcefield and the validation/check of their performance compared with OPLS-AA FF as reference
- The different properties that will be reported are:
    - density
    - glass transition temperature
    - elastic modulus / mechanical properties
        - the mechanical properties will be computed also from CG simulations, with CG FF developed based on both OPLS-AA and the new QMD-FF
    - diffusivity of molecular oxygen and water vapor
        - in each box, 32 molecules of O2 or water were added, prior to the annealing process
        - O2 was parametrized with OPLS-AA parameters, so it was not fine-tuned
        - Water was parametrized as SPC/E
        - the cross interactions (small molecule / polymer atoms) were computed with the OPLS-AA combination rules, even for the QMD-FF
- I'm responsible for the density and diffusivity MD calculations

- the diffusion calculations are a "more delicate thing" for the QMD-FF, compared to OPLS-AA:
    - the QMD-FF highliy tailors the LJ parameters to reporoduce polymer-specific descriptors
    - OPLS-AA in intrinsically a general FF, and it was developed to perform at least decently with its cross-interaction euqations
    - using "not-optimized/tailored" LJ parameters for the water/polymer and oxygen/polymer cross LJ interactions is a delicate point which should be addressed

# Diffusivity calculations
- the diffusivity calculation protocol were structured as:
    - minimization
    - annealing
    - density equilibration (20 ns)
    - 1000 ns of data collection, with data saved every 5 ps
    - unwrapping of the trajectory with cpptraj, as it implements the function to
        - unwrap the molecules in the fractional space
        - remove the effect of box fluctuations by projecting coordinates back into Cartesian space via averaged unit cell vectors
- diffusivity was computed based on the DiffusionGLS method reported in the reference paper (references/paper.pdf), as implemented in the current repository
    - m=10
    - the tc selected varied between each systems, ranging from 60 to 95 ns
- this protocol was repeated 3 times for each simulated system, obtaining:
    - 3 diffusivity values for each system, obtained from the 32 molecules in each box
        - these values were averaged and their variability was computed as the standard deviations between the replicates
    - a single diffusivity value for each system, obtained by pooling the 3x32 molecules from all simulations (hence the statistics is the SEM derived from the algorithm)


