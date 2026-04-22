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

# Diffusivity calculations
- the diffusivity calculation protocol were structured as:
    - minimization
    - annealing
    - density equilibration (20 ns)
    - 1000 ns of data collection, with data saved every 5 ps
- diffusivity was computed based on the DiffusionGLS method reported in the reference paper (references/paper.pdf), as implemented in the current repository
- this protocol was repeated 3 times for each simulated system, obtaining:
    - 3 diffusivity values for each system, obtained from the 32 molecules in each box
        - these values were averaged and their variability was computed as the standard deviations between the replicates
    - a single diffusivity value for each system, obtained by pooling the 3x32 molecules from all simulations (hence the statistics is the SEM derived from the algorithm)



