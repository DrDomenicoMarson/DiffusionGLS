# Review notes for `diffusion_DM.docx`

I revised the section to improve scientific consistency, manuscript tone, and internal coherence with the uploaded raw data and literature summary.

## Main fixes made
- tightened the pooled-vs-replica-averaged explanation and linked it explicitly to **Table DIFF**
- corrected awkward or misleading wording, especially the sentence that incorrectly referred to **density coefficients** instead of **diffusion coefficients**
- made the comparison with literature more cautious and more directly tied to the values in the uploaded spreadsheet
- clarified that the diffusivity comparison is a stringent **transferability / extrapolation test** for QMD-FF because penetrant/polymer cross Lennard-Jones terms were not specifically optimized
- removed the duplicated plain-text “Suggested Summary Table” block and kept a single formatted table
- inserted explicit in-text callouts to **Figure DiffX**, **Figure DiffY**, and **Table DIFF**

## Consistency check against uploaded data
The numerical values in the revised text are consistent with the uploaded spreadsheet:
- OPLS-AA / PEF / Water: 6.11 ± 0.68
- OPLS-AA / PEF / Oxygen: 2.53 ± 0.32
- OPLS-AA / PET / Water: 5.53 ± 0.53
- OPLS-AA / PET / Oxygen: 3.22 ± 0.41
- QMD-FF / PEF / Water: 1.88 ± 0.27
- QMD-FF / PEF / Oxygen: 7.92 ± 0.86
- QMD-FF / PET / Water: 1.98 ± 0.28
- QMD-FF / PET / Oxygen: 4.75 ± 0.52

The force-field comparison statements are also numerically consistent with the uploaded data:
- QMD-FF lowers water diffusivity by about **69% in PEF** and **64% in PET**
- QMD-FF increases oxygen diffusivity by a factor of about **3.1 in PEF** and by about **47% in PET**

## Figure and table placement recommendation
My recommendation is:

- **Figure DiffX:** keep in the **main text**. It is the primary comparison because it shows the absolute diffusivities and their relation to literature windows.
- **Figure DiffY:** move to the **Supporting Information** unless the paper wants to emphasize the failure to reproduce the PET/PEF contrast as a central result. It is useful, but it is a derived summary of the same information already contained in Figure DiffX.
- **Table DIFF:** move to the **Supporting Information**. It is valuable for completeness because it reports both pooled and replica-averaged values, but in the main text it would mostly duplicate the visual message of Figure DiffX.

## Best places to cite them in the text
Use the first in-text citations approximately as follows:

- **Figure DiffX:** in the paragraph where you first compare the absolute OPLS-AA and QMD-FF diffusivities against literature ranges.
- **Figure DiffY:** in the sentence where you discuss the **PET/PEF ratios** and the loss or reversal of the expected PEF suppression.
- **Table DIFF:** in the opening paragraph, when you first explain that both pooled and replica-averaged results are reported.

## Scientific interpretation note
The revised wording keeps the key message, but in a more defensible form:
- OPLS-AA is closer to the expected **oxygen ordering**
- QMD-FF gives **water diffusivities closer to the PET literature scale**
- neither force field reproduces the experimentally expected simultaneous suppression of both oxygen and water diffusion in PEF
- for QMD-FF, the absence of specifically optimized penetrant/polymer cross terms is a plausible and important limitation, but it should be framed as a **likely contributing factor**, not as a fully proven cause
