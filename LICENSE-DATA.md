# Licence for the result tables and figures

The files in `results/`, `figures/` and `connectome/` are (c) 2026 Taeho Jo and are released
under the Creative Commons Attribution 4.0 International licence (CC BY 4.0),
<https://creativecommons.org/licenses/by/4.0/>, with the exception below. The code in `code/`
is released under the MIT licence (`LICENSE`).

To give credit, cite the manuscript named in `README.md` and `CITATION.cff`.

## Material derived from third-party connectomes

**MaleCNS v1.0** (FlyEM at HHMI Janelia Research Campus, the University of Cambridge, the MRC
Laboratory of Molecular Biology and Google Research) is released under CC BY 4.0. Some tables
here carry MaleCNS body identifiers, cell type names, side and superclass annotations and
synapse-count summaries: the neuron tables and the hub rank file in `results/validation/`,
the files in `results/exp1/`, `results/calibration/matched_type_input_synapses.csv` and
`connectome/manifest.json`. They were made from the three source files named in
`connectome/manifest.json` by the selection and signing rules of `code/adfly/build_pack.py`
and by simulation. When you reuse them, also cite Berg S, et al. Sexual dimorphism in the
complete *Drosophila* male central nervous system connectome. Cell. 2026;189:5504-5526.e15.
doi:10.1016/j.cell.2026.08.015.

**FlyWire** data are released under CC BY-NC 4.0
(<https://creativecommons.org/licenses/by-nc/4.0/>). Six files here are derived from FlyWire
tables as distributed with the model of Shiu et al. (2024) and are released under CC BY-NC 4.0,
not CC BY 4.0:

- `results/calibration/matched_type_input_synapses.csv` (mean input synapses per neuron for
  cell types matched between FlyWire v783 and MaleCNS v1.0)
- `results/validation/brian2_rerun/sugarR.parquet`,
  `results/validation/brian2_rerun_100hz/sugarR.parquet` and
  `results/validation/brian2_rerun_200hz/sugarR.parquet` (spike times by FlyWire identifier
  from our runs of the published Brian2 model on FlyWire v630)
- `results/validation/engine_validation_rates.npz` and
  `results/validation/engine_validation_200hz_rates.npz` (per-neuron firing rates on FlyWire
  v630 from our engine, from our Brian2 runs and from the example outputs distributed with
  the published model; the second file also holds the FlyWire identifiers of the 21
  stimulated neurons and of MN9)

When you reuse them, also cite Dorkenwald S, et al. Neuronal wiring diagram of an adult brain.
Nature. 2024;634:124-138. doi:10.1038/s41586-024-07558-y; Schlegel P, et al. Whole-brain
annotation and multi-connectome cell typing of *Drosophila*. Nature. 2024;634:139-152.
doi:10.1038/s41586-024-07686-5; and Shiu PK, et al. A *Drosophila* computational brain model
reveals sensorimotor processing. Nature. 2024;634:210-219. doi:10.1038/s41586-024-07763-9.

No connectome table is redistributed in this repository.
