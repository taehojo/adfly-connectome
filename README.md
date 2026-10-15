# adfly-connectome

Code, result tables and figures of an in silico model of amyloid-like pathology on the
*Drosophila* MaleCNS connectome.

**Manuscript:** Jo T. Simulating Alzheimer's amyloid-like pathology on a fly central nervous system connectome. arXiv:2610.19153, 2026. <https://arxiv.org/abs/2610.19153>

**Interactive demo:** <https://www.jolab.ai/adfly/>

[![The demo: the virtual arena with healthy (blue) and damaged (orange) flies during a looming threat, and the neurons and connections of one healthy and one damaged fly](media/demo.gif)](https://www.jolab.ai/adfly/)

## What this is

The model simulates the central nervous system of the adult male fruit fly: 166,700
integrate-and-fire neurons wired by the MaleCNS v1.0 connectome. Three kinds of damage
reported in amyloid-β flies, a lowered spike threshold, synapse loss and neuron loss, are
imposed with assumed magnitudes, and circuit responses and simple behaviour in a virtual
arena are measured in 1,424 simulated flies. Everything here comes from simulation; no
animal was used. The methods, results and limitations are described in the manuscript.

The demo plays a separate, recorded 90-s run of 20 healthy and 20 damaged flies for
illustration; it is not part of the manuscript's analyses.

## Contents

| Path | Contents |
|---|---|
| `code/adfly/` | The model: network build, simulation engine, pathology rules, virtual arena. |
| `code/scripts/` | Scripts that run the simulations, analyse them, compute the quoted values and draw the figures. |
| `results/` | Per-fly result tables of every design and the analysis outputs. `results/facts/` holds every value quoted in the manuscript, each with a one-line definition. |
| `figures/` | The figures of the manuscript (Fig 1 to Fig 7, S1 Fig to S4 Fig). |
| `connectome/` | Checksums of the connectome files and of the built network (the connectome itself is not redistributed). |
| `media/` | The demo preview shown above. |

Release `v1.0.0` is the state of the code and tables at submission.

## Installation

Python 3.13 and the packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

`requirements-simulation.txt` lists the environment in which the simulations ran.

## Reproducing the values and figures (no connectome needed)

The values quoted in the manuscript are recomputed from the stored result tables:

```bash
python code/scripts/facts_model.py
python code/scripts/facts_battery.py
python code/scripts/facts_detection.py
python code/scripts/facts_exp1.py
python code/scripts/facts_exp2.py
python code/scripts/facts_extra.py
python code/scripts/facts_validation.py
python code/scripts/facts_text.py      # run last
```

Each script writes one table to `results/facts/`; after a rerun, `git diff results/` should
be empty. To find the value behind a statement in the manuscript, search the `_meaning`
entries of these tables.

The figures:

```bash
python code/scripts/make_figures.py       # Figs 1, 2, 3, 6 and 7, S1 Fig to S4 Fig
python code/scripts/make_fig4_burden.py   # Fig 4
python code/scripts/make_fig5_rescue.py   # Fig 5
```

## Rerunning the simulations

The simulations need the MaleCNS v1.0 connectome (CC BY 4.0). Download these three files
from <https://male-cns.janelia.org/download/> into `data/malecns_v1.0/`:

- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `body-neurotransmitters-male-cns-v1.0.feather`
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

and build the network (166,700 neurons, 24,469,412 signed connections):

```bash
python code/adfly/build_pack.py
python code/adfly/build_mb_edges.py
```

Then each design is one command; the default seeds repeat the reported runs:

```bash
# circuit assays: severity series (384 flies), rule ablation (320), map comparison (192), Kenyon-cell-only (64)
python code/scripts/assay_battery.py --design stage_curve   --scale 4 --out stage_curve_x4.csv
python code/scripts/assay_battery.py --design components    --scale 4 --out components_x4.csv
python code/scripts/assay_battery.py --design vulnerability --scale 4 --out vulnerability_x4.csv
python code/scripts/assay_battery.py --design scope         --scale 4 --out scope_x4.csv

# circuit assays and 120 s in the virtual arena in the same flies (176 flies)
python code/scripts/cohort_joint.py --stage-list "0:32,0.05:24,0.1:24,0.15:24,0.2:24,0.3:24,0.5:24" --seconds 120 --out cohort_joint.csv

# maps compared at matched synaptic burden (288 flies)
python code/scripts/exp2_burden_matched.py calibrate
python code/scripts/exp2_burden_matched.py run --n 32

# within-fly analysis: damage undone part by part in 48 flies
python code/scripts/exp1_counterfactual.py screen --stage 0.5 --n-flies 24 --batch 64
python code/scripts/exp1_counterfactual.py screen --stage 0.3 --n-flies 24 --batch 64
python code/scripts/exp1_counterfactual.py robust  --stage 0.5 --n-flies 12 --batch 64
python code/scripts/exp1_counterfactual.py confirm --stage 0.5 --n-flies 12 --batch 16
```

These runs are long: from minutes to more than a day on 48 cores. The analysis scripts
(`analyze_battery.py`, `analyze_joint.py`, `analyze_exp1.py`, `analyze_exp2.py`) read the
stored tables and need no connectome.

The engine was validated against the published fly brain model of Shiu et al. (2024) on
FlyWire v630 (rate correlation 0.9997); `validate_engine_final.py` and
`validate_engine_200hz.py` repeat this and need the tables of that model's repository,
<https://github.com/philshiu/Drosophila_brain_model>, in `data/shiu2024/`.

To try the code in seconds without the connectome, `make_test_pack.py` writes a small
synthetic network with the same file layout:

```bash
python code/scripts/make_test_pack.py --out data/pack/malecns_v1.0
python code/scripts/assay_battery.py --design scope --out smoke_scope.csv
```

Delete `data/pack/malecns_v1.0/` before building the real network.

## How the code was written

The code was written with an AI coding assistant (Claude, Anthropic, used through Claude
Code) under the author's direction, with Claude API credits provided through Anthropic's AI
for Science program. The author defined the model, the designs and the analyses, and is
responsible for the code and the results.

## Licence

The code is released under the MIT licence (`LICENSE`). The result tables, figures and
records are released under CC BY 4.0, except six files derived from FlyWire tables, which
keep its non-commercial term (`LICENSE-DATA.md`). Third-party sources and their licences are
listed in `THIRD_PARTY_NOTICES.md` and `LICENSE-DATA.md`: MaleCNS v1.0 (Berg et al., Cell 2026; CC BY 4.0), the base
model of Shiu et al. (Nature 2024; MIT), FlyWire (Dorkenwald et al. and Schlegel et al.,
Nature 2024; CC BY-NC 4.0), and the network selection rule of
<https://github.com/Kisame76/drosophila-brain-mlx> (MIT).

## Citation

Please cite the manuscript above; `CITATION.cff` has the details.

## Contact

Taeho Jo, Indiana University School of Medicine (tjo@iu.edu), or open an issue on this
repository.
