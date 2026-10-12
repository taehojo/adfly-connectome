# Third-party notices

## Drosophila_brain_model (Shiu et al. 2024)

<https://github.com/philshiu/Drosophila_brain_model>

No file of that repository is included here. What comes from it:

- `code/adfly/lif.py` and `code/adfly/house_nb.py` implement the same model equations and use
  its parameter values. They are new code, written for numba.
- The validation scripts (`code/scripts/validate_engine_flywire.py`,
  `validate_engine_final.py`, `validate_engine_200hz.py`, `rerun_brian2_sugarR.py`,
  `rerun_brian2_sugarR_rate.py`) use the stimulus definition of its sugar example (the FlyWire
  identifiers of 21 sugar-sensing neurons and of MN9). The two `rerun_brian2` scripts import
  and run the published code from a local clone.
- `results/validation/engine_validation_rates.npz` and `engine_validation_200hz_rates.npz`
  hold per-neuron firing rates computed from its two example outputs
  (`results/example/sugarR.parquet` and `sugarR_100Hz.parquet`), next to our own.
- `results/validation/shiu_example_provenance.json` quotes one line of its `model.py`, two
  calls and two sentences of its example notebook, and commit identifiers and messages.

The licence of that repository is reproduced below.

```
MIT License

Copyright (c) 2023 Philip Shiu and Nico Spiller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## drosophila-brain-mlx

<https://github.com/Kisame76/drosophila-brain-mlx>

No file of that repository is included here. `code/adfly/build_pack.py` is a much shorter
script (103 lines against 371) that follows the module `src/lif/compile_pack_malecns.py` of
that repository. Compared line by line on 10 October 2026, nine lines of `build_pack.py` are
also lines of that module (four imports, the two lines of the loop over record batches, the
main guard and two lines that hold only a quotation mark or a bracket), and thirteen more
are close to a line of it, differing in variable names or a few words: the two lines of the
docstring that state the rule, the sign table, the opening and the name of the weights
file, three counters, the dataset name, three fields of the build record and the line that
writes the record. That is 22 of the 87 lines of `build_pack.py` that are not blank.

`build_pack.py` follows that module in these things: the rule for selecting MaleCNS neurons
and edges (every annotated body with a non-null superclass, in order of body identifier;
edges whose presynaptic neuron has a consensus transmitter of acetylcholine, GABA or
glutamate, signed +1, −1, −1), the order of the steps, the wording of that rule in the
docstring, the names of three fields of the build record (`raw_rows`,
`dropped_endpoint_not_selected`, `dropped_unsigned_presynaptic`), and the dataset name
`male-cns-v1.0-superclass-non-null-known-nt`, which that module attributes to the
materialization published by flyBrain (<https://github.com/mehrantsi/flyBrain>). The field
names and the dataset name are in `connectome/manifest.json`. The rule gives the same
network size in both: 166,700 neurons and 24,469,412 signed edges.

The licence of that repository is reproduced below.

```
MIT License

Copyright (c) 2026 Kiyan Doguc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Connectome data

See `LICENSE-DATA.md` for MaleCNS v1.0 (CC BY 4.0) and FlyWire (CC BY-NC 4.0).
