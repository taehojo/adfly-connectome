"""Re-run the unmodified Shiu et al. Brian2 code (sugarR, FlyWire v630) on this machine.

  python code/scripts/rerun_brian2_sugarR.py [path to a clone of Drosophila_brain_model]

Needs Brian2 2.10.1 and a clone of https://github.com/philshiu/Drosophila_brain_model
(default location: ext/Drosophila_brain_model). The run uses the defaults of the published
code (150-Hz input, 30 trials of 1 s) and writes
results/validation/brian2_rerun/sugarR.parquet, the local reference of
validate_engine_final.py.

The published code sets no seed, so every run is a new Poisson realisation. The file stored
in the repository is the realisation the manuscript reports; it is not overwritten. To make
a new one, move the stored file away first.
"""
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
D = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "ext/Drosophila_brain_model"
OUT = ROOT / "results/validation/brian2_rerun"
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
         720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
         720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
         720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]

if __name__ == "__main__":      # the published run_exp starts worker processes
    if (OUT / "sugarR.parquet").exists():
        sys.exit(f"{OUT / 'sugarR.parquet'} exists; not overwritten")
    sys.path.insert(0, str(D))
    from model import run_exp  # the published model code, unmodified

    t = time.time()
    run_exp(exp_name='sugarR', neu_exc=SUGAR, path_res=str(OUT),
            path_comp=str(D / '2023_03_23_completeness_630_final.csv'),
            path_con=str(D / '2023_03_23_connectivity_630_final.parquet'), n_proc=30, force_overwrite=False)
    print('wall', time.time() - t)
