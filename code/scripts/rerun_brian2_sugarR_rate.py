"""Re-run the unmodified Shiu et al. Brian2 code (sugarR, FlyWire v630) at a chosen input rate.

  python scripts/rerun_brian2_sugarR_rate.py [rate_hz]        (default 200)

The released model.py sets the Poisson rate through its parameter dictionary
(default_params['r_poi'] = 150 Hz). Here a copy of that dictionary with r_poi set
to the requested rate is passed to the published run_exp(), which is how the
published example notebook changes the rate (its 100-Hz example). No published
file is edited. Like the published code, the run is not seeded: every trial is one
unseeded Poisson realisation in its own process.

Writes results/validation/brian2_rerun_<rate>hz/sugarR.parquet and, next to it,
run_info.json (versions, rate, wall time). An existing output is never overwritten
(force_overwrite=False): the script then stops without running.

On the cluster we used, the interpreter of the python/3.13.5 module carried an RPATH to the
libstdc++ of GCC 9.3.0, older than the compiler (GCC 14.2.0) that builds Brian2's
Cython code, so the compiled code objects fail to load ("GLIBCXX_3.4.32 not found")
unless the newer library is loaded first:

  LD_PRELOAD=<libstdc++.so.6 of GCC 14.2.0> python code/scripts/rerun_brian2_sugarR_rate.py 200
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"   # leave the published code folder untouched (workers too)
sys.dont_write_bytecode = True
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
EXT = ROOT / "ext/Drosophila_brain_model"
sys.path.insert(0, str(EXT))

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
         720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
         720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
         720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]
N_PROC = 30

if __name__ == "__main__":
    import brian2
    from brian2 import Hz
    from model import default_params, run_exp

    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
    out = ROOT / f"results/validation/brian2_rerun_{rate:g}hz"
    if (out / "sugarR.parquet").exists():
        sys.exit(f"{out / 'sugarR.parquet'} exists; not overwritten")
    out.mkdir(parents=True, exist_ok=True)

    params = dict(default_params)           # a copy: the published defaults stay as released
    released_rate = float(default_params["r_poi"] / Hz)
    params["r_poi"] = rate * Hz

    t = time.time()
    run_exp(exp_name="sugarR", neu_exc=SUGAR, path_res=str(out),
            path_comp=str(EXT / "2023_03_23_completeness_630_final.csv"),
            path_con=str(EXT / "2023_03_23_connectivity_630_final.parquet"),
            params=params, n_proc=N_PROC, force_overwrite=False)
    wall = time.time() - t
    info = {"code": "ext/Drosophila_brain_model (unmodified); run_exp(exp_name='sugarR', "
                    "params=copy of default_params with r_poi set)",
            "r_poi_hz": rate, "r_poi_released_default_hz": released_rate,
            "n_run": int(params["n_run"]), "t_run_ms": float(params["t_run"] / brian2.ms),
            "brian2": brian2.__version__, "codegen_target": str(brian2.prefs.codegen.target),
            "python": sys.version.split()[0], "n_proc": N_PROC, "seeded": False,
            "ld_preload": os.environ.get("LD_PRELOAD", ""), "wall_s": round(wall, 1)}
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
