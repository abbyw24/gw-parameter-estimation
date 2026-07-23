import os
os.environ['OMP_NUM_THREADS'] = '1'          # OpenMP (used by many C/Fortran libs)
os.environ['MKL_NUM_THREADS'] = '1'          # Intel MKL (numpy/scipy if built against MKL)
os.environ['OPENBLAS_NUM_THREADS'] = '1'     # OpenBLAS (numpy/scipy if built against OpenBLAS)
os.environ['NUMEXPR_NUM_THREADS'] = '1'      # numexpr (used internally by pandas sometimes)
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'   # Apple's Accelerate framework (macOS only, harmless elsewhere)
os.environ['NUMBA_NUM_THREADS'] = '1'        # numba, if you're using it (cogwheel's relative binning likelihood often does)

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from injections import run_parameter_estimation

logging.getLogger().setLevel(logging.ERROR)

SEED = 12
APPROXIMANT = 'IMRPhenomXPHM'
PARENTDIR = f'/home/abbye.williams/GWPE/data/injections/Halton/{APPROXIMANT}'

def run_one_injection(idx, params_to_inject):

    # arguments for the run
    eventname = f'GW{SEED}_{idx}'

    run_parameter_estimation(params_to_inject, PARENTDIR,
                             eventname=eventname, seed=SEED,
                             approximant=APPROXIMANT,
                             prior_class='IntrinsicLVCPrior',
                             lensed=True, verbose=True)

if __name__=='__main__':

    idxs_to_run = [0, 17, 19, 26, 40, 46, 64, 71, 73, 79, 91, 92, 93]
    nproc = len(idxs_to_run)
    print(f"Launching {nproc} processes", flush=True)

    # load the parameters to injection
    injection_params = pd.read_pickle(os.path.join(PARENTDIR,
                                             f'injection_params_{APPROXIMANT}.pkl'))

    # the data to submit
    
    # idx_start = 92
    # idxs_to_run = range(idx_start, idx_start + nproc)   # which set of parameters to run

    with ProcessPoolExecutor(max_workers=nproc) as executor:
        futures = {}

        for idx in idxs_to_run:
            params = injection_params.iloc[idx].to_dict()
            future = executor.submit(run_one_injection, idx, params)
            futures[future] = idx

        for i, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]

            try:
                future.result()
                print(f"FINISHED idx {idx} ({i}/{len(futures)})", flush=True)

            except Exception as e:
                print(f"FAILED idx {idx}: {repr(e)}", flush=True)
                raise
