import os
os.environ['OMP_NUM_THREADS'] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from injections import run_parameter_estimation

SEED = 12
APPROXIMANT = 'IMRPhenomXPHM'
RUNDIR = f'/home/abbye.williams/GWPE/data/injections/{APPROXIMANT}'

def run_one_injection(idx, params_to_inject):

    # arguments for the run
    eventname = f'GW{SEED}_{idx}'

    run_parameter_estimation(params_to_inject, RUNDIR,
                             eventname=eventname, seed=SEED,
                             approximant=APPROXIMANT,
                             prior_class='IntrinsicLVCPrior',
                             multiply_by_i=True, verbose=False)

if __name__=='__main__':

    nproc = 16
    print(f"Launching {nproc} processes", flush=True)

    # load the parameters to injection
    injection_params = pd.read_pickle(os.path.join(RUNDIR,
                                             f'injection_params_{APPROXIMANT}.pkl'))
    
    # the data to submit
    idx_start = 40
    idxs_to_run = range(idx_start, idx_start + nproc)   # which set of parameters to run

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
                print(f"Finished idx={idx} ({i}/{len(futures)})", flush=True)

            except Exception as e:
                print(f"FAILED idx={idx}: {repr(e)}", flush=True)
                raise