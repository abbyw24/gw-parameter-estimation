import os
os.environ['OMP_NUM_THREADS'] = '1'          # OpenMP (used by many C/Fortran libs)
os.environ['MKL_NUM_THREADS'] = '1'          # Intel MKL (numpy/scipy if built against MKL)
os.environ['OPENBLAS_NUM_THREADS'] = '1'     # OpenBLAS (numpy/scipy if built against OpenBLAS)
os.environ['NUMEXPR_NUM_THREADS'] = '1'      # numexpr (used internally by pandas sometimes)
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'   # Apple's Accelerate framework (macOS only, harmless elsewhere)
os.environ['NUMBA_NUM_THREADS'] = '1'        # numba, if you're using it (cogwheel's relative binning likelihood often does)

import json
import glob
# import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from parameter_estimation import run_parameter_estimation, get_table_data, check_eventname, get_chirp_mass, find_existing_sampler

# logging.getLogger().setLevel(logging.ERROR)

CATALOG = 'GWTC-5.0'
# look for events in this directory
CATALOG_DIR = f'/home/javier.roulet/cogwheel-catalog/processed_data/{CATALOG}'
APPROXIMANT = 'IMRPhenomXPHM'
PRIOR_CLASS = 'IntrinsicLVCPrior'
PARENTDIR = f'/home/abbye.williams/GWPE/data/pe_runs_lensing'

def run_one_event(eventname):

    # get the corresponding event data from the open-data catalogue
    # *for now this is to get the "guess" chirp mass directly from the published chirp mass in the catalogue
    event_table = get_table_data(eventname, CATALOG)
    # check our input eventname against the name in the table
    eventname = check_eventname(eventname, event_table)

    # do we have a config for this event?
    config_fn = os.path.join(CATALOG_DIR, eventname, 'event_data_kwargs.json')
    try:
        with open(config_fn) as file:
            event_pars = json.load(file)
    except FileNotFoundError:
        print(f"config file not found at {config_fn}; using default event_pars", flush=True)
        event_pars = None

    t_merger_guess = 0.
    # print(f"t_merger_guess = {t_merger_guess:.2f} sec", flush=True)
    mchirp_guess = get_chirp_mass(event_table)

    # print(f"detector-frame chirp mass guess = {mchirp_guess:.2f} Msun", flush=True)

    # run the PE: the posterior and samples are automatically saved in the corresponding `rundir`
    run_parameter_estimation(eventname, t_merger_guess, mchirp_guess, event_pars=event_pars,
                                approximant=APPROXIMANT, prior_class=PRIOR_CLASS, parentdir=PARENTDIR,
                                multiply_by_i=False, verbose=True)

if __name__=='__main__':

    # events to run
    event_dir_list = glob.glob(f'{CATALOG_DIR}/GW*')
    eventname_list = [
        event_dir.split('/')[-1] for event_dir in event_dir_list
    ]
    print(f"Found {len(eventname_list)} events in {CATALOG_DIR}", flush=True)

    # ens = ['GW241129_021832', 'GW241129_021832']
    # for en in ens:
    #     eventname_list.remove(en)
    #     print(f"removed {en} from the event list", flush=True)

    # which of these already have samples?
    events_to_run = eventname_list.copy()
    ncomplete = 0
    for eventname in eventname_list:
        eventdir = os.path.join(PARENTDIR, PRIOR_CLASS, eventname)
        status, _, _ = find_existing_sampler(eventdir)
        if status == 'complete':
            ncomplete += 1
            events_to_run.remove(eventname)
    
    print(f"{ncomplete} events have completed PE.", flush=True)

    nproc = len(events_to_run)
    print(f"Launching {nproc} processes", flush=True)

    with ProcessPoolExecutor(max_workers=nproc) as executor:
        futures = {}

        for idx, eventname in enumerate(events_to_run):
            future = executor.submit(run_one_event, eventname)
            futures[future] = eventname

        for i, future in enumerate(as_completed(futures), start=1):
            eventname = futures[future]

            try:
                future.result()
                print(f"FINISHED {eventname} ({i}/{len(futures)})", flush=True)

            except Exception as e:
                print(f"FAILED {eventname}: {repr(e)}", flush=True)
                raise
