import os
import json
import glob

import cogwheel
import cogwheel.data
import cogwheel.posterior
import cogwheel.sampling
import cogwheel.utils
from cogwheel.likelihood.marginalization.coherent_score_lensing import CoherentScoreLensing

from gwpy.table import EventTable

 
def main():
    
    ## MAIN INPUTS ##

    # list of GW events
    catalog = 'O4_Discovery_Papers'
    # look for events in this directory
    catalog_dir = f'/home/javier.roulet/cogwheel-catalog/processed_data/{catalog}'
    event_dir_list = glob.glob(f'{catalog_dir}/GW*')
    eventname_list = [
        event_dir.split('/')[-1] for event_dir in event_dir_list
    ]

    print(f"found {len(eventname_list)} events: ", flush=True)
    print(eventname_list, flush=True)

    # approximant
    approximant = 'IMRPhenomXPHM'
    # prior class
    prior_class = 'IntrinsicLVCPrior'
    # run directory
    parentdir = '../data/pe_runs_lensing'
    os.makedirs(parentdir, exist_ok=True)

    for i, eventname in enumerate(eventname_list):
        print(f"starting PE for {eventname} ({i+1} of {len(eventname_list)})...", flush=True)
        # get the corresponding event data from the open-data catalogue
        # *for now this is to get the "guess" chirp mass directly from the published chirp mass in the catalogue
        event_table = get_table_data(eventname, catalog)
        # check our input eventname against the name in the table
        eventname = check_eventname(eventname, event_table)

        # do we have a config for this event?
        config_fn = os.path.join(catalog_dir, eventname, 'event_data_kwargs.json')
        try:
            with open(config_fn) as file:
                event_pars = json.load(file)
        except FileNotFoundError:
            print(f"config file not found at {config_fn}; using default event_pars", flush=True)
            event_pars = None

        t_merger_guess = -0.1 if eventname == 'GW231123_135430' else 0. # !!
        print(f"t_merger_guess = {t_merger_guess:.2f} sec", flush=True)
        mchirp_guess = get_chirp_mass(event_table)

        print(f"detector-frame chirp mass guess = {mchirp_guess:.2f} Msun", flush=True)
        
        # run the PE: the posterior and samples are automatically saved in the corresponding `parentdir``
        run_parameter_estimation(eventname, t_merger_guess, mchirp_guess, event_pars=event_pars,
                                    approximant=approximant, prior_class=prior_class, parentdir=parentdir,
                                    multiply_by_i=False, verbose=True)


def run_parameter_estimation(eventname, t_merger_guess, mchirp_guess,
                                approximant='IMRPhenomXPHM', # 'IMRPhenomXAS'
                                prior_class='IntrinsicLVCPrior', # 'IntrinsicAlignedSpinIASPrior'
                                event_pars=None,
                                parentdir='../data/pe_runs',
                                multiply_by_i=False,
                                verbose=False):

    """ get the event data """
    event_pars = event_pars or {}
    event_data = get_event_data(eventname, **event_pars)

    if event_data is None:
        print(f"problem with {eventname} event data. exiting", flush=True)
        return

    # multiply the strain by i ?!
    if multiply_by_i:
        print(f"multiplying the strain by i", flush=True)
        event_data_ = event_data.reinstantiate(strain=1j*event_data.strain)
    else:
        event_data_ = event_data

    """ compute the posterior """
    if verbose:
        print(f"\ncomputing posterior...", flush=True)
    # first we want to perform a fast likelihood maximization to find our reference waveform:
    #   (this doesn't include lensing yet)
    posterior = cogwheel.posterior.Posterior.from_event(
        event_data_,
        mchirp_guess,
        approximant,
        prior_class,
        ref_wf_finder_kwargs={
            'f_ref': 100.0,  # Just so it matches the injection and it makes sense to compare parameters
            'time_range': (t_merger_guess - 0.1, t_merger_guess + 0.1)  # Edit if needed
        }
    )

    # sample from the posterior
    sample_from_posterior(posterior, parentdir, continue_run=True, verbose=verbose)


def sample_from_posterior(posterior, parentdir, n_live=1000, n_eff=2000,
                          continue_run=True, verbose=False):

    # instantiate a CoherentScoreLensing object
    coherent_score = CoherentScoreLensing(**posterior.likelihood.coherent_score.get_init_dict())
    # reinstantiate the likelihood
    lensing_likelihood = posterior.likelihood.reinstantiate(coherent_score=coherent_score)

    # now we can use the same prior with the lensing likelihood to instantiate a new posterior object
    lensing_posterior = cogwheel.posterior.Posterior(prior=posterior.prior, likelihood=lensing_likelihood)

    """ sample the posterior (using Nautilus)"""
    # if we want to try to continue a run
    if continue_run:
        eventdir = lensing_posterior.get_eventdir(parentdir)
        status, sampler, rundir = find_existing_sampler(eventdir)
        if status == 'complete':
            print(f"samples exist at {rundir}. exiting", flush=True)
            return
        if status == 'found' and verbose:
            print(f"sampling from posterior: continuing run at {rundir}...", flush=True)
        elif status == 'not_found':
            continue_run = False
        else:
            print("unknown error. you shouldn't have reached this else block!", flush=True)
            assert False
    if not continue_run:
        sampler = cogwheel.sampling.Nautilus(lensing_posterior)
        rundir = sampler.get_rundir(parentdir=parentdir)
        if verbose:
            print(f"sampling from posterior: starting a new run at {rundir}...", flush=True)

    # these trade off quality and speed:
    sampler.run_kwargs['n_live'] = n_live
    sampler.run_kwargs['n_eff'] = n_eff
    # run the sampling (this saves the samples to rundir as a .feather)
    sampler.run(rundir)


def find_existing_sampler(eventdir, max_runs=10):
    if not os.path.exists(eventdir):
        return 'not_found', None, None

    for run_id in range(max_runs):
        rundir = os.path.join(eventdir, f'run_{run_id}')
        sampler_fn = os.path.join(rundir, 'Sampler.json')

        if not os.path.exists(sampler_fn):
            continue

        if os.path.exists(os.path.join(rundir, 'samples.feather')):
            return 'complete', None, rundir

        sampler = cogwheel.utils.read_json(sampler_fn)
        return 'found', sampler, rundir

    return 'not_found', None, None


def get_event_data(eventname, **kwargs):
    """
    Returns event timeseries data from cogwheel given an event name string.
    """
    filenames, detector_names, tgps = cogwheel.data.download_timeseries(eventname)
    try:
        return cogwheel.data.EventData.from_timeseries(filenames, eventname, detector_names, tgps, **kwargs)
    except OSError as e: # !!
        print(f"OSError ({eventname}): ", e)
        print("\tfilenames: ", filenames)
        return None
    # except ValueError as e:
    #     print(f"ValueError ({eventname}): ", e)


def check_eventname(eventname, event_table, verbose=False):
    """
    Checks an input eventname against the event name in a table (e.g. from `get_table_data()`) and, if needed, returns
    a corrected eventname to match the name in the table (so it matches the name in GWOSC).
    """
    eventname_in_table = event_table['name'].value[0].split('-')[0]
    if eventname_in_table != eventname:
        # if there is a version attached to the name in the event table (as 'GWxxxxxx_xxxxxx-vY', we need to remove this
        #    from the name used to get timeseries data
        corrected_eventname = eventname_in_table
        if verbose:
            print(f"found event for {corrected_eventname}; updating eventname from {eventname}", flush=True)
        eventname = corrected_eventname
    assert eventname == eventname_in_table

    return eventname


def get_table_data(eventname, catalog):
    events_table = EventTable.fetch_open_data(catalog)
    idx = [eventname in x for x in events_table['name']]
    event_table = events_table[idx]
    assert len(event_table) == 1, f"{len(event_table)} events found for {eventname} in {catalog}; please refine eventname"

    return event_table

def get_chirp_mass(event_table):

    # try to get the (detector-frame) chirp mass
    chirp_mass = event_table['chirp_mass']
    # this is a masked numpy array; get just the value
    # some of these are reported as None:
    #   if this is the case, calculate the detector-frame chirp mass from the source chirp mass + redshift
    if chirp_mass == None:
        try:
            chirp_mass = (1 + event_table['redshift']) * event_table['chirp_mass_source']
        except TypeError:
            print(f"couldn't get redshift information. defaulting to source-frame chirp mass")
            chirp_mass = event_table['chirp_mass_source']
    # unpack the actual number, which is slightly buried in this masked array setup
    return chirp_mass.value.data[0]


if __name__=='__main__':
    main()
