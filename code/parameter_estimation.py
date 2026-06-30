import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import glob

import cogwheel
import cogwheel.cosmology
import cogwheel.data
import cogwheel.posterior
import cogwheel.sampling
import cogwheel.gw_plotting
import cogwheel.gw_utils
from cogwheel.likelihood.marginalization.coherent_score_lensing import CoherentScoreLensing
from pesummary.io import read

import lal

from gwpy.table import EventTable

 
def main():
    
    ## MAIN INPUTS ##

    # list of GW events
    catalog = 'O4_Discovery_Papers'
    # look for events in this directory
    catalog_dir = f'/home/javier.roulet/cogwheel-catalog/processed_data/{catalog}'
    # event_dir_list = glob.glob(f'{catalog_dir}/GW*')
    # eventname_list = [
    #     event_dir.split('/')[-1] for event_dir in event_dir_list
    # ]

    eventname_list = [
        'GW241110_124123'
    ]

    # !! hacky
    event_to_remove = 'GW231123_135430'
    if event_to_remove in eventname_list:
        eventname_list.remove(event_to_remove)

    print(f"found {len(eventname_list)} events: ", flush=True)
    print(eventname_list, flush=True)

    # approximant
    approximant = 'IMRPhenomXPHM'
    # prior class
    prior_class = 'IntrinsicLVCPrior'
    # run directory
    rundir = '../data/pe_runs_lensing'
    os.makedirs(rundir, exist_ok=True)

    # skip existing samples?
    skip_existing = True

    for i, eventname in enumerate(eventname_list):
        print(f"starting PE for {eventname} ({i+1} of {len(eventname_list)})...", flush=True)
        # get the corresponding event data from the open-data catalogue
        # *for now this is to get the "guess" chirp mass directly from the published chirp mass in the catalogue
        event_table = get_table_data(eventname, catalog)
        # check our input eventname against the name in the table
        eventname = check_eventname(eventname, event_table)

        # if skip_existing is True, get the run directory to see if we've already performed PE on this event
        # **TODO: this is clunky: potential mismatch between this save_dir and the one used by run_parameter_estimation()
        if skip_existing == True:
            save_dir = os.path.join(rundir, prior_class, eventname)
            samples_list = glob.glob(os.path.join(save_dir, f'*/samples.feather'))
            # if the list is not empty, don't run parameter estimation since it's already been done !
            if samples_list:
                print(f"posterior samples found for {eventname} at {samples_list[0]} and skip_existing is True. skipping this event")
                continue

        # do we have a config for this event?
        config_fn = os.path.join(catalog_dir, eventname, 'event_data_kwargs.json')
        try:
            with open(config_fn) as file:
                event_pars = json.load(file)
        except FileNotFoundError:
            print(f"config file not found at {config_fn}; using default event_pars")
            event_pars = None

        t_merger_guess = -0.1 if eventname == 'GW231123_135430' else 0. # !!
        print(f"t_merger_guess = {t_merger_guess:.2f} sec", flush=True)
        mchirp_guess = get_chirp_mass(event_table)

        print(f"detector-frame chirp mass guess = {mchirp_guess:.2f} Msun", flush=True)
        
        # run the PE: the posterior and samples are automatically saved in the corresponding `rundir`
        samples = run_parameter_estimation(eventname, t_merger_guess, mchirp_guess, event_pars=event_pars,
                                            approximant=approximant, prior_class=prior_class, rundir=rundir,
                                            multiply_by_i=False, verbose=True)


def run_parameter_estimation(eventname, t_merger_guess, mchirp_guess,
                                approximant='IMRPhenomXPHM', # 'IMRPhenomXAS'
                                prior_class='IntrinsicLVCPrior', # 'IntrinsicAlignedSpinIASPrior'
                                event_pars=None,
                                rundir='../data/pe_runs',
                                skip_existing=False,
                                multiply_by_i=False,
                                verbose=False):

    """ get the event data """
    event_pars = event_pars or {}
    event_data = get_event_data(eventname, **event_pars)

    # multiply the strain by i ?!
    if multiply_by_i:
        print(f"multiplying the strain by i", flush=True)
        event_data_ = event_data.reinstantiate(strain=1j*event_data.strain)
    else:
        event_data_ = event_data

    """ compute the posterior """
    if verbose:
        print(f"\ncomputing posterior...")
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
    sample_from_posterior(posterior, rundir, verbose=verbose)


def sample_from_posterior(posterior, rundir, n_live=1000, n_eff=2000, verbose=False):

    # instantiate a CoherentScoreLensing object
    coherent_score = CoherentScoreLensing(**posterior.likelihood.coherent_score.get_init_dict())
    # reinstantiate the likelihood
    lensing_likelihood = posterior.likelihood.reinstantiate(coherent_score=coherent_score)

    # now we can use the same prior with the lensing likelihood to instantiate a new posterior object
    lensing_posterior = cogwheel.posterior.Posterior(prior=posterior.prior, likelihood=lensing_likelihood)

    """ sample the posterior """
    if verbose:
        print(f"\nsampling from posterior...", flush=True)
    # using Nautilus
    sampler = cogwheel.sampling.Nautilus(lensing_posterior)

    # these trade off quality and speed:
    sampler.run_kwargs['n_live'] = n_live
    sampler.run_kwargs['n_eff'] = n_eff

    # get the run directory
    rundir = sampler.get_rundir(parentdir=rundir)
    if verbose:
        print(f"rundir: {rundir}", flush=True)
    # run the sampling (this saves the samples to rundir as a .feather)
    sampler.run(rundir)


def get_event_data(eventname, **kwargs):
    """
    Returns event timeseries data from cogwheel given an event name string.
    """
    filenames, detector_names, tgps = cogwheel.data.download_timeseries(eventname)
    event_data = cogwheel.data.EventData.from_timeseries(filenames, eventname, detector_names, tgps, **kwargs)
    return event_data


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
