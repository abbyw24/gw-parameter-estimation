import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import glob

import cogwheel.cosmology
import cogwheel.data
import cogwheel.posterior
import cogwheel.sampling
import cogwheel.gw_plotting
import cogwheel.gw_utils

import lal

from gwpy.table import EventTable


def main():
    
    # which catalog?
    catalog = 'GWTC-4.0'
    # look for events in this directory
    catalog_dir = f'/home/javier.roulet/cogwheel-catalog/processed_data/{catalog}' # f'../data/{catalog}'
    event_dir_list = glob.glob(f'{catalog_dir}/GW*')
    eventname_list = [
        event_dir.split('/')[-1] for event_dir in event_dir_list
    ]
    print(f"found {len(eventname_list)} events: ", flush=True)
    print(eventname_list, flush=True)

    for i, eventname in enumerate(eventname_list):
        print(f"starting PE for {eventname}...", flush=True)
        # get the corresponding event data from the open-data catalogue
        # *for now this is to get the "guess" chirp mass directly from the published chirp mass in the catalogue
        event_table = get_table_data(eventname, catalog)
        # check our input eventname against the name in the table
        eventname = check_eventname(eventname, event_table)

        # do we have a config for this event?
        config_fn = os.path.join(catalog_dir, eventname, 'event_data_kwargs.json') #'../data/event_configs.json'
        with open(config_fn) as file:
            event_configs = json.load(file)
        try:
            event_pars = event_configs[eventname]
        except KeyError:
            print(f"no config entry found for {eventname}; using default parameters to process timeseries data", flush=True)
            event_pars = {}

        t_merger_guess = 0.
        mchirp_guess = get_chirp_mass(event_table)

        print(f"detector-frame chirp mass guess = {mchirp_guess} Msun", flush=True)

        samples = run_parameter_estimation(eventname, t_merger_guess, mchirp_guess, event_pars=event_pars)

        # save the samples
        save_dir = os.path.join('../data', catalog, eventname) #os.path.join(event_dir_list[i], 'posterior_samples')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_fn = os.path.join(save_dir, 'posterior_samples')
        np.save(save_fn, samples)
        print(f"done! saved to {save_fn}\n", flush=True)


def run_parameter_estimation(eventname, t_merger_guess, mchirp_guess,
                                approximant = 'IMRPhenomXAS',
                                prior_class = 'IntrinsicAlignedSpinIASPrior',
                                event_pars = None,
                                verbose = True):

    """ get the event data """
    event_pars = event_pars or {}
    event_data = get_event_data(eventname, **event_pars)

    """ compute the posterior """
    if verbose:
        print(f"\ncomputing posterior...")
    # first we want to perform a fast likelihood maximization to find our reference waveform:
    posterior = cogwheel.posterior.Posterior.from_event(
        event_data,
        mchirp_guess,
        approximant,
        prior_class,
        ref_wf_finder_kwargs={
            'f_ref': 100.0,  # Just so it matches the injection and it makes sense to compare parameters
            'time_range': (t_merger_guess - 0.1, t_merger_guess + 0.1)  # Edit if needed
        }
    )

    """ sample the posterior """
    if verbose:
        print(f"\nsampling from posterior...", flush=True)
    # using Nautilus
    sampler = cogwheel.sampling.Nautilus(posterior)

    # these trade off quality and speed:
    sampler.run_kwargs['n_live'] = 1000
    sampler.run_kwargs['n_eff'] = 2000      # TODO: discuss with Javier, should I play around with these?

    # get the run directory
    rundir = sampler.get_rundir(parentdir='pe_runs')
    if verbose:
        print(f"rundir: {rundir}", flush=True)
    # run the sampling
    sampler.run(rundir)
    # get the samples
    samples = pd.read_feather(rundir/'samples.feather')

    return samples


def get_event_data(eventname, **kwargs):
    """
    Returns an event data object given an event name string.
    """
    filenames, detector_names, tgps = cogwheel.data.download_timeseries(eventname)
    event_data = cogwheel.data.EventData.from_timeseries(filenames, eventname, detector_names, tgps, **kwargs)
    return event_data


def check_eventname(eventname, event_table):
    """
    Checks an input eventname against the event name in a table (e.g. from `get_table_data()`) and, if needed, returns
    a corrected eventname to match the name in the table (so it matches the name in GWOSC).
    """
    eventname_in_table = event_table['name'].value[0].split('-')[0]
    if eventname_in_table != eventname:
        # if there is a version attached to the name in the event table (as 'GWxxxxxx_xxxxxx-vY', we need to remove this
        #    from the name used to get timeseries data
        corrected_eventname = eventname_in_table
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
        chirp_mass = (1 + event_table['redshift']) * event_table['chirp_mass_source']
    
    # unpack the actual number, which is slightly buried in this masked array setup
    return chirp_mass.value.data[0]


def add_derived_quantities(samples):
    """
    Add columns inplace to a dataframe of samples.

    Includes redshift, mtot, m1_source, m2_source, mtot_source, chieff, q.
    """
    samples['redshift'] = cogwheel.cosmology.z_of_d_luminosity(samples['d_luminosity'])

    samples['mtot'] = samples['m1'] + samples['m2']

    for mass_key in 'm1', 'm2', 'mtot':
        samples[f'{mass_key}_source'] = samples[mass_key] / (1 + samples['redshift'])

    samples['chieff'] = cogwheel.gw_utils.chieff(**samples[['m1', 'm2', 's1z', 's2z']])

    samples['q'] = samples['m2'] / samples['m1']

if __name__=='__main__':
    main()
