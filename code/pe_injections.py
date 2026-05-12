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

import lal

from gwpy.table import EventTable

from parameter_estimation import sample_from_posterior


def main():

    ## CREATE EVENT DATA ##
    seed = 100

    # run directory
    rundir = '../data/pe_runs_lensing/injections'

    # check if this event already exists; if so, change the seed number
    def seed_exists(seed):
        return os.path.exists(os.path.join(rundir, f'GW{seed}'))
    while seed_exists(seed):
        seed += 1

    # the parameters that we want to set explicitly
    params_to_inject = {
        'm1' : 60,   # in Msun
        'm2' : 20,   # in Msun
        'iota' : 0.01,   # in radians
        'd_luminosity' : 500.   # Mpc
    }
    print(f"creating injection with seed {seed}", flush=True)
    event_data = create_injection(seed, params_to_inject)

    # approximant
    approximant = 'IMRPhenomXPHM'
    # prior class
    prior_class = 'IntrinsicLVCPrior'
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    # multiply the strain by i?
    multiply_by_i = True

    # merger time
    t_merger_guess = 0.
    # chirp mass: cheating but we can get this exactly
    mchirp = chirp_mass(params_to_inject['m1'], params_to_inject['m2'])
    print(f"chirp mass is {mchirp:.2f} Msun")

    run_parameter_estimation_injection(event_data, t_merger_guess, mchirp, rundir=rundir, multiply_by_i=multiply_by_i, verbose=True)


def run_parameter_estimation_injection(event_data, t_merger_guess, mchirp_guess,
                                approximant='IMRPhenomXPHM',
                                prior_class='IntrinsicLVCPrior',
                                rundir='../data/pe_runs',
                                multiply_by_i=False,
                                verbose=False):

    # optionally, multiply the strain by i, i.e. pick up a Type II phase
    if multiply_by_i:
        print(f"multiplying the strain by i", flush=True)
        event_data_ = event_data.reinstantiate(strain=1j*event_data.strain)
    else:
        event_data_ = event_data

    # compute the posterior
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

    sample_from_posterior(posterior, rundir, verbose=verbose)


def create_injection(eventname=None, seed=0, params_to_inject=None):

    aux_prior = cogwheel.gw_prior.LVCPrior(
        f_ref=100.0,
        mchirp_range=(10, 50),
        detector_pair='HL',
        tgps=0,
        ref_det_name='H',
        f_avg=100.0,
        d_hat_max=100.0,
        dt0=0.01,
    )

    injection_dic = {
        'f_ref' : 100.0,
        'm1' : 60,  # by convention, the heavier object. in Msun
        'm2' : 10,    # in Msun
        's1z' : 0.,
        's2z' : 0.,
        'psi' : 0.5,       # polarization angle
        'iota' : np.pi / 2 - 0.1,      # inclination angle
        's1x_n' : 0.,
        's1y_n' : 0.,
        's2x_n' : 0.,
        's2y_n' : 0.,
        'ra' : 3.,        # in radians
        'dec' : -0.5,       # in radians
        't_geocenter' : 0., # **can I just set this to zero?
        'phi_ref' : 0.43867,    # this is from a random sample from a LVCPrior. is this correct? does this change?
        'd_luminosity' : 1000.,  # in Mpc
        'l1' : 0.0,     # **what are these??
        'l2' : 0.0
    }

    # injection_dic = dict(aux_prior.generate_random_samples(1, seed=seed).loc[0, aux_prior.standard_params])

    if params_to_inject is not None:
        for key, value in params_to_inject.items():
            injection_dic[key] = value
    
    # then create the injection
    asd_funcs = list(cogwheel.data.ASDS)  # TODO: update to O4 PSD

    eventname = f'GW{seed}' if eventname is None else eventname
    event_data = cogwheel.data.EventData.gaussian_noise(
        eventname, duration=128.0, detector_names='HLV', fmax=512.0,
        asd_funcs=asd_funcs, tgps=0.0, seed=seed)

    event_data.inject_signal(par_dic=injection_dic, approximant='IMRPhenomXPHM')

    return event_data


def chirp_mass(m1, m2):
    """Returns the chirp mass given individual masses."""
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)


if __name__=='__main__':
    main()