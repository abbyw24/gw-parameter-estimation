"""
Generate GW injection data and run PE.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import cogwheel.data
from parameter_estimation import sample_from_posterior

def run_parameter_estimation(params_to_inject, rundir, eventname=None, seed=None,
                             approximant='IMRPhenomXPHM',
                             prior_class='IntrinsicLVC',
                             multiply_by_i=True, verbose=False):

    event_data = create_injection(params_to_inject, eventname=eventname, seed=seed,
                                  approximant=approximant)
    injection_dict = event_data.get_init_dict()['injection']
    injected_par_dict = injection_dict['par_dic']

    # get the chirp mass and merger time estimates
    mchirp = cogwheel.gw_utils.m1m2_to_mchirp(injected_par_dict['m1'], injected_par_dict['m2'])
    t_merger_guess = 0.

    # optionally, multiply the strain by i, i.e. pick up a Type II phase
    #   and add a flag in the injection dictionary for if we've lensed the signal
    if multiply_by_i:
        print(f"multiplying the strain by i", flush=True)
        event_data_ = event_data.reinstantiate(strain=1j*event_data.strain)
        injection_dict['lensed'] = True
    else:
        event_data_ = event_data
        injection_dict['lensed'] = False

    # compute the posterior
    posterior = cogwheel.posterior.Posterior.from_event(
        event_data_,
        mchirp,
        injection_dict['approximant'],
        prior_class,
        ref_wf_finder_kwargs={
            'f_ref': 100.0,  # Just so it matches the injection and it makes sense to compare parameters
            'time_range': (t_merger_guess - 0.1, t_merger_guess + 0.1)  # Edit if needed
        }
    )

    sample_from_posterior(posterior, rundir, verbose=verbose)


def create_injection(params_to_inject=None, eventname=None, seed=None,
                     approximant='IMRPhenomXPHM'):
    """
    Create GW injection data and return it as an `EventData` object.

    Parameters
    ----------
    params_to_inject : dict or None, optional
        Dictionary of parameters to inject. Overwrites the defaults.
    eventname : str or None, optional
        Optional eventname. If `None`, eventname is set as 'GW{seed}'.
    seed : int or None, optional
        Optional seed for the Gaussian noise realization.
    approximant : str, optional
        Approximant waveform. Default is 'IMRPhenomXPHM'.
    
    Returns
    -------
    event_data : EventData object

    """

    injection_dic = {
        'f_ref' : 100.0,
        'm1' : 60,  # by convention, the heavier object. in Msun
        'm2' : 10,    # in Msun
        's1z' : 0.,
        's2z' : 0.,
        'psi' : 0.5,       # polarization angle
        'iota' : np.pi / 2,      # inclination angle
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

    if params_to_inject is not None:
        for key, value in params_to_inject.items():
            injection_dic[key] = value

    if eventname is None:
        eventname = f'GW{seed}' if seed is not None else 'GW'

    event_data = cogwheel.data.EventData.gaussian_noise(
        eventname, duration=128.0, detector_names='HLV', fmax=512.0,
        asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0.0, seed=seed)

    event_data.inject_signal(par_dic=injection_dic, approximant=approximant)

    return event_data