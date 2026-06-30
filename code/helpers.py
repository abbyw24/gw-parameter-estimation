"""
Helper functions...
"""
import os
import sys
import numpy as np
import pandas as pd

import cogwheel.data
import cogwheel.gw_utils
import cogwheel.gw_plotting
import cogwheel.utils
from pesummary.io import read

"""
SAMPLE LOADING
"""
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


def load_event_data_and_posterior_samples(event_path, eventname, run=0, verbose=True):
    # find the "earliest" run with posterior samples
    i = run
    while i < 10:
        try:
            samples_dir = os.path.join(event_path, f'run_{i}')
            samples = pd.read_feather(os.path.join(samples_dir, 'samples.feather'))
            if verbose:
                print(f"loaded posterior samples for {eventname} run {i}")
            # return the event data too
            event_fn = os.path.join(samples_dir, f'{eventname}.npz')
            event_data = cogwheel.data.EventData.from_npz(filename=event_fn)
            break
        except FileNotFoundError:
            samples = None
            event_data = None
            i += 1
    if samples is None and verbose:
        raise FileNotFoundError(f"could not find posterior samples at {samples_dir}")
    else:
        # add derived quantities: redshift, chieff, q, etc.
        add_derived_quantities(samples)

    return event_data, samples


def load_lvk_samples(eventname, key=None):
    # load the posterior samples from the LIGO PE data release
    try:
        pe_data_fn = f"../data/pe_data_release/IGWN-GWTC4p0-1a206db3d_721-{eventname}-combined_PEDataRelease.hdf5"
        data = read(pe_data_fn)
        key = 'C00:IMRPhenomXPHM-SpinTaylor' if key is None else key
    except FileNotFoundError:
        # try GWTC-2.1
        try:
            pe_data_fn = f"../data/pe_data_release/IGWN-GWTC2p1-v2-{eventname}_PEDataRelease_mixed_cosmo.h5"
            data = read(pe_data_fn)
            key = 'C01:IMRPhenomXPHM' if key is None else key
        except FileNotFoundError:
            print(f"couldn't find any LVK samples for {eventname} (in GWTC-4.0 or GWTC-2.1)")
            return

    # get the posterior samples using the [key] approximant
    samples = data.samples_dict[key]

    return samples


def plot_posterior_samples(samples, event_data=None, plot_injected=False,
                            pars_to_plot=['m1', 'm2', 'iota'], c='royalblue',
                            tail_probability=1e-4, max_figsize=8, return_plot=False):
    """
    Plot posterior PE samples. If `plot_injected` is `True`, must pass `event_data`
    to load the injected parameter dictionary.
    """

    # plot parameters for our cogwheel results:
    plot_params = pars_to_plot.copy()
    plot_params.append('p_lensed')
    
    # corner this guy
    corner_plot = cogwheel.gw_plotting.CornerPlot(
        samples, params=plot_params, tail_probability=tail_probability, color_2d=c,
        kwargs_1d=dict(color=c, alpha=0.8), contour_kwargs=dict(alpha=0.8)
    )
    corner_plot.plot(max_figsize=max_figsize)

    if plot_injected:
        # injected parameters
        injected_par_dict = event_data.get_init_dict()['injection']['par_dic']
        # add derived quantities
        injected_par_dict['mchirp'] = cogwheel.gw_utils.m1m2_to_mchirp(injected_par_dict['m1'], injected_par_dict['m2'])
        injected_par_dict['chieff'] = cogwheel.gw_utils.chieff(injected_par_dict['m1'], injected_par_dict['m2'],
                                                                injected_par_dict['s1z'], injected_par_dict['s2z'])
        # injected parameters to plot
        # get the subset of the data release posterior parameters
        injected_pars_to_plot = {}
        for key, value in injected_par_dict.items():
            if key in pars_to_plot:
                injected_pars_to_plot[key] = value

        corner_plot.scatter_points(injected_pars_to_plot, colors=['#FF5C5C'], s=150,
                                    zorder=2, marker='+', adjust_lims=True)
    if return_plot:
        return corner_plot

def get_medians(samples, corner_plot):
    medians = {}
    for key in samples.keys():
        medians[key] = corner_plot._get_median_and_central_interval(key)[0]
    return medians


def get_pII_median_span(samples, confidence_level=0.9):
    """
    Compute the median and span of pII (`p_lensed`) from posterior samples.
    `confidence_level` = 0.9 is the default `corner_plot.plotstyle.confidence_level`.
    """
    #   (this is copied from plotting.py's _get_median_and_central_interval())
    tail_prob = (1 - confidence_level) / 2
    median, *span = cogwheel.utils.quantile(samples['p_lensed'], (.5, tail_prob, 1 - tail_prob), weights=samples['weights'])
    return median, span

"""
PARAMETER THINGS
"""
def m1_m2_from_mchirp_q(mchirp, q):
    """
    Returns the individual masses `m1` and `m2` given chirp mass `mchirp` and mass ratio `q`.
    """

    m1 = (1 + q)**(1/5) * q**(-3/5) * mchirp
    m2 = q * m1

    return m1, m2

"""
BAYES FACTOR
"""
def expected_lnB(rhosq, Fplus, Fcross, mu):
    """
    Expected ln Bayes factor, following Javier's derivation from the degeneracy hII ~ hI(psi + pi/4).
    """
    assert len(Fplus) == len(Fcross) == 3
    Fplus_sq = np.linalg.norm(Fplus)**2
    Fcross_sq = np.linalg.norm(Fcross)**2

    num = Fplus_sq * Fcross_sq * ((1 - mu**2) / 2)**4
    den = (Fplus_sq * mu**2 + Fcross_sq * ((1 + mu**2) / 2)**2) * (Fcross_sq * mu**2 + Fplus_sq * ((1 + mu**2) / 2)**2)

    return rhosq / 2 * num / den