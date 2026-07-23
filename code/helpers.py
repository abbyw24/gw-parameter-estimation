"""
Helper functions...
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize

import cogwheel.data
import cogwheel.gw_utils
import cogwheel.plotting
import cogwheel.gw_plotting
import cogwheel.utils
import cogwheel.gw_utils
from pesummary.io import read

"""
SAMPLE LOADING
"""
def add_derived_quantities(samples):
    """
    Add columns inplace to a dataframe of samples.

    Includes redshift, mtot, m1_source, m2_source, mtot_source, chieff, chip, q.
    """
    samples['redshift'] = cogwheel.cosmology.z_of_d_luminosity(samples['d_luminosity'])

    samples['mtot'] = samples['m1'] + samples['m2']

    for mass_key in 'm1', 'm2', 'mtot':
        samples[f'{mass_key}_source'] = samples[mass_key] / (1 + samples['redshift'])

    samples['chieff'] = cogwheel.gw_utils.chieff(**samples[['m1', 'm2', 's1z', 's2z']])

    samples['chip'] = chip(**samples[['m1', 'm2', 's1x_n', 's2x_n', 's1y_n', 's2y_n']])

    samples['q'] = samples['m2'] / samples['m1']

def add_derived_quantities_injection(par_dict):
    """
    Add derived quantities to an injection dictionary.

    """
    par_dict['mchirp'] = cogwheel.gw_utils.m1m2_to_mchirp(par_dict['m1'], par_dict['m2'])
    par_dict['q'] = par_dict['m2'] / par_dict['m1']
    par_dict['chieff'] = cogwheel.gw_utils.chieff(par_dict['m1'], par_dict['m2'],
                                                  par_dict['s1z'], par_dict['s2z'])
    par_dict['chip'] = chip(par_dict['m1'], par_dict['m2'],
                            par_dict['s1x_n'], par_dict['s2x_n'], par_dict['s1y_n'], par_dict['s2y_n'])

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

    return event_data, samples, samples_dir


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
    
def plot_p_lensed(samples, eventname=None, fig=None,
                  figsize=(5,4), c='navy', bins=100, **kwargs):
    hist, edges = np.histogram(samples['p_lensed'], bins=bins,
                               weights=samples['weights'])
    # we want the pdf at edges, not midpoints:
    pdf = np.array([hist[0], *(hist[1:] + hist[:-1]) / 2, hist[-1]])
    
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig, plt.gca()
    ax.plot(edges, pdf, c=c, alpha=0.9, **kwargs)
    ax.axvline(0.5, c='k', alpha=0.5, ls='--', lw=1.)
    ax.set_xlabel(r'$p_\mathrm{II}$')
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., None)
    
    # vlines
    median, *span = get_pII_median_span(samples)
    for val in (median, *span):
        ax.plot([val] * 2, [0, np.interp(val, edges, pdf)], c=c, alpha=0.8, lw=0.5)
    idx = (edges > span[0]) & (edges < span[1])
    ax.fill_between(edges[idx], np.zeros_like(pdf[idx]), pdf[idx], color=c, alpha=0.1)
    val_err_str = '${}={}$' + cogwheel.plotting.latex_val_err(median, np.subtract(median, span))
    title_str = r"$p_\mathrm{II}$" if eventname is None else f"{eventname}: "r"$p_\mathrm{II}$"
    ax.set_title(title_str + val_err_str)

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
    return median, *span

def lensed_sanity_check(samples):
    unlensed = ~samples['lensed']
    w = samples['weights']
    N = len(samples)
    median, *span = get_pII_median_span(samples)
    print(f"{unlensed.sum() / N * 100:.2f}% of samples are unlensed ({w[unlensed].sum() / w.sum():.2e} of weights)")
    print(f"pII = {median:.5f} +{span[1]-median:.5f} -{median-span[0]:.5f}")

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

def chip(m1, m2, s1x_n, s2x_n, s1y_n, s2y_n):
    """
    Returns the precession spin parameter `chi_p`.

    Parameters
    ----------
    m1 : float
    m2 : float
    s1x_n : float
    s2x_n : float
    s1y_n : float
    s2y_n : float

    Returns
    -------
    chi_p : float

    """
    q = m2 / m1
    a1sintheta1 = s1x_n**2 + s1y_n**2
    a2sintheta2 = s2x_n**2 + s2y_n**2

    a = a1sintheta1
    b = q * (4 * q + 3) / (4 + 3 * q) * a2sintheta2
    
    return np.max([a, b])

"""
WAVEFORM THINGS
"""
def S_m(m, iota):
    return np.sin(iota)**(m-2)

def compute_HH(likelihood):
    # get the injected parameters
    par_dic = likelihood.event_data.injection['par_dic']
    # waveform (frequency domain) in each detector
    h = likelihood._get_h_f(par_dic, by_m=True)
    H = np.zeros_like(h[0,:])
    for m, h_m in zip(likelihood.waveform_generator.m_arr, h):
        H += S_m(m, par_dic['iota']) * h_m
    # and the inner product
    return likelihood._compute_h_h(H)

def max_over_distance(likelihood, par_dic, shift, print_res=False):
    """
    Returns the likelihood of the unlensed "imposter" waveform psi -> psi + pi/4 maximized over luminosity distance.
    """
    psi_shifted = par_dic['psi'] + shift
    def f(dL):
        return -likelihood.lnlike_fft(par_dic | dict(psi=psi_shifted, d_luminosity=dL))
    # then we can use scipy to find the value of dL that maximizes f
    res = scipy.optimize.minimize_scalar(f)
    if print_res:
        print(res)
    return res

"""
BAYES FACTOR
"""
def expected_mismatch(Fplus, Fcross, mu):
    """
    Expected mismatch ~ lnB / rhosq, following our derivation from the degeneracy hII ~ hI(psi + pi/4).
    """
    assert len(Fplus) == len(Fcross) == 3
    num = (np.sum([
        Fplus_d**2 + Fcross_d**2 for (Fplus_d, Fcross_d) in zip(Fplus, Fcross)
    ]))**2
    def denom(a, b):
        return np.sum([
            ((1 + mu**2) / 2 * a_d)**2 + (mu * b_d)**2 for (a_d, b_d) in zip(a, b)
        ])
    denom1 = denom(Fplus, Fcross)
    denom2 = denom(Fcross, Fplus)
    return 1 / 2 * (1 - (mu * (1 + mu**2) / 2)**2 * num / (denom1 * denom2))

def expected_mismatch_with_HH(Fplus, Fcross, mu, HH):

    num = (np.sum([
            (Fplus_d**2 + Fcross_d**2) * HH_d for (Fplus_d, Fcross_d, HH_d) in zip(Fplus, Fcross, HH)
    ]))**2

    def denom(a, b, HH):
        return np.sum([
            (((1 + mu**2) / 2 * a_d)**2 + (mu * b_d)**2) * HH_d for (a_d, b_d, HH_d) in zip(a, b, HH)
        ])

    denom1 = denom(Fplus, Fcross, HH)
    denom2 = denom(Fcross, Fplus, HH)

    return 1 / 2 * (1 - (mu * (1 + mu**2) / 2)**2 * num / (denom1 * denom2))

def expected_lnB_from_injection(event_data):
    init_dict = event_data.get_init_dict()
    injection_dict = init_dict['injection']
    par_dict = injection_dict['par_dic']
    rhosq = np.sum(injection_dict['h_h'])   # optimal network SNR squared from the waveform
    fplus, fcross = cogwheel.gw_utils.fplus_fcross(init_dict['detector_names'],
                                                    par_dict['ra'], par_dict['dec'], par_dict['psi'],
                                                    init_dict['tgps'])
    mu = np.cos(par_dict['iota'])
    # HH

    return rhosq * expected_mismatch(fplus, fcross, mu)

# def expected_lnB_from_injection(event_data):
#     init_dict = event_data.get_init_dict()
#     injection_dict = init_dict['injection']
#     par_dict = injection_dict['par_dic']
#     rhosq = np.sum(injection_dict['h_h'])   # optimal network SNR squared from the waveform
#     fplus, fcross = cogwheel.gw_utils.fplus_fcross(init_dict['detector_names'],
#                                                     par_dict['ra'], par_dict['dec'], par_dict['psi'],
#                                                     init_dict['tgps'])
#     mu = np.cos(par_dict['iota'])
#     return rhosq * expected_mismatch(fplus, fcross, mu)