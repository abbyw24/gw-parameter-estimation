"""
Generate GW injection parameters via QMC (Halton) sequence.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy.stats
import pandas as pd
from cogwheel import gw_prior, data, waveform, likelihood


def generate_injection_parameters(n_samples, approximant='IMRPhenomXODE'):
    """
    Generate injection parameters.
    """
    # prior
    prior = gw_prior.LVCPrior(
        f_avg=50,
        mchirp_range=(10, 50),
        detector_pair='HL',
        tgps=0,
        ref_det_name='L',
        f_ref=100.,
    )
    # draw a random set of parameters to inject
    samples_aux = generate_qmc_samples(prior, n_samples)

    # compute the SNR
    event_data = data.EventData.gaussian_noise(
        '', 64, 'HLV', ['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], prior.get_init_dict()['tgps'])
    wfg = waveform.WaveformGenerator.from_event_data(event_data, approximant)
    like = likelihood.CBCLikelihood(event_data, wfg)
    samples_aux['h_h'] = [
        like._compute_h_h(like._get_h_f(row)).sum()
        for _, row in samples_aux.iterrows()
    ]

    # modify the distances so that the SNR is flat in log
    samples = samples_aux[prior.standard_params].copy()
    samples['d_luminosity'] = rescale_d_luminosity(samples_aux, prior)

    return samples


def generate_qmc_samples(prior, n_samples, seed=None):
    """
    Sample the parameter space uniformly.
    Note: samples won't correspond to the prior (unless the prior is
    uniform).

    Parameters
    ----------
    n_samples : int
        How many samples to generate.

    seed:
        Passed to ``numpy.default_rng``, for reproducibility.

    Returns
    -------
    pd.DataFrame with columns per
    ``.sampled_params + .standard_params``, with samples distributed
    uniformly.
    """
    samples = pd.DataFrame(
        prior.cubemin + prior.cubesize * scipy.stats.qmc.Halton(
            len(prior.sampled_params), seed=seed).random(n_samples),
        columns=prior.sampled_params)

    prior.transform_samples(samples)
    return samples


def rescale_d_luminosity(samples, prior, snr_range=(10,200)):
    """
    Rescales `samples['d_luminosity']` such that the SNR is flat in log.
    """
    u = scipy.stats.qmc.scale(samples[['d_hat']], *prior.range_dic['d_hat'], reverse=True)[:, 0]
    logsnr_min, logsnr_max = np.log(snr_range)
    snr = np.exp(logsnr_min + u * (logsnr_max - logsnr_min))
    return samples['d_luminosity'] * np.sqrt(samples['h_h']) / snr