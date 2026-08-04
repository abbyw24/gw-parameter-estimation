import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["MPLCONFIGDIR"] = f"/tmp/{os.environ['USER']}_matplotlib_cache"
import sys
import warnings
from copy import deepcopy

import numpy as np
from cogwheel import utils

sys.path.insert(0, '/home/abbye.williams/GWPE/code')
import helpers

warnings.simplefilter("ignore", RuntimeWarning)

def main():

    approximant = 'IMRPhenomXPHM'
    # get the SNR and compute the Bayes factor
    resdir = f'/home/abbye.williams/GWPE/data/injections/Halton/{approximant}/IntrinsicLVCPrior'
    seed = 12
    nevents = 100

    # include the noise?
    noise = True

    idxs = [93] #range(nevents)

    for idx in idxs:
        # load the sampler for this event
        eventname = f'GW{seed}_{idx}'
        eventdir = os.path.join(resdir, eventname)
        try:
            event_data, _, samples_dir = helpers.load_event_data_and_posterior_samples(eventdir, eventname)
        except FileNotFoundError:
            print(f"no samples for idx {idx}. continuing")
            continue

        # load the sampler
        sampler = utils.read_json(os.path.join(samples_dir, 'Sampler.json'))
        # unlens the event data
        ed_u = event_data.reinstantiate(strain=event_data.strain * -1j)

        mismatch, rhosq = compute_mismatch(sampler.posterior, ed_u, noise=noise)

        # save these
        noise_tag = '_nonoise' if noise == False else ''
        np.save(os.path.join(eventdir, f'mismatch{noise_tag}.npy'), dict(mismatch=mismatch, rhosq=rhosq, noise=noise))


def compute_mismatch(post, ed_u, noise=False, verbose=True):

    approximant = ed_u.injection['approximant']
    injected_par_dic = ed_u.injection['par_dic']

    if noise == False:
        # reinstantiate with zero noise
        ed_u_0 = ed_u.reinstantiate(strain=np.zeros_like(ed_u.strain))
        ed_u_0.inject_signal(par_dic=injected_par_dic, approximant=approximant)
    else:
        # otherwise just keep the event data the same
        ed_u_0 = ed_u

    # likelihood
    like_u_0 = post.likelihood
    like_u_0.event_data = ed_u_0
    like_u_0._set_summary()
    like_u_0.asd_drift = None

    # Type II lens
    ed_0 = ed_u_0.reinstantiate(strain=ed_u_0.strain * 1j)
    like_0 = deepcopy(like_u_0)
    like_0.event_data = ed_0
    like_0._set_summary()
    like_0.asd_drift = None

    # what's the likelihood of the unlensed event data given the injected parameters?
    maxlnlike_u = like_u_0.lnlike_fft(injected_par_dic)
    if verbose:
        print(f"max ln likelihood of the unlensed data given injected parameters is {maxlnlike_u:.1f}")

    # what's the likelihood of the lensed event data given the imposter parameters?
    shifts = [+np.pi/4, -np.pi/4]
    shift_strs = ['+ pi/4', '- pi/4']
    maxlnlikes = []
    for shift_str, shift in zip(shift_strs, shifts):
        maxlnlike_imposter = like_0.lnlike_fft(injected_par_dic | dict(psi=injected_par_dic['psi'] + shift))
        if verbose:
            print(f"max ln likelihood of the lensed data given unlensed imposter (psi -> psi {shift_str}) is {maxlnlike_imposter:.1f}")
        maxlnlikes.append(maxlnlike_imposter)
    # take the max
    idx_max = np.argmax(maxlnlikes)
    maxlnlike_imposter = maxlnlikes[idx_max]
    shift_best = shifts[idx_max]
    if verbose:
        print(f"taking psi {shift_strs[idx_max]} as the best imposter")

    # maximize the likelihood over distance given the unlensed imposter
    maxD_res = helpers.max_over_distance(like_0, injected_par_dic, shift_best)
    maxlnlike_imposter_dL = -maxD_res['fun']
    if verbose:
        print(f"max ln likelihood given imposter, maximized over distance, is {maxlnlike_imposter_dL:.1f} at {maxD_res['x']:.1f} Mpc")

    rhosq = np.sum(ed_0.injection['h_h'])
    maxlnlike_diff = maxlnlike_u - maxlnlike_imposter_dL
    lnlike_mismatch = (maxlnlike_u - maxlnlike_imposter_dL) / rhosq
    if verbose:
        print(f"max lnlike difference (mismatch) = {maxlnlike_diff:.1f} ({lnlike_mismatch:.3f})")

    return lnlike_mismatch, rhosq

if __name__=='__main__':
    main()