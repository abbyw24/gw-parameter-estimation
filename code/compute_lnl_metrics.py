import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import numpy as np
from cogwheel import utils

sys.path.insert(0, '/home/abbye.williams/GWPE/code')
import helpers

def main():

    nevents = 100
    seed = 12
    approximant = 'IMRPhenomXPHM'
    resdir = f'/home/abbye.williams/GWPE/data/injections/Halton/{approximant}/IntrinsicLVCPrior'

    for idx in range(nevents):
        # load the sampler for this event
        eventname = f'GW{seed}_{idx}'
        flag_relative_binning_failure(eventname, resdir=resdir, verbose=True, overwrite=False)


def flag_relative_binning_failure(eventname,
                                  resdir='/home/abbye.williams/GWPE/data/injections/Halton/IMRPhenomXPHM/IntrinsicLVCPrior',
                                  verbose=True, overwrite=False):
    eventdir = os.path.join(resdir, eventname)
    save_fn = os.path.join(eventdir, f'pn_phase_tol_diagnostic.npy')
    if overwrite == False and os.path.exists(save_fn):
        print(f"{save_fn} exists. continuing")
        return
    try:
        event_data, _, samples_dir = helpers.load_event_data_and_posterior_samples(eventdir, eventname)
    except FileNotFoundError:
        print(f"no samples for {eventname}. continuing")
        return
    # what's the SNR?
    snr = np.sqrt(event_data.injection['h_h'].sum())
    if verbose:
        print(f"SNR is {snr:.1f}")
    # load the sampler
    sampler = utils.read_json(os.path.join(samples_dir, 'Sampler.json'))
    # get the likelihood
    like = sampler.posterior.likelihood
    # reinstantiate with the unlensed event data
    ed_u = event_data.reinstantiate(strain=event_data.strain * -1j)
    like_u = like.reinstantiate(event_data=ed_u)

    # # what's the max marginalized likelihood of the lensed posterior samples?
    # lidx = samples['lensed']
    # lnl_marg = samples['lnl_marginalized'][lidx]
    # lnl_max = lnl_marg.max()
    # if verbose:
    #     print(f"max (marginalized) lnlike of the lensed posterior samples is {lnl_max:.2f}" +
    #           f"\n\nwith pn_phase_tol = {like_u.pn_phase_tol}:")

    # what is the marginalized likelihood of the true (injected) parameters?
    pn_phase_tol = like_u.pn_phase_tol
    pn_phase_tols = [pn_phase_tol]  # start a list to keep track of all the ones we test
    lnl_previous = like_u.lnlike(ed_u.injection['par_dic'])
    max_iters = 4
    diffs = []  # start a list to keep track of all the differences
    for iiter in range(max_iters):
        pn_phase_tol /= 2   # double the resolution
        pn_phase_tols.append(pn_phase_tol)
        like_u_ = like_u.reinstantiate(pn_phase_tol=pn_phase_tol)
        lnl = compute_lnlike_marg(like_u_, ed_u.injection['par_dic'], max_n=10)
            # at this point lnl is either a number or ±inf
        diff = lnl - lnl_previous
        diffs.append(diff)
        if np.abs(diff) < 1:
            print(f"∆lnL = {diff:.1f} after {iiter+1} iterations (pn_phase_tol = {pn_phase_tol})")
            break
        else:
            lnl_previous = lnl  # save this run's lnl to compare to the next iteration
    if not np.abs(diff) < 1:
        print(f"∆lnL = {diff:.1f} after max iterations ({max_iters}; pn_phase_tol = {pn_phase_tol})")

    res = dict(diff_lnl=diff, pn_phase_tol=pn_phase_tol, niters=iiter+1, snr=snr, diffs=diffs, pn_phase_tols=pn_phase_tols)
    np.save(save_fn, res)

def compute_lnlike_marg(like, par_dic, max_n=10):
    for _ in range(max_n):
        lnl = like.lnlike(par_dic)
        if (np.isnan(lnl) == False) or not (np.isinf(lnl) == False):
            return lnl
    print(f"WARNING: lnlike_marg is {lnl} after {max_n} calls")
    return np.nan


def compute_delta_lnl_deprecated(eventname,
                                 resdir='/home/abbye.williams/GWPE/data/injections/Halton/IMRPhenomXPHM/IntrinsicLVCPrior'):

    try:
        event_data, samples, samples_dir = helpers.load_event_data_and_posterior_samples(os.path.join(resdir, eventname), eventname)
    except FileNotFoundError:
        print(f"no samples for {eventname}. continuing")
        return

    save_fn = os.path.join(samples_dir, 'lnl_metric.npy')
    if not os.path.exists(save_fn):
        # load the sampler
        sampler = utils.read_json(os.path.join(samples_dir, 'Sampler.json'))
        # get the likelihood
        like = sampler.posterior.likelihood
        # reinstantiate with the unlensed event data
        ed_u = event_data.reinstantiate(strain=event_data.strain * -1j)
        like_u = like.reinstantiate(event_data=ed_u)

        # what is the marginalized likelihood of the true (injected) parameters?
        lnl_true = like_u.lnlike(ed_u.injection['par_dic'])
        print(f"(marginalized) lnlike of the injected pars is {lnl_true:.1f}")

        # and the max marginalized likelihood of the lensed posterior samples?
        lidx = samples['lensed']
        lnl_marg = samples['lnl_marginalized'][lidx]
        lnl_max = lnl_marg.max()
        print(f"max (marginalized) lnlike of the lensed posterior samples is {lnl_max:.1f}")

        delta_lnl = lnl_true - lnl_max
        print(f"Delta lnl = {delta_lnl:.1f}")

        lnl_res = dict(lnl_true=lnl_true, lnl_max_lensed=lnl_max, delta_lnl=delta_lnl)
        np.save(save_fn, lnl_res)


if __name__=='__main__':
    main()