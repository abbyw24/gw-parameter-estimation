import os
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
        try:
            event_data, samples, samples_dir = helpers.load_event_data_and_posterior_samples(os.path.join(resdir, eventname), eventname)
        except FileNotFoundError:
            print(f"no samples for idx {idx}. continuing")
            continue

        save_fn = os.path.join(samples_dir, 'lnl_metric.npy')
        # # !! the following three lines just because I screwed up the initial calculation attempt...
        # if os.path.exists(save_fn):
        #     os.remove(save_fn)
        #     print(f"removed {save_fn}")
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


def flag_relative_binning_failure(idx, seed=12,
                                  resdir='/home/abbye.williams/GWPE/data/injections/Halton/IMRPhenomXPHM/IntrinsicLVCPrior'):
    eventname = f'GW{seed}_{idx}'
    eventdir = os.path.join(resdir, eventname)
    event_data, samples, samples_dir = helpers.load_event_data_and_posterior_samples(eventdir, eventname)
    # what's the SNR?
    snr = np.sqrt(event_data.injection['h_h'].sum())
    print(f"SNR is {snr:.1f}")
    # load the sampler
    sampler = utils.read_json(os.path.join(samples_dir, 'Sampler.json'))
    # get the likelihood
    like = sampler.posterior.likelihood
    # reinstantiate with the unlensed event data
    ed_u = event_data.reinstantiate(strain=event_data.strain * -1j)
    like_u = like.reinstantiate(event_data=ed_u)

    # what's the max marginalized likelihood of the lensed posterior samples?
    lidx = samples['lensed']
    lnl_marg = samples['lnl_marginalized'][lidx]
    lnl_max = lnl_marg.max()
    print(f"max (marginalized) lnlike of the lensed posterior samples is {lnl_max:.2f}")

    print(f"\nwith pn_phase_tol = {like_u.pn_phase_tol}:")
    # what is the marginalized likelihood of the true (injected) parameters?
    lnl_true = like_u.lnlike(ed_u.injection['par_dic'])
    print(f"(marginalized) lnlike of the injected pars is {lnl_true:.2f}")
    delta_lnl = lnl_true - lnl_max
    print(f"(marginalized) ∆lnl = {delta_lnl:.2f}")

    # reinstantiate the likelihood with a lower PN phase tolerance
    pn_phase_tol_hires = like_u.pn_phase_tol / 2
    like_u_hires = like_u.reinstantiate(pn_phase_tol=pn_phase_tol_hires)
    print(f"\nwith pn_phase_tol = {like_u_hires.pn_phase_tol}:")
    lnl_true_hires = like_u_hires.lnlike(ed_u.injection['par_dic'])
    print(f"(marginalized) lnlike of the injected pars is {lnl_true_hires:.2f}")
    delta_lnl_hires = lnl_true_hires - lnl_max
    print(f"(marginalized) ∆lnl = {delta_lnl_hires:.2f}")

    # what's the difference between these two?
    diff_pn_phase_tol = delta_lnl_hires - delta_lnl
    print(f"difference in ∆lnL with change in PN phase tol = {diff_pn_phase_tol:.3f}\n")

    res = dict(lnl_max=lnl_max, lnl_true_lores=lnl_true, delta_lnl_lores=delta_lnl, pn_phase_tol_hires = pn_phase_tol_hires,
               lnl_true_hires=lnl_true_hires, delta_lnl_hires=delta_lnl_hires, diff_pn_phase_tol=diff_pn_phase_tol)
    np.save(os.path.join(eventdir, f'pn_phase_tol_diagnostic.npy'), res)


if __name__=='__main__':
    main()