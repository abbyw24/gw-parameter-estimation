import numpy as np
import os
import sys

import cogwheel.data

from pe_injections import create_injection, run_parameter_estimation_injection


def main():

    # set other event parameters explicitly
    # keep chirp mass fixed
    mchirp = 30.    # Msun
    params_to_inject = {
        'd_luminosity' : 500.,   # Mpc
        's1z' : 0.,
        's2z' : 0.,
        'psi' : 0.5
    }

    seed = 12

    # construct the grid
    iota_grid = np.array([0.1, 0.3]) * np.pi    # np.linspace(0., 1., 3)
    q_grid = np.arange(0.1, 1.01, 0.1)

    # overwrite things
    overwrite_injections = False    # overwrite existing injection event data?
    overwrite_samples = False       # overwrite existing PE samples?

    # lots of words?
    verbose = True

    # where to store results
    rundir = f'/home/abbye.williams/GWPE/data/pe_runs_lensing/injections/iota_q_grid/mchirp-{mchirp:.2f}'
        # this needs to be an absolute path for cogwheel.data.EventData.from_npz()
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    # prior class
    prior_class = 'IntrinsicLVCPrior'

    def create_grid_point_injection(iota, q, eventname):
        """
        Create a GW injection at grid point (`iota`, `q`). Wrapper for `create_injection()`.

        Requires the following variables to be defined:
        - params_to_inject
        - mchirp
        - seed
        """

        injection_dict = params_to_inject.copy()
        injection_dict['iota'] = iota

        # compute the individual masses based on the chirp mass and q
        m1, m2 = m1_m2_from_mchirp_q(mchirp, q)
        injection_dict['m1'] = m1   # Msun
        injection_dict['m2'] = m2   # Msun

        event_data = create_injection(eventname, seed, injection_dict)

        return event_data

    # loop through the grid
    for i, iota in enumerate(iota_grid):
        for j, q in enumerate(q_grid):

            print(f"iota = {iota / np.pi:.2f}pi, q = {q:.2f}")

            # label the inclination angles in terms of pi
            eventname = f'GW{seed}_iota-{iota / np.pi:.2f}_q-{q:.2f}'
            eventdir = os.path.join(rundir, prior_class, eventname, 'run_0')
            fn = os.path.join(eventdir, eventname)
            # !! hacky: need to fix hard-coded run_0. I should check for runs and/or delete the run directories altogether

            # if the event data exists and overwrite_injections is False:
            if os.path.exists(f'{fn}.npz') and overwrite_injections is False:
                # if the PE samples also exist and overwrite_samples is False, just skip this point
                if os.path.exists(os.path.join(eventdir, f'samples.feather')) and overwrite_samples is False:
                    if verbose:
                        print(f"PE samples exist for {eventname}. continuing to next point on the grid.")
                    continue
                # otherwise load the event data
                else:
                    if verbose:
                        print(f"loading preexisting event data for {eventname}")
                    event_data = cogwheel.data.EventData.from_npz(fn)
            
            # otherwise, create a new injection at this point
            else:
                event_data = create_grid_point_injection(iota, q, eventname)

            # run PE on this injection:
            t_merger_guess = 0.
            mchirp_guess = mchirp   # note that we cheat here and input the exact chirp mass as our guess
            run_parameter_estimation_injection(event_data, t_merger_guess, mchirp_guess,
                                                rundir=rundir, multiply_by_i=True, verbose=verbose)


def m1_m2_from_mchirp_q(mchirp, q):
    """
    Returns the individual masses `m1` and `m2` given chirp mass `mchirp` and mass ratio `q`.
    """

    m1 = (1 + q)**(1/5) * q**(-3/5) * mchirp
    m2 = q * m1

    return m1, m2

if __name__=='__main__':
    main()
