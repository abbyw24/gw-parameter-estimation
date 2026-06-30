import os
os.environ['OMP_NUM_THREADS'] = '1'
import itertools as it
from multiprocessing import Pool
import numpy as np

from pe_injections import create_injection, run_parameter_estimation_injection
from helpers import m1_m2_from_mchirp_q

def main(dL):

    """injection parameters"""
    mchirp = 30.    # Msun
    # construct the grid
    iota_grid = np.array([0.5]) * np.pi    # np.linspace(0., 1., 3)
    q_grid = np.array([0.5])

    seed = 14

    # overwrite things
    overwrite_samples = False       # overwrite existing PE samples?

    # lots of words?
    verbose = True

    # where to store results
    rundir = os.path.join('/home/abbye.williams/GWPE/data/pe_runs_lensing',
                            f'injections/d_luminosity/mchirp-{mchirp:.2f}')
        # this needs to be an absolute path for cogwheel.data.EventData.from_npz()
    os.makedirs(rundir, exist_ok=True)

    # prior class
    prior_class = 'IntrinsicLVCPrior'

    params_to_inject = {
        'd_luminosity' : dL,   # Mpc
        's1z' : 0.,
        's2z' : 0.,
        'psi' : 0.5
    }

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
    for (iota, q) in it.product(iota_grid, q_grid):

        print(f"iota = {iota / np.pi:.2f}pi, q = {q:.2f}")

        # label the inclination angles in terms of pi
        dL = params_to_inject['d_luminosity']
        eventname = f'GW{seed}_iota-{iota / np.pi:.2f}_q-{q:.2f}_dL-{dL:.0f}Mpc'
        eventdir = os.path.join(rundir, prior_class, eventname, 'run_0') # !!
        # !! hacky: need to fix hard-coded run_0.
        #   I should check for runs and/or delete the run directories altogether

        # if the PE samples also exist and overwrite_samples is False, just skip this point
        if os.path.exists(os.path.join(eventdir, 'samples.feather')) \
        and overwrite_samples is False:
            if verbose:
                print(f"PE samples exist for {eventname}. \
                        continuing to next point on the grid.")
            continue
        # otherwise, create a new injection at this point
        event_data = create_grid_point_injection(iota, q, eventname)

        # run PE on this injection:
        t_merger_guess = 0.
        mchirp_guess = mchirp
            # note that we cheat here and input the exact chirp mass as our guess
        run_parameter_estimation_injection(event_data, t_merger_guess, mchirp_guess,
                                            rundir=rundir, multiply_by_i=True, verbose=verbose)


if __name__=='__main__':
    main(1000.)
    # dLs = np.arange(500., 1001., 100.)  # Mpc
    # with Pool(len(dLs)) as p:
    #     p.map(main, dLs)