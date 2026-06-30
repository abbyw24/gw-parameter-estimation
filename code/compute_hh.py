import os
os.environ['OMP_NUM_THREADS'] = '1'
import itertools as it

import numpy as np
import cogwheel.utils

def main():

    # define the injection parameters to load PE results
    mchirp = 30.    # Msun

    iota_grid = np.arange(0., 0.51, 0.1) * np.pi
    q_grid = np.arange(0.1, 1.01, 0.1)

    seed = 12
    prior_class = 'IntrinsicLVCPrior'

    overwrite = True

    hh_type = 'hh_hm'

    # results directory
    resdir = os.path.join('/home/abbye.williams/GWPE/data/pe_runs_lensing/injections/iota_q_grid',
                            f'mchirp-{mchirp:.2f}', prior_class)

    for i, (iota, q) in enumerate(it.product(iota_grid, q_grid)):
        print(f"({iota / np.pi:.1f}pi, {q:.1f}) ({i / (len(iota_grid) * len(q_grid)) * 100:.2f}%)",
                end='\r')

        # event file
        eventname = f'GW{seed}_iota-{iota / np.pi:.2f}_q-{q:.2f}'
        save_fn = os.path.join(resdir, eventname, f'{hh_type}.npy')
        # if the file already exists, continue
        if os.path.exists(save_fn) and overwrite is False:
            print(f"{hh_type} exists for {eventname}. continuing to next point on the grid.")
            continue

        irun = 0
        while irun < 10:
            sampler_fn = os.path.join(resdir, eventname, f'run_{irun}', 'Sampler.json')
            if os.path.exists(sampler_fn):
                break
            irun += 1
        else:
            raise FileNotFoundError(f"{sampler_fn} does not exist")

        # compute the hh inner product
        hh = compute_hh(sampler_fn, hh_type=hh_type)
            # *note this function returns None if no sampler is found

        # save for this event only
        if hh is not None:
            np.save(save_fn, {(iota, q) : hh})

    print("done")


def compute_hh(fn, hh_type='hh_hm'):
    """
    Compute the waveform inner product <h|h> from a sampler json at `fn`,
    given the type of modes `hh_type`. If `hh_type == 'hh_hm'`, the inner product
    is computed from only the modes with m ≠ 2. If `hh_type == 'hh'`, the inner product
    is computed from all of the modes. If `hh_type == 'hh3'`, the inner product is
    computed from only the (3,3) mode.

    """
    # try to load the file
    try:
        # load the sampler json
        sampler = cogwheel.utils.read_json(fn)
        harmonic_modes = sampler.posterior.likelihood.waveform_generator.harmonic_modes
        # get the harmonic modes: which modes depends on `hh_type`
        if 'hm' in hh_type:
            sampler.posterior.likelihood.waveform_generator.harmonic_modes = [
                (ell, m) for (ell, m) in harmonic_modes if m != 2
            ]
        elif '3' in hh_type:
            sampler.posterior.likelihood.waveform_generator.harmonic_modes = [
                (ell, m) for (ell, m) in harmonic_modes if (ell == 3) and (m == 3)
            ]
        else:
            raise ValueError("unrecognized hh_type")
        # get the injected parameters
        par_dic = sampler.posterior.likelihood.event_data.injection['par_dic']
        # waveform (frequency domain) in each detector
        h = sampler.posterior.likelihood._get_h_f(par_dic)
        # compute the inner product for each detector
        h_h = sampler.posterior.likelihood._compute_h_h(h)
        # return the inner product from each detector
        return h_h
    # if the file isn't found:
    except FileNotFoundError:
        print(f"{fn} does not exist")
        return None

if __name__=='__main__':
    main()
