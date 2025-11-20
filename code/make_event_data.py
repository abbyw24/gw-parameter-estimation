"""
Generate processed event data files for a gravitational-wave event.

This script converts raw HDF5 strain data into `.npz` event files used
by cogwheel, optionally applying inpainting and producing spectrograms.
"""

import argparse
import os
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

import gwosc
import cogwheel.data

def make_event_data(eventname, catalog, event_data_kwargs=None, tgps=None,
                    inpaint_times_by_det=None, overwrite=False, plot=True,
                    base_dir='/home/javier.roulet/cogwheel-catalog'):
    """
    Build a processed EventData object for a gravitational-wave event.

    Parameters
    ----------
    eventname : str
        Name of the GW event (e.g. 'GW150914').

    catalog : str
        Name of the GWOSC catalog (e.g. 'GWTC-4.0').

    event_data_kwargs : dict, optional
        Keyword arguments passed to
        `cogwheel.data.EventData.from_timeseries()`.

    tgps : float or None, optional
        GPS time, defaults to `gwosc.datasets.event_gps(eventname)`.

    inpaint_times_by_det : dict or str, optional
        Time ranges (per detector) to inpaint. If a JSON string, it
        will be parsed. See `cogwheel.data.EventData.inpaint`.

    overwrite : bool, default=False
        Whether to delete any existing processed data directory before
        writing new output.

    plot : bool, default=True
        If True, save a spectrogram of the processed data.

    Notes
    -----
    The function expects raw data in
    `base_dir/raw_data/{catalog}/{eventname}` as HDF5 files (one per
    detector). It writes processed results to
    `base_dir/processed_data/{catalog}/{eventname}`.
    """
    if isinstance(inpaint_times_by_det, str):
        inpaint_times_by_det = json.loads(inpaint_times_by_det)

    if isinstance(event_data_kwargs, str):
        event_data_kwargs = json.loads(event_data_kwargs)
    elif event_data_kwargs is None:
        event_data_kwargs = {}

    input_dir = Path(os.path.join(base_dir, 'raw_data', catalog, eventname))
    output_dir = Path(os.path.join(base_dir, 'processed_data', catalog, eventname))

    if overwrite and output_dir.is_dir():
        shutil.rmtree(output_dir)

    filenames = sorted(input_dir.glob('*.hdf5'))
    detector_names = tuple(f.name[0] for f in filenames)
    tgps = tgps or gwosc.datasets.event_gps(eventname)

    event_data = cogwheel.data.EventData.from_timeseries(
        filenames, eventname, detector_names, tgps, **event_data_kwargs
    )

    output_dir.mkdir()
    if inpaint_times_by_det:
        event_data = event_data.inpaint(inpaint_times_by_det)
        with open(output_dir / 'inpaint_times_by_det.json', 'w',
                  encoding='utf-8') as file:
            json.dump(inpaint_times_by_det, file, indent=2)

    with open(output_dir / 'event_data_kwargs.json', 'w',
              encoding='utf-8') as file:
        json.dump(event_data_kwargs, file, indent=2)

    event_data.to_npz(filename=output_dir / f'{eventname}.npz')

    for file in output_dir.iterdir():
        file.chmod(0o444)  # Make files read-only

    if plot:
        event_data.specgram(nfft=256)
        plt.savefig(output_dir / 'spectrogram.pdf', bbox_inches='tight')

    print(f'Created event_data in {output_dir}')
    return event_data


def _build_argparser():
    """Return an argparse.ArgumentParser for command-line use."""
    p = argparse.ArgumentParser(
        description="Generate processed event data for a GW event."
    )
    p.add_argument("eventname", type=str, help="Event name (e.g. GW150914).")
    p.add_argument("catalog", type=str, help="GWOSC catalog name (e.g. GWTC-3).")
    p.add_argument("--event-data-kwargs", type=json.loads, default="{}",
                   help="JSON string of kwargs for EventData.from_timeseries.")
    p.add_argument("--tgps", type=json.loads, default=None,
                   help="GPS time.")
    p.add_argument("--inpaint-times-by-det", type=json.loads, default=None,
                   help="JSON string of inpaint time ranges per detector.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing processed data.")
    p.add_argument("--no-plot", action="store_true",
                   help="Do not generate spectrogram plots.")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    make_event_data(
        args.eventname,
        args.catalog,
        event_data_kwargs=args.event_data_kwargs,
        tgps=args.tgps,
        inpaint_times_by_det=args.inpaint_times_by_det,
        overwrite=args.overwrite,
        plot=not args.no_plot,
    )
