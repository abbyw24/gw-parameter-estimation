import os
import glob

def main():

    # list of events to download

    # list of GW events
    catalog = 'GWTC-4.0'
    # look for events in this directory
    catalog_dir = f'/home/javier.roulet/cogwheel-catalog/processed_data/{catalog}'
    event_dir_list = glob.glob(f'{catalog_dir}/GW*')
    eventname_list = [
        event_dir.split('/')[-1] for event_dir in event_dir_list
    ]
    print(f"found {len(eventname_list)} events")

    save_dir = '../data/pe_data_release'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for eventname in eventname_list:

        fn = f'IGWN-GWTC4p0-1a206db3d_721-{eventname}-combined_PEDataRelease.hdf5'
        if not os.path.exists(os.path.join(save_dir, fn)):
            url = f'https://zenodo.org/records/17014085/files/{fn}'
            bashcmd = f"wget -P {save_dir} {url}"

            os.system(bashcmd)

    print("done!", flush=True)

if __name__=='__main__':
    main()
