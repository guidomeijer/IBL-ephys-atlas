# -*- coding: utf-8 -*-
"""
Created on Sun May 15 11:54:00 2022

@author: guido
"""

from brainbox.io.one import SpikeSortingLoader
from one.api import ONE
import pandas as pd
import numpy as np
from os.path import join
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

SAVE_PATH = 'F:\\'

# Query all resolved sessions
ins = one.alyx.rest('insertions', 'list',
                    django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                          'session__qc__lt,50,'
                          'json__extended_qc__alignment_resolved,True')

data_df = pd.DataFrame()
for i, this_ins in enumerate(ins):
    # Load in spike data
    print(f'Loading spike data of session {i+1} of {len(ins)}')
    try:
        # Load in spiking data
        sl = SpikeSortingLoader(pid=this_ins['id'], one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
        channels['rawInd'] = one.load_dataset(this_ins['session'], dataset='channels.rawInd.npy',
                                              collection=sl.collection)
        
        # Get spiking properties per channel
        spike_rate, isi = np.zeros(channels['rawInd'].shape[0]), np.zeros(channels['rawInd'].shape[0])
        spike_amp, peak_to_trough = np.zeros(channels['rawInd'].shape[0]), np.zeros(channels['rawInd'].shape[0])
        for j, chn in enumerate(channels['rawInd']):
            chn_spikes = np.isin(spikes.clusters, np.where(clusters.channels == chn)[0])
            if chn_spikes.sum() == 0:
                continue
            spike_rate[j] = spikes.times[chn_spikes].shape[0] / spikes.times[chn_spikes][-1]
            isi[j] = np.median(np.diff(spikes.times[chn_spikes]))
            spike_amp[j] = np.median(spikes.amps[chn_spikes])
            
        # Load in LFP power
        rms_lf = one.load_object(this_ins['session'], 'ephysTimeRmsLF',
                                 collection=f'raw_ephys_data/{this_ins["name"]}',
                                 attribute=['rms'])
        
        # Load in AP band RMS
        rms_ap = one.load_object(this_ins['session'], 'ephysTimeRmsAP',
                                 collection=f'raw_ephys_data/{this_ins["name"]}',
                                 attribute=['rms'])
        
        # Process RMS data (I don't really know what this does, should ask Mayo)
        rms_ap_data = rms_ap['rms'] * 1e6  # convert to uV
        median = np.mean(np.apply_along_axis(lambda x: np.median(x), 1, rms_ap_data))
        rms_ap_data_median = (np.apply_along_axis(lambda x: x - np.median(x), 1, rms_ap_data)
                              + median)
        rms_lf_data = rms_ap['rms'] * 1e6  # convert to uV
        median = np.mean(np.apply_along_axis(lambda x: np.median(x), 1, rms_lf_data))
        rms_lf_data_median = (np.apply_along_axis(lambda x: x - np.median(x), 1, rms_lf_data)
                              + median)
        
        # Take median of RMS data per channel
        rms_ap_median = np.median(rms_ap_data_median, axis=0)
        rms_lf_median = np.median(rms_lf_data_median, axis=0)
        
        # Add data to dataframe
        data_df = pd.concat((data_df, pd.DataFrame(data={
            'pid': this_ins['id'], 'channel': channels['rawInd'], 'acronym': channels['acronym'],
            'x': channels['x'], 'y': channels['y'], 'z': channels['z'],
            'atlas_id': channels['atlas_id'], 'axial_um': channels['axial_um'],
            'lateral_um': channels['lateral_um'],
            'spike_rate': spike_rate, 'isi': isi, 'spike_amp': spike_amp,
            'rms_ap': rms_ap_median, 'rms_lf': rms_lf_median})))
        
        # Save dataframe to disk
        data_df.to_pickle(join(SAVE_PATH, 'ephys_atlas_data.pickle'))
        
    except Exception as err:
        print(err)
        continue