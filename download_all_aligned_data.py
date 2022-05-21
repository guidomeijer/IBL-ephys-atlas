# -*- coding: utf-8 -*-
"""
Created on Sun May 15 11:54:00 2022

@author: guido
"""

from brainbox.io.one import SpikeSortingLoader
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Query all resolved sessions
ins = one.alyx.rest('insertions', 'list',
                    django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                          'session__qc__lt,50,'
                          'json__extended_qc__alignment_resolved,True')

for i, this_ins in enumerate(ins):
    # Load in spike data
    print(f'Loading spike data of session {i+1} of {len(ins)}')
    try:
        sl = SpikeSortingLoader(pid=this_ins['id'], one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue