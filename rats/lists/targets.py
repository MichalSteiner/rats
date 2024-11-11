chaos.#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:34:12 2022

@author: chamaeleontis
"""
import seaborn as sns
from dataclasses import dataclass
import specutils as sp
import astropy.units as u
import matplotlib.pyplot as plt

#%% ProminentLines
@dataclass
class TargetList:
    name: str
    targets: list
    
#%%
HEARTSandSPADESSodiumDetection = TargetList(
    name = 'HEARTS + SPADES sodium detections',
    targets= [
        'WASP-49 b',
        'WASP-76 b',
        'WASP-121 b',
        'WASP-166 b',
        'KELT-9 b',
        'KELT-11 b',
        ]
    )

HEARTSSodiumNonDetections = TargetList(
    name = 'HEARTS sodium non-detections',
    targets= [
        'WASP-127 b',
        'KELT-10 b'
        ]
    )

list_of_cmaps = [
    sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
    sns.dark_palette("#69d", reverse=False, as_cmap=True),
    sns.dark_palette("#fff", reverse=False, as_cmap=True),
    ]


