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
#%%
SpectralRegion = sp.spectra.spectral_region.SpectralRegion
Axes = plt.Axes
Figure = plt.figure

#%% ProminentLines
@dataclass
class TargetList:
    name: str
    targets: list
    
#%%
HEARTS = TargetList(
    name = 'HEARTS',
    targets = [
        'GJ 9827 b',
        'HD 106315 b',
        'KELT-10 b',
        'KELT-11 b',
        'Kepler-444 b',
        'WASP-6 b',
        'WASP-101 b',
        'WASP-107 b',
        'WASP-121 b',
        'WASP-127 b',
        'WASP-166 b',
        'WASP-17 b',
        'WASP-31 b',
        'WASP-49 b',
        'WASP-76 b',
        'WASP-78 b',
        'WASP-94 A b',
        ]
    )

# SPADES targets
SPADES = TargetList(
    name = 'SPADES',
    targets= [
        'GJ 436 b',
        'HAT-P-11 b',
        'HAT-P-41 b',
        'KELT-9 b',
        'HD 219134 b',
        'KELT-5 b',
        'WASP-43 b',
        'XO-3 b'
        ]
    )

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

Langeveld2022 = TargetList(
    name = 'Langeveld et al (2022)',
    targets= [
        'WASP-69 b',
        'HD 189733 b',
        'WASP-21 b',
        'WASP-49 b',
        'WASP-79 b',
        'WASP-76 b',
        'MASCARA-2 b',
        'WASP-121 b',
        'WASP-189 b',
        'KELT-9 b',
        ]
    )

ESPRESSO_WG2_targets = TargetList(
    name = 'ESPRESSO WG2 targets',
    targets= [
        'WASP-76 b',
        'WASP-121 b',
        'CoRoT-7 b',
        'WASP-127 b',
        'GJ 436 b',
        'WASP-107 b',
        'HD 189733 b',
        'Mascara-1 b',
        'HD 209458 b',
        'GJ 9827 d',
        'GJ 9827 c',
        'GJ 9827 b',
        'HD 3167 b',
        'WASP-126 b',
        'HD136352 c',
        'HAT-P-26 b',
        'WASP-69 b', 
        'HD 106315 b', 
        'WASP-34 b',
        'WASP-54 b',
        'WASP-178 b',
        'Kelt-14 b',
        '55 Cnc A e',
        'TOI-132 b',
        'HD213885 b',
        'TOI-129 b',
        'WASP-118 b',
        'WASP-156 b',
        'WASP-20 b',
        'WASP-12 b',
        'WASP-62 b',
        'WASP-31 b',
        'Kelt-8 b',
        'TOI-1130 c',
        'WASP-103 b',
        'TOI-824 b',
        'K2-237 b',
        'HD 202772 A b',
        'WASP-94 A b',
        'TOI-849 b',
        'WASP-88 b',
        'WASP-90 b',
        'TOI-451 d',
        'TOI-451 b ',
        'TOI-421 b',
        'TOI-954 b',
        'K2-100 b',
        'WASP-101 b',
        'GJ 9827 b',
        'K2-141 b',
        'HAT-P-57 b',
        'WASP-21 b',
        'WASP-103 b',
        ]
    )

ATREIDES = TargetList(
    name= 'ATREIDES',
    targets = [
        'AU Mic c',
        'CoRoT-22 b',
        'HAT-P-26 b',
        'HATS-12 b',
        'HATS-37 A b',
        'HATS-38 b',
        'HATS-7 b',
        'HD 219666 b',
        'HD 56414 b',
        'HD 93963 A c',
        'K2-10 b',
        'K2-100 b',
        'K2-105 b',
        'K2-108 b',
        'K2-121 b',
        'K2-138 e',
        'K2-172 c',
        'K2-178 b',
        'K2-19 b',
        'K2-19 c',
        'K2-198 b',
        'K2-201 c',
        'K2-217 b',
        'K2-245 b',
        'K2-27 b',
        'K2-271 b',
        'K2-285 c',
        'K2-32 b',
        'K2-334 b',
        'K2-353 b',
        'K2-370 b',
        'K2-39 b',
        'K2-398 c',
        'K2-399 b',
        'K2-405 b',
        'K2-406 b',
        'K2-60 b',
        'K2-79 b',
        'K2-87 b',
        'K2-98 b',
        'NGTS-14 A b',
        'TOI-1231 b',
        'TOI-132 b',
        'TOI-181 b',
        'TOI-2000 c',
        'TOI-2374 b',
        'TOI-2498 b',
        'TOI-257 b',
        'TOI-3071 b',
        'TOI-421 c',
        'TOI-431 d',
        'TOI-451 d',
        'TOI-5174 b',
        'TOI-620 b',
        'TOI-908 b',
        'TOI-942 b',
        'TOI-942 c',
        'WASP-47 d',
        ]
    )

ATREIDESobserved = TargetList(
    name = 'ATREIDES observed targets',
    targets= [
        'TOI-421 c'
        'TOI-942 b',
        'K2-79 b'
    ]
    )

list_of_cmaps = [
    sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
    sns.dark_palette("#69d", reverse=False, as_cmap=True),
    sns.dark_palette("#fff", reverse=False, as_cmap=True),
    ]


