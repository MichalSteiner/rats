chaos.#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:34:12 2022

@author: chamaeleontis
"""
import seaborn as sns
from dataclasses import dataclass
import specutils as sp
from typing import Callable, Iterator, Union, Optional, Tuple, Any
import astropy.units as u
import matplotlib.pyplot as plt
#%%
SpectralRegion = sp.spectra.spectral_region.SpectralRegion
Axes = plt.Axes
Figure = plt.figure

#%% ProminentLines
@dataclass
class ProminentLines:
    '''
        Class that holds setting for given set of prominent lines
    '''
    
    name: str
    lines: list
    wavelength_range: SpectralRegion
    
    
    def create_figure(self,
                      n_rows:int = 1,
                      ) -> tuple[Figure, list[Axes]]:
        '''
        Initialize a figure for given line

        Parameters
        ----------
        n_rows : int, optional
            Number of rows in plot. The default is 1.

        Returns
        -------
        tuple[Figure, list[Axes]]
            Main figure that has been initialized.
            List of artists (Axes) to draw on
        

        '''
        fig, axs = plt.subplots(n_rows,len(self.wavelength_range))
        
        return fig,axs
        
    def set_xlim(self,
                 axs:list[Axes]
                 ) -> None:
        '''
        Given a list of axes, set x limits based on the class wavelength_range settings.

        Parameters
        ----------
        axs : list[Axes]
            Axes to set x-limits to.

        Returns
        -------
        None

        '''
        for subregion, ax in zip(self.wavelength_range,axs):
            ax.set_xlim(subregion.lower, subregion.upper)
        return None



#%% Prominent lines list
Sodium_doublet = ProminentLines(
    'Sodium',
    [5889.950,5895.924],
    sp.SpectralRegion(5886*u.AA, 5900*u.AA)
    )

Balmer_series = ProminentLines(
    'Balmer series',
    [6562.79],
    sp.SpectralRegion(6559*u.AA, 6566*u.AA)
    )

Potassium_doublet = ProminentLines(
    'Potassium',
    [7664.8991, 7698.9645],
    sp.SpectralRegion(7661*u.AA, 7668*u.AA) + sp.SpectralRegion(7695*u.AA,7702*u.AA)
    )

Calcium_lines = ProminentLines(
    'Calcium',
    [3933.66, 3968.47],
    sp.SpectralRegion(3930*u.AA, 3937*u.AA) + sp.SpectralRegion(3965*u.AA,3972*u.AA)
    )
#%%
# HEARTS targets
h_targets = [
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
# SPADES targets
s_targets = [
'GJ 436 b',
'HAT-P-11 b',
'HAT-P-41 b',
'KELT-9 b',
'HD 219134 b',
'KELT-5 b',
'WASP-43 b',
'XO-3 b'
]
# HEARTS and SPADES sodium detection
h_s_na_detection = [
    'WASP-49 b',
    'WASP-76 b',
    'WASP-121 b',
    'WASP-166 b',
    # 'WASP-127 b',
    'KELT-9 b',
    'KELT-11 b',
    ]
# HEARTS detection
h_na_detection = [
    'WASP-49 b',
    'WASP-76 b',
    'WASP-121 b',
    'WASP-166 b',
    'KELT-11 b',
    ]
# HEARTS non detection
h_na_non_detection = [
    'WASP-127 b',
    'KELT-10 b'
    ]

# Massive HEARTS planets not analyzed for TS
mass_planet_hearts = [
    'HAT-P-41 b', # 67 spectra, 2 transits, no idea about SNR, no access, 
    'WASP-43 b', # 4 transits, 3 at DACE (maybe access issue), all SNR below 30
    'XO-3 b' # 54 spectra, 1 night, sevice mode
    ]
# Langeveld2022 paper - sodium detections
langeveld = [
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

# KELT-10b list filters for lightcurves
list_filters = [
    'i', #i' filter
    'R', #R filter
    'g', #g' filter
    'z', #z' filter
    'Ecam_RG', #ECAM
    'B' #B
    ]

# KELT-10b paper, alias for list of filters
alias_filters = [
    "i'",
    "R",
    "g'",
    "z'",
    "ECAM",
    "B"
    ]

# WG2 GTO targets
observed_by_GTO = [
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

#
atreides_targets = [
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




list_of_cmaps = [
    sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
    sns.dark_palette("#69d", reverse=False, as_cmap=True),
    sns.dark_palette("#fff", reverse=False, as_cmap=True),
    ]


