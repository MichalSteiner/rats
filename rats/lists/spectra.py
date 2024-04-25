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
class _ProminentLines:
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
Sodium_doublet = _ProminentLines(
    'Sodium',
    [5889.950,5895.924],
    sp.SpectralRegion(5886*u.AA, 5900*u.AA)
    )

Balmer_series = _ProminentLines(
    'Balmer series',
    [6562.79],
    sp.SpectralRegion(6559*u.AA, 6566*u.AA)
    )

Potassium_doublet = _ProminentLines(
    'Potassium',
    [7664.8991, 7698.9645],
    sp.SpectralRegion(7661*u.AA, 7668*u.AA) + sp.SpectralRegion(7695*u.AA,7702*u.AA)
    )

Calcium_lines = _ProminentLines(
    'Calcium',
    [3933.66, 3968.47],
    sp.SpectralRegion(3930*u.AA, 3937*u.AA) + sp.SpectralRegion(3965*u.AA,3972*u.AA)
    )


list_of_cmaps = [
    sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
    sns.dark_palette("#69d", reverse=False, as_cmap=True),
    sns.dark_palette("#fff", reverse=False, as_cmap=True),
    ]


