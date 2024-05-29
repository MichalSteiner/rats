#!/usr/bin/env python3
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
import numpy as np
import rats.spectra_manipulation as sm
import plotly.graph_objects as go
import astropy.modeling as asmod
#%% ProminentLines
@dataclass
class _ProminentLines:
    '''
        Class that holds setting for given set of prominent lines
    '''
    
    name: str
    lines: list
    wavelength_range: sp.SpectralRegion
    
    
    def create_figure(self,
                      n_rows: int = 1,
                      ) -> tuple[plt.Figure, list[plt.Axes]]:
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
        fig, axs = plt.subplots(n_rows, len(self.wavelength_range))
        
        return fig,axs
        
    def set_xlim(self,
                 axs: list
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
    
    def add_line_plotly(self,
                        fig: go.Figure
                        ):
        """
        Add line to figure of plotly.

        Parameters
        ----------
        fig : go.Figure
            Figure to add vline to.
        """
        for line in self.lines:
            fig.add_vline(x=line.value,
                          line_width= 2,
                          line_dash='dash',
                          line_color='black',
                          )
            
    def _fit_local_continuum(self,
                             spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                             polynomial_order: int = 1) -> sp.Spectrum1D | sp.SpectrumCollection:

        match polynomial_order:
            case 0:
                model = asmod.models.Const1D(amplitude= np.nanmedian(spectrum.flux))
            case 1:
                fit = asmod.fitting.LinearLSQFitter()
                model = asmod.models.Linear1D()
                return fit(model, spectrum.spectral_axis, spectrum.flux)
            case _ if polynomial_order > 1:
                model = asmod.models.Polynomial1D(degree= polynomial_order,
                                                  )
        fit = asmod.fitting.LevMarLSQFitter()
        fitted_model = fit(model, spectrum.spectral_axis, spectrum.flux)
        return fitted_model

    
    def local_normalization(self,
                            spectrum_list: sp.SpectrumList,
                            polynomial_order: int = 1,
                            ) -> sp.SpectrumList:
        
        new_spectrum_list = sp.SpectrumList()
        for spectrum in spectrum_list:
            fitted_model = self._fit_local_continuum(
                spectrum= spectrum,
                polynomial_order= polynomial_order,
                )
            
            new_spectrum = spectrum.divide(fitted_model(spectrum.spectral_axis))
            new_spectrum_list.append(new_spectrum)
        
        return new_spectrum_list
        
    
    def velocity_fold(self,
                      spectra: sp.SpectrumList | sp.Spectrum1D | sp.SpectrumCollection,
                      constraint: list = [-200, 201]*u.km/u.s) -> tuple[sp.SpectrumList | sp.Spectrum1D | sp.SpectrumCollection, sp.SpectrumList]:
        """
        Velocity folding based on the line list attribute.

        Parameters
        ----------
        spectra : sp.SpectrumList | sp.Spectrum1D | sp.SpectrumCollection
            Spectra to velocity fold. Can be in any specutils format.
        constraint : list, optional
            Constraint to limit the region to, by default [-200, 201]*u.km/u.s

        Returns
        -------
        sp.SpectrumList | sp.Spectrum1D | sp.SpectrumCollection
            Velocity folded and averaged spectra. Shape is (len(spectra))
        sp.SpectrumList
            Separate lines separated in velocity range. Shape is (len(spectra), len(self.lines))
        """
        if type(spectra) == sp.SpectrumList:
            new_spectra, separate_line_spectra = sm.velocity_fold_spectrum_list(spectra, self.lines)
        else:
            new_spectra, separate_line_spectra = sm.velocity_fold_single_spectrum(spectra, self.lines)
        return new_spectra, separate_line_spectra
        
    
    def extract_region(self,
                       spectra: sp.SpectrumList | sp.Spectrum1D | sp.SpectrumCollection,
                       normalization: bool = False,
                       velocity_folding: bool = False,
                       **kwargs) -> sp.SpectrumList | sp.Spectrum1D | sp.SpectrumCollection:
        """
        Extract spectral region from spectra.
        
        Parameters
        ----------
        spectra : sp.SpectrumList | sp.Spectrum1D | sp.SpectrumCollection
            Spectra to extract region from. This can be in any of the specutils formats, and will output the same.
        normalization : bool
            Whether to renormalize the spectra as well by local continuum
        velocity_folding : bool
            Whether to fold the spectra in line list series.

        Returns
        -------
        sp.SpectrumList | sp.Spectrum1D | sp.SpectrumCollection
            Spectra with only the specific regions extracted. The output is the same type as the input

        Raises
        ------
        TypeError
            If spectra are in incorrect format, raise an error.
        """
        
        if type(spectra)== sp.SpectrumList:
            new_spectra = sm.extract_region_in_list(spectra, self.wavelength_range)  
        elif type(spectra)== sp.Spectrum1D or type(spectra) == sp.SpectrumCollection:
            new_spectra =  sm.extract_region_in_list(spectra, self.wavelength_range)
        else:
            raise TypeError('Spectra needs to be either sp.SpectrumList, sp.Spectrum1D or sp.SpectrumCollection type')
        
        if normalization:
            new_spectra = self.local_normalization(new_spectra,
                                                   **kwargs)
        if velocity_folding:
            new_spectra, _ = self.velocity_fold(new_spectra)

        return new_spectra
    
    def mask_velocity_region():
        ...



#%% Prominent lines list
Sodium_doublet = _ProminentLines(
    'Sodium',
    [5889.950 *u.AA, 5895.924*u.AA],
    sp.SpectralRegion(5886*u.AA, 5900*u.AA)
    )

Balmer_series = _ProminentLines(
    'Balmer series',
    [6562.79*u.AA,
    #  4861.35*u.AA,
    #  4340.472 *u.AA,
    #  4101.734 * u.AA
     ],
    (sp.SpectralRegion(6559*u.AA, 6566*u.AA) #+ 
    #  sp.SpectralRegion(4855*u.AA, 4865*u.AA) + 
    #  sp.SpectralRegion(4335*u.AA, 4345*u.AA) + 
    #  sp.SpectralRegion(4095*u.AA, 4105*u.AA))
    )
    )

Potassium_doublet = _ProminentLines(
    'Potassium',
    [7664.8991*u.AA, 7698.9645*u.AA],
    sp.SpectralRegion(7661*u.AA, 7668*u.AA) + sp.SpectralRegion(7695*u.AA,7702*u.AA)
    )

Calcium_lines = _ProminentLines(
    'Calcium',
    [3933.66*u.AA, 3968.47*u.AA],
    sp.SpectralRegion(3930*u.AA, 3937*u.AA) + sp.SpectralRegion(3965*u.AA,3972*u.AA)
    )

Manganium_lines = _ProminentLines(
    'Manganium',
    [4030.76*u.AA, 4033.07*u.AA, 4034.49*u.AA],
    sp.SpectralRegion(4026*u.AA, 4040*u.AA)
    )

Magnesium_lines = _ProminentLines(
    'Magnesium',
    [4571.10*u.AA, 5167.32*u.AA, 5172.68*u.AA, 5183.60*u.AA],
    sp.SpectralRegion(4568*u.AA, 4573*u.AA) + sp.SpectralRegion(5160*u.AA, 5190*u.AA)
    )

Lithium_lines = _ProminentLines(
    'Lithium',
    [6707.76*u.AA],
    sp.SpectralRegion(6700*u.AA, 6714*u.AA)
    )

RESOLVED_LINE_LIST = [
    Sodium_doublet,
    Balmer_series,
    Potassium_doublet,
    Calcium_lines,
    Manganium_lines,
    Magnesium_lines,
    Lithium_lines
]

list_of_cmaps = [
    sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
    sns.dark_palette("#69d", reverse=False, as_cmap=True),
    sns.dark_palette("#fff", reverse=False, as_cmap=True),
    ]


