import matplotlib.pyplot as plt
from functools import singledispatch
from dataclasses import dataclass
import specutils as sp
import seaborn as sns
import functools
from enum import Enum
import dill as pickle
import screeninfo

import logging
from rats.utilities import default_logger_format
from rats.spectra_manipulation import binning_spectrum

import rats.plots.utilities as ut
import matplotlib as mpl
#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)





@dataclass
class PlotsUtils:
    """
    Default utilities for plots. Provides methods for saving (and loading) plots and toggling between different modes. It also allows plotting all modes of plots at the same time.
    
    
    """
    
    def save(self):
        raise NotImplementedError
        
    def load(self):
        raise NotImplementedError
    
    def _plot(self):
        match self.mode:
            case ut.PlotModes.WHITEMODE_NORMAL | ut.PlotModes.WHITEMODE_PRESENTATION | ut.PlotModes.WHITEMODE_POSTER:
                self._plot_WHITEMODE()
            case ut.PlotModes.DARKMODE_NORMAL | ut.PlotModes.DARKMODE_PRESENTATION | ut.PlotModes.DARKMODE_POSTER:
                self._plot_DARKMODE()
    
    def plot_all_modes(self):
        """
        Plots all available modes at the same time. Recommended to turn off interactive mode of plt backends.
        """
        
        self._plot_WHITEMODE_NORMAL()
        self._plot_WHITEMODE_PRESENTATION()
        self._plot_WHITEMODE_POSTER()
        self._plot_DARKMODE_NORMAL()
        self._plot_DARKMODE_PRESENTATION()
        self._plot_DARKMODE_POSTER()
        return
    
    def _set_context(self):
        # DOCUMENTME
        
        save_context = sns.plotting_context()
        match self.mode:
            case ut.PlotModes.WHITEMODE_NORMAL:
                sns.set_context('paper')
            case ut.PlotModes.WHITEMODE_PRESENTATION:
                sns.set_context('talk')
            case ut.PlotModes.WHITEMODE_POSTER:
                sns.set_context('poster')
            case ut.PlotModes.DARKMODE_NORMAL:
                sns.set_context('paper')
            case ut.PlotModes.DARKMODE_PRESENTATION:
                sns.set_context('talk')
            case ut.PlotModes.DARKMODE_POSTER:
                sns.set_context('poster')
        return save_context
    
    def _set_lightmode(self) -> dict:
        """
        Changes between light and dark mode for plots.

        Returns
        -------
        old_rcParameters : dict
            Dictionary holding old rcParams to reset to after the plot is done.
        """
        
        if self.mode.value.startswith('dark'):
            old_rcParameters = plt.rcParams
            plt.style.use('dark_background')
        else:
            old_rcParameters = plt.rcParams
        return old_rcParameters
    
    # By default, no modes are implemented. Each class figure have to overwrite these functions.abs
    # This ensures an error is raised when calling respective functions, unless they were defined.
    def _plot_WHITEMODE(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_DARKMODE(self):
        raise NotImplementedError('Not implemented yet!')


#%%
@dataclass
class PlotSingleSpectrum(PlotsUtils):
    """
    Class used for plotting a single spectrum.

    Attributes:
    -----------
    spectrum: sp.Spectrum1D | sp.SpectrumCollection
        Spectrum to plot.
    binning_factor: int | None = 15
        Binning factor for the spectrum, by default 15. If set to None, no binning is performed.
    fig: plt.Figure | None = None
        Figure object with given artists, by default None. If set to None, a new figure is created.
    axs: plt.Axes | None = None
        List of artists, uncollapsed, by default None. If set to None, fig also must be set to None.
    
    color: str = 'black'
        Color of the spectrum, by default black. Check _get_colors_whitemode() for choices.
    figure_size: FigureSize = FigureSize.FULLSCREEN
        Size of the figure window, by default using the fullscreen mode. This is not implemented properly yet.
    
    mode: PlotModes | str = PlotModes.WHITEMODE_NORMAL 
        Mode of the plot. Choices are in PlotModes class.
    
    plot_filename: str | None = None
        Filename of the plot. To be implemented.
    data_filename: str | None = None
        Filename of the saved data. To be implemented.
    """
    
    
    spectrum: sp.Spectrum1D | sp.SpectrumCollection
    binning_factor: int | None = 15
    
    fig: plt.Figure | None = None
    axs: plt.Axes | None = None
    
    color: str = 'black'
    figure_size: ut.FigureSize = ut.FigureSize.FULLSCREEN
    
    mode: ut.PlotModes | str = ut.PlotModes.WHITEMODE_NORMAL 
    
    plot_filename: str | None = None
    data_filename: str | None = None
    
    def __post_init__(self):
        if type(self.mode) == str:
            self.mode = ut.PlotModes(self.mode)
        
        save_context = self._set_context()
        old_rcParameters = self._set_lightmode()
        
        if self.fig is None:
            self.fig, self.axs = plt.subplots(1,
                                              squeeze= False,
                                              figsize= self.figure_size.value
                                              )
        self._plot()
        
        sns.set_context(save_context)
        mpl.style.use(old_rcParameters)
        return
    
    def _plot_WHITEMODE(self):
        color_spectrum, color_bin = ut._get_colors_whitemode(self.color)
        self.plot_spectrum(
            ax= self.axs[0,0],
            spectrum= self.spectrum,
            binning_factor= self.binning_factor,
            color_spectrum= color_spectrum,
            color_bin= color_bin,
            )
        
    def _plot_DARKMODE(self):
        color_spectrum, color_bin = ut._get_colors_darkmode(self.color)
        self.plot_spectrum(
            ax= self.axs[0,0],
            spectrum= self.spectrum,
            binning_factor= self.binning_factor,
            color_spectrum= color_spectrum,
            color_bin= color_bin,
            )
        
    @staticmethod
    def plot_spectrum(ax: plt.Axes,
                      spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                      binning_factor: int | None = 15,
                      color_spectrum: tuple = sns.color_palette("pastel")[7],
                      color_bin: tuple = sns.color_palette("dark")[7],
                      ):
        ax.errorbar(spectrum.spectral_axis.value,
                    spectrum.flux.value,
                    spectrum.uncertainty.array,
                    color= color_spectrum,
                    alpha= 0.4,
                    fmt= '.'
                    )
        
        if binning_factor is not None:
            x, y, yerr = binning_spectrum(spectrum, bin_factor = binning_factor)
            ax.errorbar(
                x,
                y,
                yerr,
                color= color_bin,
                fmt= '.'
                )

#%%
@dataclass
class PlotMasterList(PlotsUtils):
    
    master_list: sp.SpectrumList
    binning_factor: int | None = 15
    
    fig: plt.Figure | None = None
    axs: list | None = None
    
    color: str = 'black'
    figure_size: ut.FigureSize = ut.FigureSize.FULLSCREEN
    
    mode: ut.PlotModes | str = ut.PlotModes.WHITEMODE_NORMAL 
    
    plot_filename: str | None = None
    data_filename: str | None = None
    
    
    def __post_init__(self):
        if type(self.mode) == str:
            self.mode = ut.PlotModes(self.mode)
        
        save_context = self._set_context()
        old_rcParameters = self._set_lightmode()
        
        if self.fig is None:
            figsize = self.figure_size.value
            figsize[1] *= len(master_list)
            self.fig, self.axs = plt.subplots(len(master_list),
                                              squeeze= False,
                                              figsize= figsize
                                              )
        self._plot()
        
        sns.set_context(save_context)
        mpl.style.use(old_rcParameters)
        return
    
    def _plot_WHITEMODE(self):
        color_spectrum, color_bin = ut._get_colors_whitemode(self.color)
        self.plot_master_list(
            axs= self.axs,
            master_list= self.master_list,
            binning_factor = self.binning_factor,
            color_spectrum = color_spectrum,
            color_bin= color_bin
            )
        
    def _plot_DARKMODE(self):
        color_spectrum, color_bin = ut._get_colors_darkmode(self.color)
        self.plot_master_list(
            axs= self.axs,
            master_list= self.master_list,
            binning_factor = self.binning_factor,
            color_spectrum = color_spectrum,
            color_bin= color_bin
            )
        
        raise NotImplementedError('Not implemented yet!')


    @staticmethod
    def plot_master_list(axs: plt.Axes,
                         master_list: sp.SpectrumList,
                         binning_factor: int| None = 15,
                         color_spectrum: tuple = sns.color_palette("pastel")[7],
                         color_bin: tuple = sns.color_palette("dark")[7]):
        # Probably index error
        for ni, master in enumerate(master_list):
            PlotSingleSpectrum.plot_spectrum(
                # FIXME
                ax= axs[ni, 0],# The 0 index is weird here. Needs to be changed
                spectrum= master,
                binning_factor= binning_factor,
                color_spectrum= color_spectrum,
                color_bin= color_bin
                )
        return
        
#%%
@dataclass
class PlotColormap(PlotsUtils):
    
    spectrum_list: sp.SpectrumList
    
    
    fig: plt.Figure | None = None
    axs: list | None = None
    
    divergence: bool = True
    low_color: str = 'blue'
    high_color: str = 'red'
    
    saturation_level: float = 75
    luminance_level: float = 50
    
    # cmap = plt.colormap('RdBu')
    
    def __post_init__(self):
        if self.fig is None:
            fig, axs = plt.subplots(1,
                                    squeeze= False)
            
        self.plot()
        return
    
    def _colormap(self,
                  inverse: bool = False):
        cmap = _get_colormap(
            low_color= self.low_color,
            high_color= self.high_color,
            inverse= inverse,
            saturation_level= self.saturation_level,
            luminance_level= self.luminance_level
            )
        return cmap
    
    
    def _plot_WHITEMODE_NORMAL(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_WHITEMODE_PRESENTATION(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_DARKMODE_NORMAL(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_DARKMODE_PRESENTATION(self):
        raise NotImplementedError('Not implemented yet!')
    
    def plot_colormap(self,
                      spectrum_list: sp.SpectrumList,
                      ):
        
        self.ax.pcolormesh(x,y,z)
        
        
        
        return