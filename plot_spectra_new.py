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
#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

def style_decorator(style):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with plt.style.context(style):
                return func(*args, **kwargs)
        return wrapper
    return decorator

#%% 
class PlotModes(Enum):
    """
    Class of all possible Plotmodes. 

    Possible options of colors are:
        WHITEMODE_*
            Suitable for white-mode context.
        DARKMODE_*
            Suitable for dark-mode context
        
    Possible context options are:
        *_NORMAL
            Suitable for papers.
        *_PRESENTATION
            Suitable for presentations.
        *_POSTER
            Suitable for posters.
            
    These two options are merged together, so for darkmode presentation style, use:
        DARKMODE_PRESENTATION
    """
    
    WHITEMODE_NORMAL = 'whitemode_normal'
    DARKMODE_NORMAL = 'darkmode_normal'
    WHITEMODE_PRESENTATION = 'whitemode_presentation'
    DARKMODE_PRESENTATION = 'darkmode_presentation'
    WHITEMODE_POSTER = 'whitemode_poster'
    DARKMODE_POSTER = 'darkmode_poster'


class FigureSize(Enum):
    DOULBECOLUMN = (3.5,3)
    ONECOLUMN = (7,3)
    FULLSCREEN = (screeninfo.get_monitors()[0].width_mm  / 25.4,
                  screeninfo.get_monitors()[0].height_mm / 25.4)

def _plot_spectrum(spectrum: sp.Spectrum1D,
                   ax: plt.Axes,
                   color: str = 'black',
                   binning_factor: int | None = None):
    color_spectrum, color_bin = _get_colors_whitemode(color= color)
    
    ax.errorbar(spectrum.spectral_axis.value,
                spectrum.flux.value,
                spectrum.uncertainty.array,
                color= color_spectrum
                )
    
    # ax.errorbar(binned_spectrum.spectral_axis.value,
    #         binned_spectrum.flux.value,
    #         binned_spectrum.uncertainty.array,
    #         color= color_bin
    #         )
    
    
    return

def _get_colors_whitemode(color: str):
    """
    Get colors given a string

    Parameters
    ----------
    color : str
        Name of the color. 
        The choices are:
            'blue'
            'orange'
            'green'
            'red'
            'purple'
            'brown'
            'pink'
            'black'
            'gold'
            'cyan'

    Returns
    -------
    color_spectrum : color
        Color to use for the spectrum.
    color_bin : color
        Color of the bins to go with the spectrum.

    """
    
    match color:
        case 'blue':
            color_spectrum = sns.color_palette("pastel")[0]
            color_bin = sns.color_palette("dark")[0]
        case 'orange':
            color_spectrum = sns.color_palette("pastel")[1]
            color_bin = sns.color_palette("dark")[1]
        case 'green':
            color_spectrum = sns.color_palette("pastel")[2]
            color_bin = sns.color_palette("dark")[2]
        case 'red':
            color_spectrum = sns.color_palette("pastel")[3]
            color_bin = sns.color_palette("dark")[3]
        case 'purple':
            color_spectrum = sns.color_palette("pastel")[4]
            color_bin = sns.color_palette("dark")[4]
        case 'brown':
            color_spectrum = sns.color_palette("pastel")[5]
            color_bin = sns.color_palette("dark")[5]
        case 'pink':
            color_spectrum = sns.color_palette("pastel")[6]
            color_bin = sns.color_palette("dark")[6]
        case 'black':
            color_spectrum = sns.color_palette("pastel")[7]
            color_bin = sns.color_palette("dark")[7]
        case 'gold':
            color_spectrum = sns.color_palette("pastel")[8]
            color_bin = sns.color_palette("dark")[8]
        case 'cyan':
            color_spectrum = sns.color_palette("pastel")[9]
            color_bin = sns.color_palette("dark")[9]
        case _:
            raise ValueError('Color not found.')
        
    return color_spectrum, color_bin

def _invert_color(color: tuple) -> tuple:
    """
    Inverts a color to 1-fraction of 1 of each RGB values.

    Parameters
    ----------
    color : tuple
        Original color as tuple of three (RGB) values

    Returns
    -------
    new_color : tuple
        _description_
    """
    
    new_color = (
        1-color[0],
        1-color[1],
        1-color[2]
        )
    
    return new_color

def _get_colors_darkmode(color: str):
    """
    Get colors given a string

    Parameters
    ----------
    color : str
        Name of the color. 
        The choices are:
            'blue'
            'orange'
            'green'
            'red'
            'purple'
            'brown'
            'pink'
            'black'
            'gold'
            'cyan'

    Returns
    -------
    color_spectrum : color
        Color to use for the spectrum.
    color_bin : color
        Color of the bins to go with the spectrum.

    """
    color_spectrum, color_bin = _get_colors_whitemode(color)
    
    color_spectrum = _invert_color(color_spectrum)
    color_bin = _invert_color(color_bin)
        
    return color_spectrum, color_bin

def _get_hue(color: str):
    
    match color:
        case 'red':
            hue = 0
        case 'orange':
            hue = 30
        case 'yellow':
            hue = 60
        case 'green':
            hue = 120
        case 'cyan':
            hue = 180
        case 'blue':
            hue = 240
        case 'purple' | 'violet':
            hue = 270
        case 'pink':
            hue = 300
    return hue

def _get_colormap(low_color: str,
                  high_color: str,
                  inverse: bool = False,
                  saturation_level: float= 75,
                  luminance_level: float= 50,
                  separation_size:int = 1
                  ):
    if inverse:
        center = 'dark'
    else:
        center = 'light'
        
    cmap= sns.diverging_palette(
        _get_hue(low_color),
        _get_hue(high_color),
        s=saturation_level,
        l=luminance_level,
        sep=separation_size,
        center=center,
        as_cmap=True
        )
    return cmap


@dataclass
class PlotsUtils:
    def save(self):
        return
    def load(self):
        return
    
    def _plot(self):
        match self.mode:
            case PlotModes.WHITEMODE_NORMAL:
                self._plot_WHITEMODE_NORMAL()
            case PlotModes.WHITEMODE_PRESENTATION:
                self._plot_WHITEMODE_PRESENTATION()
            case PlotModes.WHITEMODE_POSTER:
                self._plot_WHITEMODE_POSTER()
            case PlotModes.DARKMODE_NORMAL:
                self._plot_DARKMODE_NORMAL()
            case PlotModes.DARKMODE_PRESENTATION:
                self._plot_DARKMODE_PRESENTATION()
            case PlotModes.DARKMODE_POSTER:
                self._plot_DARKMODE_POSTER
    
    def plot_all_modes(self):
        self._plot_WHITEMODE_NORMAL()
        self._plot_WHITEMODE_PRESENTATION()
        self._plot_WHITEMODE_POSTER()
        self._plot_DARKMODE_NORMAL()
        self._plot_DARKMODE_PRESENTATION()
        self._plot_DARKMODE_POSTER()
        return
    
    def _set_context(self):
        save_context = sns.plotting_context()
        match self.mode:
            case PlotModes.WHITEMODE_NORMAL:
                sns.set_context('paper')
            case PlotModes.WHITEMODE_PRESENTATION:
                sns.set_context('talk')
            case PlotModes.WHITEMODE_POSTER:
                sns.set_context('poster')
            case PlotModes.DARKMODE_NORMAL:
                sns.set_context('paper')
            case PlotModes.DARKMODE_PRESENTATION:
                sns.set_context('talk')
            case PlotModes.DARKMODE_POSTER:
                sns.set_context('poster')
        return save_context
    
    def _set_lightmode(self):
        if self.mode.value.startswith('dark'):
            old_rcParameters = plt.rcParams
            plt.style.use('dark_background')
        else:
            old_rcParameters = plt.rcParams
        return old_rcParameters
    
    # By default, no modes are implemented. Each class figure have to overwrite these functions.abs
    # This ensures an error is raised when calling respective functions, unless they were defined.
    def _plot_WHITEMODE_NORMAL(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_WHITEMODE_PRESENTATION(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_WHITEMODE_POSTER(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_DARKMODE_NORMAL(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_DARKMODE_PRESENTATION(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_DARKMODE_POSTER(self):
        raise NotImplementedError('Not implemented yet!')


#%%
@dataclass
class PlotSingleSpectrum(PlotsUtils):
    spectrum: sp.Spectrum1D | sp.SpectrumCollection
    bin_factor: int | None = 15
    
    fig: plt.Figure | None = None
    axs: list | None = None
    
    color: str = 'black'
    figure_size: FigureSize = FigureSize.FULLSCREEN
    
    mode: PlotModes | str = PlotModes.WHITEMODE_NORMAL 
    
    plot_filename: str | None = None
    data_filename: str | None = None
    
    def __post_init__(self):
        if type(self.mode) == str:
            self.mode = PlotModes(self.mode)
        
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
    
    def _plot_WHITEMODE_NORMAL(self):
        color_spectrum, color_bin = _get_colors_whitemode(self.color)
        self.plot_spectrum(
            ax= self.axs[0,0],
            spectrum= self.spectrum,
            bin_factor= self.bin_factor,
            color_spectrum= color_spectrum,
            color_bin= color_bin,
            )
    
    def _plot_WHITEMODE_PRESENTATION(self):
        color_spectrum, color_bin = _get_colors_whitemode(self.color)
        self.plot_spectrum(
            ax= self.axs[0,0],
            spectrum= self.spectrum,
            bin_factor= self.bin_factor,
            color_spectrum= color_spectrum,
            color_bin= color_bin,
            )
        
    def _plot_WHITEMODE_POSTER(self):
        color_spectrum, color_bin = _get_colors_whitemode(self.color)
        self.plot_spectrum(
            ax= self.axs[0,0],
            spectrum= self.spectrum,
            bin_factor= self.bin_factor,
            color_spectrum= color_spectrum,
            color_bin= color_bin,
            )
    def _plot_DARKMODE_NORMAL(self):
        color_spectrum, color_bin = _get_colors_darkmode(self.color)
        self.plot_spectrum(
            ax= self.axs[0,0],
            spectrum= self.spectrum,
            bin_factor= self.bin_factor,
            color_spectrum= color_spectrum,
            color_bin= color_bin,
            )
        
    def _plot_DARKMODE_PRESENTATION(self):
        color_spectrum, color_bin = _get_colors_darkmode(self.color)
        self.plot_spectrum(
            ax= self.axs[0,0],
            spectrum= self.spectrum,
            bin_factor= self.bin_factor,
            color_spectrum= color_spectrum,
            color_bin= color_bin,
            )
    
    def _plot_DARKMODE_POSTER(self):
        color_spectrum, color_bin = _get_colors_darkmode(self.color)
        self.plot_spectrum(
            ax= self.axs[0,0],
            spectrum= self.spectrum,
            bin_factor= self.bin_factor,
            color_spectrum= color_spectrum,
            color_bin= color_bin,
            )

    
    def plot_spectrum(self,
                      ax: plt.Axes,
                      spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                      bin_factor: int | None = 15,
                      color_spectrum: tuple = sns.color_palette("pastel")[7],
                      color_bin: tuple = sns.color_palette("dark")[7],
                      ):
        ax.errorbar(spectrum.spectral_axis.value,
                    spectrum.flux.value,
                    spectrum.uncertainty.array,
                    color= color_spectrum,
                    alpha= 0.7,
                    fmt= '.'
                    )
        
        if bin_factor is not None:
            x, y, yerr = binning_spectrum(spectrum, bin_factor = bin_factor)
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
    y_keyword: str | None = None
    
    fig: plt.Figure | None = None
    axs: list | None = None
    
    colormap: str = 'black'
    figure_size: FigureSize = FigureSize.FULLSCREEN
    
    mode: PlotModes | str = PlotModes.WHITEMODE_NORMAL 
    
    plot_filename: str | None = None
    data_filename: str | None = None
    
    
    def __post_init__(self):
        if self.fig is None:
            fig, axs = plt.subplots(1,
                                    squeeze= False)
            
        self.plot()
        return
    
    def _plot_WHITEMODE_NORMAL(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_WHITEMODE_PRESENTATION(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_DARKMODE_NORMAL(self):
        raise NotImplementedError('Not implemented yet!')
    def _plot_DARKMODE_PRESENTATION(self):
        raise NotImplementedError('Not implemented yet!')

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