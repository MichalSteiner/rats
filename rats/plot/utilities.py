"""
This module provides utility functions for plotting, including setting up plot styles and creating subplots.

Functions:
- set_plot_style(style: str) -> None:
    Sets the plot style.

- create_subplots(nrows: int, ncols: int, figsize: tuple) -> tuple[plt.Figure, np.ndarray]:
    Creates subplots with the specified number of rows, columns, and figure size.

- add_colorbar(fig: plt.Figure, ax: plt.Axes, mappable: plt.cm.ScalarMappable, orientation: str = 'vertical') -> plt.colorbar:
    Adds a colorbar to the plot.
"""

#%% Importing libraries
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import screeninfo


import numpy as np

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

def _get_hue(color: str) -> float:
    """
    Giving a string name of a color, provides respective hue.

    Parameters
    ----------
    color : str
        Name of the color. Options are:
            'red'
            'orange'
            'yellow'
            'green'
            'cyan'
            'blue'
            'purple' or 'violet'
            'pink'

    Returns
    -------
    hue : float
        Number between 0 and 360 providing respective hue value.
    """
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
    # DOCUMENTME
    
    
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


def _marker_instrument(Instruments: list) -> list:
    """Generate a list of markers to put in the plot for each instrument uniquely"""
    marker_list = ['o', "v", "p", 'P', "*", 'thin_diamond'] 
    unique_inst = np.unique(Instruments)
    if len(unique_inst) > len(marker_list):
        raise ValueError("Please add more markers in the list, as the number of instruments is too large. Regardless, the plot will likely be too crowded and needs to be split anyway.")
    markers_instrument = []
    for test_instrument in Instruments:
        for ind, unique in enumerate(unique_inst):
            if test_instrument == unique:
                markers_instrument.append(marker_list[ind])
    return markers_instrument

def _color_night(Night_number: int) -> tuple:
    """Defines a color for each night uniquely"""
    return sns.color_palette('dark')[Night_number - 1]

# List of undocumented functions for further review:
# - set_plot_style
# - create_subplots
# - add_colorbar

