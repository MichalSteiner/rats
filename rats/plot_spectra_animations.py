import matplotlib.pyplot as plt
from functools import singledispatch
from dataclasses import dataclass
import specutils as sp
import seaborn as sns
from enum import Enum
import matplotlib.widgets as wgt
import logging
from rats.utilities import default_logger_format

#%%

@dataclass
class AnimateList():
    # Spectrum list to animate through
    spectrumlist: sp.SpectrumList
    # Figure and ax objects, if we want to combine the functionality
    fig: plt.Figure | None = None
    ax: plt.Axes | None = None
    # Axes for buttons to add. This allows stacking multiple previous/next buttons together.
    ax_button_previous: plt.Axes | None = None
    ax_button_previous_label: str | None = None
    
    ax_button_next: plt.Axes | None = None
    ax_button_next_label: str | None = None
    
    # Whether to bin the spectrum as well
    binned : int | None = None
    
    def _next_spectrum():
        ...
    
    def _previous_spectrum():
        ...
        
    def _generate_empty_plot(self):
        ...
        
    def __post_init__(self):
        
        if (self.fig is None and self.ax is None):
            self._generate_empty_plot()
        
        return