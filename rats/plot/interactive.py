#%% Importing libraries
import specutils as sp
import rats.spectra_manipulation as sm
import seaborn as sns
import pandas as pd
import plotly.express as px


#%% Color wheels
COLOR_WHEEL_DARK = [color for color in sns.color_palette('dark')]
COLOR_WHEEL_BRIGHT = [color for color in sns.color_palette('bright')]
COLOR_WHEEL_PASTEL = [color for color in sns.color_palette('pastel')]
COLOR_WHEEL_COLORBLIND = [color for color in sns.color_palette('colorblind')]




#%% Plot master list
def master_list(spectrum_list: sp.SpectrumList,
                figure_filename: str | None = None,
                ):
    
    first_spectrum = spectrum_list[0]
    
    x_axis_label = f'Wavelength [{first_spectrum.spectral_axis.unit}]'
    y_axis_label = f'Flux [{first_spectrum.flux.unit}]'
    
    if figure_filename is None:
        figure_filename = './master_list.html'

    for ind, spectrum in enumerate(spectrum_list):
        df = pd.DataFrame(
            {'wave': spectrum.spectral_axis.value,
            'flux': spectrum.flux.value,
            'flux_err': spectrum.uncertainty.array}
            )
        
        fig = px.scatter(
            df,
            x="wave",
            y="flux",
            error_y="flux_err"
            color= COLOR_WHEEL_PASTEL[ind]
            )
        
        x, y, yerr = sm.binning_spectrum(spectrum, 15)
        df_binned = pd.DataFrame(
            {'wave': spectrum.spectral_axis.value,
            'flux': spectrum.flux.value,
            'flux_err': spectrum.uncertainty.array}
            )
        
        fig = px.scatter(
            df_binned,
            x="wave",
            y="flux",
            error_y="flux_err"
            color= COLOR_WHEEL_DARK[ind]
            )
        
    fig.update_layout(template='simple_white')
    fig.write_html(figure_filename)
    
    return

def colormap(spectrum_list: sp.SpectrumList):
    
    
    
    ...
    



def _response_binning(change):
    
    ...

def _widget_binning():
    binning_factor = widgets.IntSlider(
        value=15.0,
        min=1.0,
        max=100.0,
        step=1.0,
        description='Binning factor:',
        continuous_update=False
        )
    
    binning_factor.observe(_response_binning, names="value")
    
    
    ...