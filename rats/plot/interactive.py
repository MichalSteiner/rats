#%% Importing libraries
import specutils as sp
import rats.spectra_manipulation as sm
import seaborn as sns
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import rats.lists.spectra as ratslist
import dash
#%% Color wheels


COLOR_WHEEL_DARK = [color for color in sns.color_palette('dark')]
COLOR_WHEEL_BRIGHT = [color for color in sns.color_palette('bright')]
COLOR_WHEEL_PASTEL = [color for color in sns.color_palette('pastel')]
COLOR_WHEEL_COLORBLIND = [color for color in sns.color_palette('colorblind')]

for wheel in [COLOR_WHEEL_DARK, COLOR_WHEEL_BRIGHT, COLOR_WHEEL_PASTEL, COLOR_WHEEL_COLORBLIND]:
    wheel.pop(3)
    wheel.pop(4)

#%%


def _prepare_master_list_figure_directory(first_spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                                   figure_filename: str | None,
                                   mode: str,
                                   linelist: ratslist._ProminentLines,
                                   prepend: str = '') -> str:
    """
    Prepare a master list directory and plot filename for the plot.
    
    The file (and directory format) is in format /plot_mode/master_list/master_type/rest_frame/prepend_linelist.name.html

    Parameters
    ----------
    first_spectrum : sp.Spectrum1D | sp.SpectrumCollection
        First spectrum to get meta parameters from.
    figure_filename : str | None
        Figure filename. If None, automatic filename is generated based on meta parameters, otherwise unchanged.
    mode : str
        Plot mode (whitemode/darkmode)
    prepend : str, optional
        Whether to add some prepend to the filename, by default ''. This can be used if multiple similar plots are generated (e.g., master out with and without masking)

    Returns
    -------
    str
        Figure filename to save the plot to.
    """
    
    os.makedirs(f'./figures/{mode}/master_list/{first_spectrum.meta["type"]}/{first_spectrum.meta["rf"]}/',
            mode = 0o777,
            exist_ok = True
            )
    
    if figure_filename is None:
        figure_filename = f'./figures/{mode}/master_list/{first_spectrum.meta["type"]}/{first_spectrum.meta["rf"]}/{prepend}_{linelist.name}.html'
        
    return figure_filename

def _get_axis_labels(first_spectrum: sp.Spectrum1D | sp.SpectrumCollection) -> [str, str]:
    """
    Get axis labels for the plot from the first spectrum.

    Parameters
    ----------
    first_spectrum : sp.Spectrum1D | sp.SpectrumCollection
        First spectrum to get labels from.

    Returns
    -------
    x_axis_label : str
        Label on the x-axis
    y_axis_label : str
        Label on the y-axis
    """
    
    x_axis_label = f'Wavelength [{first_spectrum.spectral_axis.unit}]'
    try:
        if first_spectrum.flux.unit.name == '':
            y_axis_label = f'Flux [unitless]'
        else:
            y_axis_label = f'Flux [{first_spectrum.flux.unit}]'
    except:
        y_axis_label = 'Flux [unitless]'
    return x_axis_label, y_axis_label

def _update_axes_format(fig: go.Figure,
                        x_axis_label: str,
                        y_axis_label: str):
    fig.update_xaxes(
        title_text=x_axis_label,  # Set the x-axis label text
        title_font=dict(size=20),  # Set the font size of the x-axis label
        tickfont=dict(size=15)  # Set the font size of the x-axis tick labels
    )
    fig.update_yaxes(
        title_text=y_axis_label,  # Set the x-axis label text
        title_font=dict(size=20),  # Set the font size of the x-axis label
        tickfont=dict(size=15)  # Set the font size of the x-axis tick labels
    )
    

#%% Plot master list
def master_list(spectrum_list: sp.SpectrumList,
                linelist: ratslist._ProminentLines,
                figure_filename: str | None = None,
                mode: str = 'whitemode_normal',
                prepend: str = '',
                ):
    
    first_spectrum = spectrum_list[0]
    figure_filename = _prepare_master_list_figure_directory(first_spectrum,figure_filename,mode,linelist,prepend)
    x_axis_label, y_axis_label = _get_axis_labels(first_spectrum)

    fig = go.Figure()
    _update_axes_format(fig, x_axis_label, y_axis_label)
    
    spectrum_list = linelist.extract_region(spectrum_list)

    # Create the plot
    binning= range(2,101)
    for ind, spectrum in enumerate(spectrum_list):
        if ind != 0:
            visibility = 'legendonly'
        else:
            visibility = True
        
        
        opacity = 0.3
        errorbar_plot = go.Scatter(
                x=spectrum.spectral_axis.value,
                y=spectrum.flux.value,
                error_y=dict(
                    type='data',  # value of error bar given in data coordinates
                    array=spectrum.uncertainty.array,
                ),
                name=f'Night {spectrum.meta["num_night"]}',
                mode='markers',
                marker= {
                    'color': f'rgba{*COLOR_WHEEL_PASTEL[ind], opacity}',
                    },
                connectgaps=False,
                visible=visibility,
                legendgroup = f'{ind}',
                meta= {
                        'binning_factor': 'combined',
                        'night': ind,
                    }
            )
        fig.add_trace(errorbar_plot)
        
        # Create the plot for each binning step, but invisible
        for binning_factor in binning:
            x, y, yerr= sm.binning_spectrum(spectrum, binning_factor)
            errorbar_binned_plot = go.Scatter(
                    x=x,
                    y=y,
                    error_y=dict(
                        type='data',  # value of error bar given in data coordinates
                        array=yerr,
                    ),
                    name=f'Night {spectrum.meta["num_night"]}',
                    mode='markers',
                    marker= {
                        'color': f'rgba{*COLOR_WHEEL_DARK[ind], 0.8}',
                        },
                    connectgaps=False,
                    visible=False,
                    legendgroup = 'binning',
                    meta= {
                        'binning_factor': binning_factor,
                        'night': ind,
                    }
                )
            if binning_factor == 15: # Default binning step
                errorbar_binned_plot.legendgroup = f'{ind}'
                errorbar_binned_plot.visible = visibility
            
            fig.add_trace(errorbar_binned_plot)
        
        linelist.add_line_plotly(fig)
    fig.update_layout(template='simple_white')
    
    from dash import Dash, html, dcc, Input, Output, callback
    app = Dash()
    app.layout = html.Div([
    dcc.Slider(
            id='binning-slider',
            min=binning[0],
            max=binning[-1],
            step=1,
            value=15,
        ),
        dcc.Graph(figure=fig)
    ])
    

    @callback(
        Output('my-graph', 'figure'),
        Input('binning-slider', 'value')
    )
    def update_graph(value):
        updated_fig = fig
        for i, trace in enumerate(updated_fig.data):
            if trace.meta['binning_factor'] == value:
                updated_fig.data[i].legendgroup = updated_fig.data[i].meta['night']
                updated_fig.data[i].visible = night_visibility
            elif trace.meta['binning_factor'] == 'combined':
                night_visibility = updated_fig.data[i].visible
            else:
                updated_fig.data[i].legendgroup = 'binning'
                updated_fig.data[i].visible = False
                
        return updated_fig

    fig.write_html(figure_filename)
    
    print('Done')
    return

def master_list_test(spectrum_list: sp.SpectrumList,
                linelist: ratslist._ProminentLines,
                figure_filename: str | None = None,
                mode: str = 'whitemode_normal',
                prepend: str = '',
                ):
    
    first_spectrum = spectrum_list[0]
    figure_filename = _prepare_master_list_figure_directory(first_spectrum,figure_filename,mode,linelist,prepend)
    x_axis_label, y_axis_label = _get_axis_labels(first_spectrum)

    fig = go.Figure()
    _update_axes_format(fig, x_axis_label, y_axis_label)
    
    spectrum_list = linelist.extract_region(spectrum_list)

    # Create the plot
    for ind, spectrum in enumerate(spectrum_list):
        if ind != 0:
            visibility = 'legendonly'
        else:
            visibility = True
        
        
        opacity = 0.3
        errorbar_plot = go.Scatter(
                x=spectrum.spectral_axis.value,
                y=spectrum.flux.value,
                error_y=dict(
                    type='data',  # value of error bar given in data coordinates
                    array=spectrum.uncertainty.array,
                ),
                name=f'Night {spectrum.meta["num_night"]}',
                mode='markers',
                marker= {
                    'color': f'rgba{*COLOR_WHEEL_PASTEL[ind], opacity}',
                    },
                connectgaps=False,
                visible=visibility,
                legendgroup = f'{ind}',
                meta= {
                        'binning_factor': 'combined',
                        'night': ind,
                    }
            )
        fig.add_trace(errorbar_plot)
        
        x, y, yerr= sm.binning_spectrum(spectrum, 15)
        errorbar_binned_plot = go.Scatter(
                x=x,
                y=y,
                error_y=dict(
                    type='data',  # value of error bar given in data coordinates
                    array=yerr,
                ),
                name=f'Night {spectrum.meta["num_night"]}_binned',
                mode='markers',
                marker= {
                    'color': f'rgba{*COLOR_WHEEL_DARK[ind], 0.8}',
                    },
                connectgaps=False,
                visible=visibility,
                legendgroup = f'{ind}_binned',
                meta= {
                    'binning_factor': 15,
                    'night': ind,
                    'orig_spectrum': spectrum,
                }
            )
        fig.add_trace(errorbar_binned_plot)
        
    
    # slider_x = [[val * scale if (val.meta['binning_factor'] != 'combined') else val for idx, val in enumerate(x)] for x in x_data]
    
    # steps =  [
    #     {
    #         'label': f'Binning factor: {binning_factor}',
    #         'method': 'restyle',
    #         'args': [
    #             {'x': [[sm.binning_spectrum(x.meta['orig_spectrum'], binning_factor)[0]
    #                     if x.meta['binning_factor'] != 'combined'
    #                     else val for idx, val in enumerate(x)] for x in x_data],
    #              'y': [[sm.binning_spectrum(x.meta['orig_spectrum'], binning_factor)[1]
    #                     if x.meta['binning_factor'] != 'combined'
    #                     else val for idx, val in enumerate(y)] for y in y_data],
    #              f'error_y.array': [[sm.binning_spectrum(x.meta['orig_spectrum'], binning_factor)[2]
    #                                  if x.meta['binning_factor'] != 'combined'
    #                                  else val for idx, val in enumerate(error)] for error in error_data]},
    #         ]
    #     } for binning_factor in range(2, 101)
    # ]
    
    # # Create layout with slider
    # layout = go.Layout(
    #     sliders=[{
    #         'active': 15,
    #         'currentvalue': {'prefix': 'Binning Factor: '},
    #         'pad': {'t': 50},
    #         'steps': [{'label': str(binning_factor), 'method': 'restyle', 'args': [
    #             {
    #                 'x': [sm.binning_spectrum(spectrum, binning_factor)[0]],
    #                 'y': [sm.binning_spectrum(spectrum, binning_factor)[1]],
    #                 'error_y.array': [sm.binning_spectrum(spectrum, binning_factor)[2]],
    #             }
    #                 ]
    #             } for binning_factor in range(2, 101)]
    #     }]
    # )
        
    #     fig.update_layout(sliders=[{
    #                         'active': 0,
    #                         'steps': [
    #                             {
    #                                 'label': f'Scale: {scale}',
    #                                 'method': 'restyle',
    #                                 'args': [
    #                                     {'x': [[val * scale if idx in [0, 2] else val for idx, val in enumerate(x)] for x in x_data],
    #                                     'y': [[val * scale if idx in [0, 2] else val for idx, val in enumerate(y)] for y in y_data],
    #                                     f'error_y.array': [[val * scale if idx in [0, 2] else val for idx, val in enumerate(error)] for error in error_data]}
    #                                 ]
    #                             } for scale in range(1, 11)
    #                         ]
    #                     }])
        
        # fig.update_layout(layout)
    linelist.add_line_plotly(fig)
    
    
    fig.update_layout(template='simple_white')

    fig.write_html(figure_filename)
    
    print('Done')
    return


def plot_all_species_transmission(spectrum_list: sp.SpectrumList,
                                  prepend: str = '',
                                  ):
    for line in ratslist.RESOLVED_LINE_LIST:
        master_list(
            spectrum_list= spectrum_list,
            linelist= line,
            prepend= prepend
            )
    return

def plot_all_species_transmission_velocity():
    ...





def colormap(spectrum_list: sp.SpectrumList):
    
    
    
    ...
    