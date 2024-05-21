#%%

import numpy as np
import bokeh
import bokeh.plotting as blt
import specutils as sp 
import rats.lists.spectra as ratslist
import rats.spectra_manipulation as sm
import seaborn as sns

COLORWHEEL_WHITEMODE_PASTEL = (sns.color_palette('pastel').as_hex())
COLORWHEEL_WHITEMODE_BRIGHT = (sns.color_palette('bright').as_hex())
COLORWHEEL_WHITEMODE_DARK = (sns.color_palette('dark').as_hex())
COLORWHEEL_WHITEMODE_COLORBLIND = (sns.color_palette('colorblind').as_hex())



x = np.arange(100)
y = np.random.random(100) + 1
y_err = np.random.random(100)/4

p = blt.figure(width=400, height=400)
#%%
def _get_ColumnDataSource_from_spectrum(spectrum: sp.Spectrum1D | sp.SpectrumCollection):
    if type(spectrum) == sp.SpectrumCollection:
        raise NotImplementedError()
    
    CDS = bokeh.models.ColumnDataSource(
        {
        'x': spectrum.spectral_axis.value,
        'y': spectrum.flux.value,
        'y_error': spectrum.uncertainty.array,
        'y_error_upper': spectrum.flux.value + spectrum.uncertainty.array,
        'y_error_lower': spectrum.flux.value - spectrum.uncertainty.array,
        }
    )
    
    return CDS

def _whitemode_normal():
    ...
def _whitemode_presentation():
    ...
def _darkmode_normal():
    ...
def _darkmode_presentation():
    ...


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

def master_list(spectrum_list: sp.SpectrumList,
                linelist: ratslist._ProminentLines,
                figure_filename: str | None = None,
                mode: str = 'whitemode_normal',
                prepend: str = '',
                ):
    
    first_spectrum = spectrum_list[0]
    figure_filename = _prepare_master_list_figure_directory(first_spectrum,
                                                            figure_filename,
                                                            mode,
                                                            linelist,
                                                            prepend
                                                            )
    # blt.output_file(filename=figure_filename)
    
    # x_axis_label, y_axis_label = _get_axis_labels(first_spectrum)
    
    
    p = blt.figure(sizing_mode='stretch_both')
    
    spectrum_list = linelist.extract_region(spectrum_list)
    
    for ind, spectrum in enumerate(spectrum_list):
        CDS = _get_ColumnDataSource_from_spectrum(spectrum)       
        err_xs = []
        err_ys = []
        
        for x, y, yerr in zip(CDS.data['x'], CDS.data['y'], CDS.data['y_error']):
            err_xs.append((x, x))
            err_ys.append((y - yerr, y + yerr))

        # plot them
        p.multi_line(err_xs,
                     err_ys,
                     color=COLORWHEEL_WHITEMODE_PASTEL[ind],
                     line_alpha=0.2,
                     legend_label=f'Spectrum night {spectrum.meta["num_night"]}',
                     visible = True if ind == 0 else False,
                     )
        
        data_point_scatter = p.scatter(
            x=spectrum.spectral_axis.value,
            y=spectrum.flux.value,
            legend_label=f'Spectrum night {spectrum.meta["num_night"]}',
            visible = True if ind == 0 else False,
            line_color= COLORWHEEL_WHITEMODE_PASTEL[ind],
            line_alpha= 0.2,
            hatch_alpha= 0.2,
            fill_alpha= 0.2
        )
        
        x_binned,y_binned,yerr_binned = sm.binning_spectrum(spectrum, 15, False)
        err_xs_binned = []
        err_ys_binned = []
        
        for x, y, yerr in zip(x_binned,y_binned,yerr_binned):
            err_xs_binned.append((x, x))
            err_ys_binned.append((y - yerr, y + yerr))

        # plot them
        p.multi_line(err_xs_binned,
                     err_ys_binned,
                     color=COLORWHEEL_WHITEMODE_DARK[ind],
                     legend_label=f'Spectrum night {spectrum.meta["num_night"]}',
                     visible = True if ind == 0 else False,
                     )
        
        data_point_scatter_binned = p.scatter(
            x=x_binned,
            y=y_binned,
            legend_label=f'Spectrum night {spectrum.meta["num_night"]}',
            visible = True if ind == 0 else False,
            line_color= COLORWHEEL_WHITEMODE_DARK[ind],
        )
        
    
    
    p.legend.location = "top_right"
    p.legend.click_policy="hide"
    
    blt.show(p)
    
    
    
    # blt.save(p)
    
    ...
