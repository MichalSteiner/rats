#%%

import numpy as np
import bokeh
import bokeh.plotting as blt
import specutils as sp 
import rats.lists.spectra as ratslist
import rats.spectra_manipulation as sm
import seaborn as sns
import astropy.units as u
import os

COLORWHEEL_WHITEMODE_PASTEL = (sns.color_palette('pastel').as_hex())
COLORWHEEL_WHITEMODE_BRIGHT = (sns.color_palette('bright').as_hex())
COLORWHEEL_WHITEMODE_DARK = (sns.color_palette('dark').as_hex())
COLORWHEEL_WHITEMODE_COLORBLIND = (sns.color_palette('colorblind').as_hex())

BINNING_JS_CODE = """
    var data = source_spectrum.data;
    var bin_data = source_binned.data;
    var bin_width = slider.value;
    var x = data['x'];
    var y = data['y'];
    var y_error = data['y_error'];
    var y_error_lower = data['y_error_lower'];
    var y_error_upper = data['y_error_upper'];
    var num_bins = Math.floor(y.length / bin_width);
    
    // Compute new bins
    var x_bins = [];
    var y_bins = [];
    var y_error_bins = [];
    var y_err_lower_bins = [];
    var y_err_upper_bins = [];
    
    for (var i = 0; i < num_bins; i++) {
        var start = i * bin_width;
        var end = start + bin_width;
        
        var x_bin_values = x.slice(start, end); // Extract x values for the current bin
        var y_bin_values = y.slice(start, end);  // Extract values for the current bin
        var y_bin_error_values = y_error.slice(start, end);
        var y_bin_error_lower_values = y_error_lower.slice(start, end);
        var y_bin_error_upper_values = y_error_upper.slice(start, end);
        
        var valid_values_y = y_bin_values.filter(value => !isNaN(value));
        var valid_values_y_err = y_bin_error_values.filter(value => !isNaN(value));
        var valid_values_y_err_low = y_bin_error_lower_values.filter(value => !isNaN(value));
        var valid_values_y_err_upp = y_bin_error_upper_values.filter(value => !isNaN(value));
        
        if (valid_values_y.length > 0) {
            var x_mean = x_bin_values.reduce((a, b) => a + b, 0) / x_bin_values.length;
            x_bins.push(x_mean);
            
            var y_mean = valid_values_y.reduce((a, b) => a + b, 0) / valid_values_y.length;
            y_bins.push(y_mean);
            
            var y_errors = valid_values_y_err.reduce((a, b) => a + Math.pow(b, 2), 0) / ((valid_values_y_err.length - 1) * valid_values_y_err.length);
            
            y_error_bins.push(Math.sqrt(y_errors));
            y_err_lower_bins.push(y_mean - Math.sqrt(y_errors));
            y_err_upper_bins.push(y_mean + Math.sqrt(y_errors));
            
        } else {
            x_bins.push(NaN);
            y_bins.push(NaN);
            y_error_bins.push(NaN);
            y_err_lower_bins.push(NaN);
            y_err_upper_bins.push(NaN);
        }
    }
    
    var errors_combined = [y_err_lower_bins, y_err_upper_bins];
    var x_values_error_combined = [x_bins, x_bins];

    // Function to transpose a 2D array
    function transposeArray(array) {
        return array[0].map((_, colIndex) => array.map(row => row[colIndex]));
    }

    // Transpose the arrays
    var transposed_errors = transposeArray(errors_combined);
    var transposed_x_values_error = transposeArray(x_values_error_combined);

    
    // Update the plot values
    bin_data['x'] = x_bins;
    bin_data['y'] = y_bins;
    bin_data['y_error'] = y_err_lower_bins;
    bin_data['y_error_lower'] = y_err_lower_bins;
    bin_data['y_error_upper'] = y_err_upper_bins;
    

    // Assign the transposed arrays to bin_data
    bin_data['errors'] = transposed_errors;
    bin_data['x_values_error'] = transposed_x_values_error;
    
    source_binned.change.emit();
"""

#%%
def _get_ColumnDataSource_from_spectrum(spectrum: sp.Spectrum1D | sp.SpectrumCollection):
    if type(spectrum) == sp.SpectrumCollection:
        raise NotImplementedError()
    
    CDS = bokeh.models.ColumnDataSource( #type:ignore
        {
        'x': spectrum.spectral_axis.value, #type:ignore
        'y': spectrum.flux.value,
        'y_error': spectrum.uncertainty.array, #type:ignore
        'y_error_upper': spectrum.flux.value + spectrum.uncertainty.array, #type:ignore
        'y_error_lower': spectrum.flux.value - spectrum.uncertainty.array, #type:ignore
        'errors': list(np.asarray([(spectrum.flux.value + spectrum.uncertainty.array), (spectrum.flux.value - spectrum.uncertainty.array)]).T),#type:ignore
        'x_values_error': list(np.asarray([(spectrum.spectral_axis.value), (spectrum.spectral_axis.value)]).T),#type:ignore
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


def _get_axis_labels(first_spectrum: sp.Spectrum1D | sp.SpectrumCollection) -> tuple[str, str]:
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
    
    
    if first_spectrum.spectral_axis.unit.is_equivalent(u.AA): #type:ignore
        x_axis_label = f'Wavelength [{first_spectrum.spectral_axis.unit}]' #type:ignore
    elif first_spectrum.spectral_axis.unit.is_equivalent(u.m/u.s): #type:ignore
        x_axis_label = f'Velocity [{first_spectrum.spectral_axis.unit}]' #type:ignore
    else:
        raise ValueError('Unknown unit type')
    try:
        if first_spectrum.flux.unit.name == '': #type:ignore
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
    
    os.makedirs(f'./figures/{mode}/master_list/{first_spectrum.meta["type"]}/{first_spectrum.meta["rf"]}/', #type:ignore
            mode = 0o777,
            exist_ok = True
            )
    
    if figure_filename is None:
        if prepend != '':
            prepend = prepend + '_'
        
        figure_filename = f'./figures/{mode}/master_list/{first_spectrum.meta["type"]}/{first_spectrum.meta["rf"]}/{prepend}{linelist.name}.html' #type:ignore
        
    return figure_filename

def master_list(spectrum_list: sp.SpectrumList,
                linelist: ratslist._ProminentLines,
                figure_filename: str | None = None,
                mode: str = 'whitemode_normal',
                prepend: str = '',
                normalization: bool = False,
                polynomial_order: None | int = None,
                velocity_fold: bool = False,
                ):
    
    first_spectrum = spectrum_list[0]
    figure_filename = _prepare_master_list_figure_directory(first_spectrum,
                                                            figure_filename,
                                                            mode,
                                                            linelist,
                                                            prepend
                                                            )
    blt.output_file(filename=figure_filename)
    
    x_axis_label, y_axis_label = _get_axis_labels(first_spectrum)
    
    p = blt.figure(width=1600, height=1200)
    
    spectrum_list = linelist.extract_region(spectrum_list,
                                            normalization=normalization,
                                            velocity_folding= velocity_fold,
                                            polynomial_order=polynomial_order)
    
    binning_slider = bokeh.models.Slider(
        start=2,
        end=100,
        value=15,
        step=1,
        title="Binning slider"
    )
    
    
    for ind, spectrum in enumerate(spectrum_list):
        CDS = _get_ColumnDataSource_from_spectrum(spectrum)

        p.multi_line('x_values_error',
                     'errors',
                     source= CDS,
                     color=COLORWHEEL_WHITEMODE_PASTEL[ind],
                     line_alpha=0.2,
                     legend_label=f'Spectrum night {spectrum.meta["num_night"]}',
                     visible = True if ind == 0 else False,
                     )
        
        data_point_scatter = p.scatter(
            x='x',
            y='y',
            source= CDS,
            legend_label=f'Spectrum night {spectrum.meta["num_night"]}',
            visible = True if ind == 0 else False,
            line_color= COLORWHEEL_WHITEMODE_PASTEL[ind],
            line_alpha= 0.2,
            hatch_alpha= 0.2,
            fill_alpha= 0.2
        )
        
        binned_spectrum = sm.binning_spectrum(spectrum, 15, True)
        CDS_binned = _get_ColumnDataSource_from_spectrum(binned_spectrum)

        p.multi_line('x_values_error',
                     'errors',
                     source=CDS_binned,
                     color=COLORWHEEL_WHITEMODE_DARK[ind],
                     legend_label=f'Spectrum night {spectrum.meta["num_night"]}',
                     visible = True if ind == 0 else False,
                     )
        
        data_point_scatter_binned = p.scatter(
            x='x',
            y='y',
            source= CDS_binned,
            legend_label=f'Spectrum night {spectrum.meta["num_night"]}',
            visible = True if ind == 0 else False,
            line_color= COLORWHEEL_WHITEMODE_DARK[ind],
        )
        
        binning_slider.js_on_change("value",
                                    bokeh.models.CustomJS(
                                        args=dict(
                                            source_spectrum= CDS,
                                            source_binned= CDS_binned,
                                            slider=binning_slider),
                                        code= BINNING_JS_CODE)
                                    )
    
    if first_spectrum.spectral_axis.unit.is_equivalent(u.AA):
        for line in linelist.lines:
            p.vspan(
                x=[line.value],
                line_width=[2],
                line_color="black",
                line_dash= 'dashed'
            )
    elif first_spectrum.spectral_axis.unit.is_equivalent(u.m/u.s):
        p.vspan(
            x=[0],
            line_width=[2],
            line_color="black",
            line_dash= 'dashed'
        )

    
    p.legend.location = "top_right"
    p.legend.click_policy="hide"
    p.xaxis.axis_label = x_axis_label 
    p.yaxis.axis_label = y_axis_label
    p.title.text = f"{first_spectrum.meta['type']} : {linelist.name}"
    
    layout = bokeh.layouts.column(p, binning_slider, sizing_mode='scale_height') 
    blt.curdoc().theme = 'caliber'  # Apply the theme
    # blt.show(layout)
    blt.save(layout)
    

def plot_all_lines_transmission_spectrum(spectrum_list: sp.SpectrumList,
                                         prepend: str = '',
                                         normalization: bool = False,
                                         polynomial_order: None | int = None,
                                         velocity_fold: bool = False,
                                         ):
    """
    Plots all resolvable lines in transmission spectrum in separate plots

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list, should be transmission spectrum. It expects a list with first spectrum being combined transmission spectrum, and the rest being night specific. 
    prepend : str, optional
        Prepend to the file, useful to differentiate between masked/unmasked version of plot for example, by default ''.
    """
    
    for line in ratslist.RESOLVED_LINE_LIST:
        master_list(
            spectrum_list= spectrum_list,
            linelist= line,
            prepend= prepend,
            normalization= normalization,
            polynomial_order= polynomial_order,
            velocity_fold = velocity_fold
            )
    return


def fiber_B_correction(fiber_A_master: sp.SpectrumList,
                       fiber_B_master: sp.SpectrumList,
                       mode: str = 'whitemode_normal',
                       prepend: str = '',
                       ):
    if prepend != '':
        prepend = prepend + '_'

    os.makedirs(f'./figures/{mode}/master_list/fiber_comparison/',
                mode = 0o777,
                exist_ok = True
                )

    x_axis_label, y_axis_label = _get_axis_labels(fiber_A_master[0])

    for ind, (spectrum_A, spectrum_B) in enumerate(zip(fiber_A_master, fiber_B_master)):
        if ind != 0:
            num_night = ind
        else:
            num_night = 'combined'
        blt.output_file(
        filename=f'./figures/{mode}/master_list/fiber_comparison/{prepend}fiber_comparison_night_{num_night}.html'
        )
        p = blt.figure(width=1600, height=1200)
        
        CDS_A = _get_ColumnDataSource_from_spectrum(spectrum_A)
        CDS_B = _get_ColumnDataSource_from_spectrum(spectrum_B)
        
        scatter_A = p.line(
            x='x',
            y='y',
            source= CDS_A,
            legend_label=f'Spectrum night {spectrum_A.meta["num_night"]}',
            visible = True,
            line_color= COLORWHEEL_WHITEMODE_DARK[0],
            )
        
        scatter_B = p.line(
            x='x',
            y='y',
            source= CDS_B,
            legend_label=f'Spectrum night {spectrum_B.meta["num_night"]}',
            visible = True,
            line_color= COLORWHEEL_WHITEMODE_DARK[1],
            )

        p.legend.location = "top_right"
        p.xaxis.axis_label = x_axis_label 
        p.yaxis.axis_label = y_axis_label
        p.title.text = f"Fiber A and B comparison for night: {num_night}" #type:ignore
        blt.curdoc().theme = 'caliber'
        blt.save(p)

    ...
    
def telluric_correction(telluric_corrected_master: sp.SpectrumList,
                        telluric_profile_master: sp.SpectrumList,
                        uncorrected_master: sp.SpectrumList):
    
    ...
