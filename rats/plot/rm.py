#%% Importing libraries
import matplotlib.pyplot as plt
import os
import specutils as sp
import numpy as np
import seaborn as sns
import rats.spectra_manipulation as sm
import arviz as az
import astropy.units as u
from matplotlib import patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% Plotting RV plot during transit
def plot_RV(spectrum_list: sp.SpectrumList):
    """
    Plot all RV from the spectrum list header.
    
    The plot is saved automatically in the ./figures/whitemode_normal/RV_plot directory.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list from which to draw the RV. Requires the spectrum to hold the header and its RV and RV ERROR keywords. Furthermore it requires the Phase key to be calculated beforehand.
    """
    
    fig, ax = plt.subplots(1)
    
    for ind, night in enumerate(np.unique([item.meta['Night'] for item in spectrum_list])):
        sublist = sm.get_sublist(spectrum_list, 'Night', night)
        phase = [item.meta['Phase'].data  for item in sublist]
        rv = [item.meta['header']['HIERARCH ESO QC CCF RV'] for item in sublist]
        rv_error = [item.meta['header']['HIERARCH ESO QC CCF RV ERROR']  for item in sublist]
        
        ax.errorbar(
            phase,
            rv,
            rv_error,
            fmt= '.',
            alpha=0.6,
            label= f'{sublist[0].meta["instrument"]} : {night}'
            )
        
    ax.legend()
    ax.set_xlabel('Phase [1]')
    ax.set_ylabel('Radial velocity [km/s]')
        
    os.makedirs(f'./figures/whitemode_normal/RV_plot',
            mode = 0o777,
            exist_ok = True
            )
    
    return fig, ax 

def _plot_posterior(data_chain: list,
                    type: str,
                    ):
    fig_size = (15,5)

    # Create a 10xn grid based on number of total spectra
    if divmod(len(data_chain), 10)[1] != 0:
        fig, axs = plt.subplots(
            divmod(len(data_chain),10)[0]+1,10,
            sharex=True,
            sharey=True,
            figsize=fig_size
            )
    else:
        fig, axs = plt.subplots(
            divmod(len(data_chain),10)[0],10,
            sharex=True,
            sharey=True,
            figsize=fig_size
            )
    
    
    for ind, test_chain in enumerate(data_chain):
        match type:
            case 'rvcenter':
                az.plot_dist(
                    test_chain.posterior.rvcenter,
                    ax = axs[divmod(ind,10)],
                    color = sns.color_palette("dark")[test_CCF[ind].meta['Night_num']-1],
                    label = test_CCF[ind].meta['Spec_num']
                    )
                axs[divmod(ind,10)].axvline(
                    test_chain.posterior.rvcenter.median(),
                    color='black',
                    ls ='--',
                    label='_nolegend_'
                    )
                
            case 'fwhm':
                az.plot_dist(
                    test_chain.posterior.fwhm,
                    ax=axs[divmod(ind,10)],
                    color=sns.color_palette("dark")[test_CCF[ind].meta['Night_num']-1],
                    label = test_CCF[ind].meta['Spec_num']
                    )
                axs[divmod(ind,10)].axvline(
                    test_chain.posterior.fwhm.median(),color='black', ls ='--', label='_nolegend_')
        axs[divmod(ind,10)].legend()
        axs[divmod(ind,10)].legend()
        axs[divmod(ind,10)].label_outer(remove_inner_ticks=True)
        axs[divmod(ind,10)].set_yticks([])
        axs[divmod(ind,10)].set_xticks([])
    fig.supxlabel('Center of Gaussian profile [km/s]')
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    fig.savefig('./figures/whitemode_normal/posterior_rvcenter.pdf')



def plot_posterior_local_CCF(all_data_chain:list,
                             data_raw_A: sp.SpectrumList):
    test_CCF = [item for item in data_raw_A if (item.meta['Transit_partial'])]
    fig_size = (15,5)

    # Create a 10xn grid based on number of total spectra
    if divmod(len(all_data_chain), 10)[1] != 0:
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0]+1,10, sharex=False, sharey=False,figsize=fig_size, squeeze=False)
    else:
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0],10, sharex=False, sharey=False,figsize=fig_size, squeeze=False)
    
    for ind, test_chain in enumerate(all_data_chain):
        az.plot_dist(
            test_chain.posterior.rvcenter,
            ax=axs[divmod(ind,10)],
            color=sns.color_palette("dark")[test_CCF[ind].meta['Night_num']-1],
            label = test_CCF[ind].meta['Spec_num']
            )
        axs[divmod(ind,10)].axvline(test_chain.posterior.rvcenter.median(),color='black', ls ='--', label='_nolegend_')
        axs[divmod(ind,10)].legend()
        axs[divmod(ind,10)].label_outer(remove_inner_ticks=True)
        axs[divmod(ind,10)].set_yticks([])
        axs[divmod(ind,10)].set_xticks([])
    fig.supxlabel('Center of Gaussian profile [km/s]')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig('./figures/whitemode_normal/posterior_rvcenter.pdf')
    
    if divmod(len(all_data_chain), 10)[1] != 0:
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0]+1,10, sharex=False, sharey=False,figsize=fig_size, squeeze=False)
    else:
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0],10, sharex=False, sharey=False,figsize=fig_size, squeeze=False)
    
    for ind, test_chain in enumerate(all_data_chain):
        az.plot_dist(test_chain.posterior.contrast,
                     ax=axs[divmod(ind,10)],
                     color=sns.color_palette("dark")[test_CCF[ind].meta['Night_num']-1],
                     label = test_CCF[ind].meta['Spec_num'])
        axs[divmod(ind,10)].axvline(test_chain.posterior.contrast.median(),color='black', ls ='--', label='_nolegend_')
        axs[divmod(ind,10)].legend()
        axs[divmod(ind,10)].label_outer(remove_inner_ticks=True)
        axs[divmod(ind,10)].set_yticks([])
        axs[divmod(ind,10)].set_xticks([])
    fig.supxlabel('Contrast of the Gaussian profile [unitless]')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig('./figures/whitemode_normal/posterior_contrast.pdf')
    
    if divmod(len(all_data_chain), 10)[1] != 0:
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0]+1,10, sharex=False, sharey=False,figsize=fig_size, squeeze=False)
    else:
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0],10, sharex=False, sharey=False,figsize=fig_size, squeeze=False)
    for ind, test_chain in enumerate(all_data_chain):
        az.plot_dist(test_chain.posterior.fwhm,
                     ax=axs[divmod(ind,10)],
                     color=sns.color_palette("dark")[test_CCF[ind].meta['Night_num']-1],
                     label = test_CCF[ind].meta['Spec_num']
                     )
        axs[divmod(ind,10)].axvline(test_chain.posterior.fwhm.median(),color='black', ls ='--', label='_nolegend_')
        axs[divmod(ind,10)].legend()
        axs[divmod(ind,10)].label_outer(remove_inner_ticks=True)
        axs[divmod(ind,10)].set_yticks([])
        axs[divmod(ind,10)].set_xticks([])
    fig.supxlabel('Full-Width Half Maximum [km/s]')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig('./figures/whitemode_normal/posterior_fwhm.pdf')


import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

def model_velocity(CCF: sp.Spectrum1D,
                   system_parameters,
                   revolutions_result,
                   ):
    if CCF.meta is None:
        raise ValueError('CCF does not have meta parameters')
    
    if 'contrast' in revolutions_result.posterior.data_vars:
        contrast = revolutions_result.posterior.contrast.mean().to_numpy()
    else:
        contrast = revolutions_result.posterior[f"contrast_night{CCF.meta['Night']}_0"].mean().to_numpy()
        if f"contrast_night{CCF.meta['Night']}_1" in revolutions_result.posterior.data_vars:
            raise NotImplementedError('High-order values not implemented')
        
    if 'fwhm' in revolutions_result.posterior.data_vars:
        fwhm = revolutions_result.posterior.fwhm.mean().to_numpy()
    else:
        fwhm = revolutions_result.posterior[f"fwhm_night{CCF.meta['Night']}_0"].mean().to_numpy()
        if f"fwhm_night{CCF.meta['Night']}_1" in revolutions_result.posterior.data_vars:
            raise NotImplementedError('High-order values not implemented')

    veqsini = revolutions_result.posterior.veqsini.mean().to_numpy()
    obliquity = revolutions_result.posterior.obliquity.mean().to_numpy()
    
    if CCF.meta is None:
        raise ValueError('CCF does not have meta parameters')
    
    phase = CCF.meta['Phase'].data
    aRs = (system_parameters.Planet.semimajor_axis.divide(system_parameters.Star.radius)).convert_unit_to(u.dimensionless_unscaled).data
    
    
    x_p = aRs * np.sin(2*np.pi * phase)
    y_p = aRs * np.cos(2*np.pi * phase) * np.cos(system_parameters.Planet.inclination.data/180*np.pi)
    
    x_perpendicular = x_p* np.cos(obliquity/180*np.pi) - y_p * np.sin(obliquity/180*np.pi)
    y_perpendicular = x_p * np.sin(obliquity/180*np.pi) - y_p * np.cos(obliquity/180*np.pi)
    
    local_stellar_velocity = x_perpendicular * veqsini
    
    
    expected = (-contrast * np.exp(-((CCF.spectral_axis.value - local_stellar_velocity)**2)/(2*((fwhm/2.35482)**2))) + 1)
    return expected

def get_gaussian_fit(CCF, system_parameters, result_Revolutions):
    
    expected_x = CCF.spectral_axis.value
    expected_y = model_velocity(CCF, system_parameters, result_Revolutions)
    
    return expected_x, expected_y

class CompareIntr_vs_RevolutionsFit:
    def __init__(self, CCF_list, system_parameters, result):
        self.CCF_list = CCF_list
        self.result = result
        self.system_parameters = system_parameters
        self.current_index = 0
        self.n_spectra = len(CCF_list)

        # Create the main figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.2)  # Make room for buttons

        # Create buttons
        self.ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')

        self.text_artist = self.fig.text(0.05, 0.05, '', transform=self.fig.transFigure)

        # Connect button click events
        self.btn_prev.on_clicked(self.previous_spectrum)
        self.btn_next.on_clicked(self.next_spectrum)

        # Plot initial spectrum
        self.update_plot()

    def update_text(self, new_text):
        # Update the text
        self.text_artist.set_text(new_text)

    def update_plot(self):
        self.ax.clear()
        CCF = self.CCF_list[self.current_index]
        result = self.result

        # Plot CCF with error bars
        self.ax.errorbar(CCF.spectral_axis, CCF.flux, CCF.uncertainty.array, fmt='.')
        self.ax.set_ylim(-0.2, 1.4)
        
        # Plot Gaussian fit
        x, y = get_gaussian_fit(CCF, self.system_parameters, result)
        self.ax.plot(x, y, color='darkred')
        
        match (CCF.meta['Transit_partial'], CCF.meta['Transit_full']):
            case (True, True):
                self.ax.set_facecolor('#A8DAB5')
            case (True, False):
                self.ax.set_facecolor('#FFF59D')
            case (False, False):
                self.ax.set_facecolor('#F8BBD0')
        # Update title
        self.ax.set_title(f'Spectrum {self.current_index}/{self.n_spectra -1}')
        
        # Update text (example: showing current spectrum info)
        self.update_text(f'Spectrum index: {CCF.meta.get("Spec_num", "N/A")}\n Night: {CCF.meta.get("Night", "N/A")} \n Revolutions fit: {CCF.meta.get("Revolutions", "N/A")}')

        # Refresh the plot
        self.fig.canvas.draw_idle()

    def next_spectrum(self, event):
        if self.current_index < self.n_spectra - 1:
            self.current_index += 1
            self.update_plot()

    def previous_spectrum(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot()

class CompareIntr_vs_RevolutionsFit_vs_localfit:
    def __init__(self, CCF_list, system_parameters, result, local_result):
        self.CCF_list = CCF_list
        self.result = result
        self.local_result = local_result
        self.system_parameters = system_parameters
        self.current_index = 0
        self.n_spectra = len(CCF_list)

        # Create the main figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.2)  # Make room for buttons

        # Create buttons
        self.ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')

        self.text_artist = self.fig.text(0.05, 0.05, '', transform=self.fig.transFigure)

        # Connect button click events
        self.btn_prev.on_clicked(self.previous_spectrum)
        self.btn_next.on_clicked(self.next_spectrum)

        # Plot initial spectrum
        self.update_plot()

    def update_text(self, new_text):
        # Update the text
        self.text_artist.set_text(new_text)

    def update_plot(self):
        self.ax.clear()
        CCF = self.CCF_list[self.current_index]
        result = self.result

        # Plot CCF with error bars
        self.ax.errorbar(CCF.spectral_axis, CCF.flux, CCF.uncertainty.array, fmt='.')
        self.ax.set_ylim(-0.2, 1.4)
        
        # Plot Gaussian fit
        x, y = get_gaussian_fit(CCF, self.system_parameters, result)
        self.ax.plot(x, y, color='darkred')
        
        match (CCF.meta['Transit_partial'], CCF.meta['Transit_full']):
            case (True, True):
                self.ax.set_facecolor('#A8DAB5')
            case (True, False):
                self.ax.set_facecolor('#FFF59D')
            case (False, False):
                self.ax.set_facecolor('#F8BBD0')
        # Update title
        self.ax.set_title(f'Spectrum {self.current_index}/{self.n_spectra -1}')
        
        # Update text (example: showing current spectrum info)
        self.update_text(f'Spectrum index: {self.current_index}\n Night: {CCF.meta.get("Night", "N/A")} \n Revolutions fit: {CCF.meta.get("Revolutions", "N/A")}')

        # Refresh the plot
        self.fig.canvas.draw_idle()

    def next_spectrum(self, event):
        if self.current_index < self.n_spectra - 1:
            self.current_index += 1
            self.update_plot()

    def previous_spectrum(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot()

#%%
def plot_Gaussian_values(data_chain_list: list,
                         CCF_intrinsic: sp.SpectrumList,
                         system_parameters,
                         CCF_intrinsic_full: sp.SpectrumList,
                         ):
    in_transit = [CCF for CCF in CCF_intrinsic if CCF.meta['Transit_partial']]
    import arviz as az
    import matplotlib as mpl
    sns.set_context('talk')
    plt.ion()
    mpl.use('TkAgg')
    colors = sns.color_palette('dark')
    
    considered_indices = [CCF.meta['Spec_num'] for CCF in CCF_intrinsic_full if CCF.meta['Transit_partial']]
    
    fig, axs = plt.subplots(3,2)
    for CCF, data_chain in zip(in_transit, data_chain_list):
        # FIXME
        # This is probably wrong formula, Whops! Also, move this to a meta keyword instead
        mu = abs(system_parameters.Planet.a_rs_ratio * np.sin(2*np.pi * CCF.meta['Phase'].data))
        
        # if mu > 0.6:
        #     continue
        
        # if CCF.meta['Spec_num'] in [6, 22, 48, 49, 62, 78]:
        #     continue
        
        # if CCF.meta['Spec_num'] not in considered_indices:
        #     continue
        
        contrast_error = np.asarray([
            (data_chain.posterior.contrast.median().to_numpy() - az.hdi(data_chain.posterior.contrast, hdi_prob=0.683)['contrast'].to_numpy())[0],
            (- data_chain.posterior.contrast.median().to_numpy() + az.hdi(data_chain.posterior.contrast, hdi_prob=0.683)['contrast'].to_numpy())[1]
            ]).reshape(2,1)
        fwhm_error = np.asarray([
            (data_chain.posterior.fwhm.median().to_numpy() - az.hdi(data_chain.posterior.fwhm, hdi_prob=0.683)['fwhm'].to_numpy())[0],
            (- data_chain.posterior.fwhm.median().to_numpy() + az.hdi(data_chain.posterior.fwhm, hdi_prob=0.683)['fwhm'].to_numpy())[1]
            ]).reshape(2,1)
        
        rvcenter_error = np.asarray([
            (data_chain.posterior.rvcenter.median().to_numpy() - az.hdi(data_chain.posterior.rvcenter, hdi_prob=0.683)['rvcenter'].to_numpy())[0],
            (- data_chain.posterior.rvcenter.median().to_numpy() + az.hdi(data_chain.posterior.rvcenter, hdi_prob=0.683)['rvcenter'].to_numpy())[1]
            ]).reshape(2,1)
        
        axs[0,0].errorbar(CCF.meta['Phase'].data,
                        data_chain.posterior.contrast.median().to_numpy(),
                        contrast_error,
                        color= colors[CCF.meta['Night_num']-1], fmt='.')
        
        axs[1,0].errorbar(CCF.meta['Phase'].data,
                        data_chain.posterior.fwhm.median().to_numpy(),
                        fwhm_error,
                        color= colors[CCF.meta['Night_num']-1], fmt='.'
                        )
        
        axs[2,0].errorbar(CCF.meta['Phase'].data,
                        data_chain.posterior.rvcenter.median().to_numpy(),
                        rvcenter_error,
                        color= colors[CCF.meta['Night_num']-1], fmt='.'
                        )
        

        
        axs[0,1].errorbar(mu,
                        data_chain.posterior.contrast.median().to_numpy(),
                        contrast_error,
                        color= colors[CCF.meta['Night_num']-1], fmt='.')
        
        axs[1,1].errorbar(mu,
                        data_chain.posterior.fwhm.median().to_numpy(),
                        fwhm_error,
                        color= colors[CCF.meta['Night_num']-1], fmt='.'
                        )
        axs[2,1].errorbar(mu,
                        data_chain.posterior.rvcenter.median().to_numpy(),
                        rvcenter_error,
                        color= colors[CCF.meta['Night_num']-1], fmt='.'
                        )
        
    phase_model = np.linspace(axs[0,0].get_xlim()[0],
                             axs[0,0].get_xlim()[1],
                             1000)
    import astropy
    system_parameters.Planet.projected_obliquity = astropy.nddata.NDDataArray(
        -0.26 * u.deg,
        uncertainty= astropy.nddata.StdDevUncertainty(0.31)
    )
    rv_loc = []
    for phase in phase_model:
        rv_loc.append(system_parameters._local_stellar_velocity(phase)) 
    rv_loc_value = np.asarray([rv.data for rv in rv_loc])
    rv_loc_error = np.asarray([rv.uncertainty.array for rv in rv_loc])
    
    axs[2,0].plot(phase_model, rv_loc_value, color='black')
    axs[2,0].fill_between(phase_model, rv_loc_value-rv_loc_error, rv_loc_value+rv_loc_error, color='green', alpha=0.2)
    
    system_parameters.Planet.projected_obliquity = astropy.nddata.NDDataArray(
        -8*u.deg,
        uncertainty= astropy.nddata.StdDevUncertainty(0.34)
    )
    rv_loc = []
    for phase in phase_model:
        rv_loc.append(system_parameters._local_stellar_velocity(phase)) 
    rv_loc_value = np.asarray([rv.data for rv in rv_loc])
    rv_loc_error = np.asarray([rv.uncertainty.array for rv in rv_loc])
    
    axs[2,0].plot(phase_model, rv_loc_value, color='black')
    axs[2,0].fill_between(phase_model, rv_loc_value-rv_loc_error, rv_loc_value+rv_loc_error, color='red', alpha=0.2)
    
    axs[0,0].set_ylabel('Contrast [unitless]')
    axs[1,0].set_ylabel('FWHM [km/s]')
    axs[2,0].set_ylabel('RV center [km/s]')
    axs[2,0].set_xlabel('Phase')
    axs[2,1].set_xlabel('$\mu$')
    fig.savefig('./figures/whitemode_normal/Gaussian_values_plot.pdf')

def _get_residuals(CCF_intrinsic,
                   system_parameters,
                   revolutions_result,
                   ):
    CCF_residual = sp.SpectrumList()
    
    for CCF in CCF_intrinsic:
        residual_flux = CCF.flux.value - model_velocity(CCF, system_parameters, revolutions_result)
        
        residual = sp.Spectrum1D(
            spectral_axis = CCF.spectral_axis,
            flux = residual_flux * u.dimensionless_unscaled,
            uncertainty = CCF.uncertainty,
            meta = CCF.meta.copy() if CCF.meta is not None else None
        )
        
        CCF_residual.append(residual)
    
    return CCF_residual


def plot_CCF_night(CCF_list, night_num, system_parameters, ax, cmap='magma_r', vmin=0.2, vmax=1.2):
    """
    Helper function to plot CCF data for a single night.

    Parameters
    ----------
    CCF_list : list
        List of CCF objects
    night_num : int
        Night number to plot
    ax : plt.Axes
        Matplotlib axes object to plot on
    cmap : str, optional
        Colormap to use for plotting (default: 'magma_r')
    vmin : float, optional
        Minimum value for color scaling (default: 0.2)
    vmax : float, optional
        Maximum value for color scaling (default: 1.2)

    Returns
    -------
    pcm : matplotlib.collections.QuadMesh
        The pcolormesh object created
    """
    # Get data for specific night
    CCF_night = sm.get_sublist(CCF_list, 'Night_num', night_num, mode='equal')
    CCF_night = sm.get_sublist(CCF_night, 'Transit_partial', True, mode='equal')

    # Extract plotting data
    x = CCF_night[0].spectral_axis
    y = [CCF.meta['Phase'].data for CCF in CCF_night]
    z = np.asarray([CCF.flux for CCF in CCF_night])

    # Create plot
    pcm = ax.pcolormesh(x.value, y, z, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_facecolor("black")
    system_parameters.plot_contact_points(ax, ls='--')
    
    for CCF in CCF_night:
        if not(CCF.meta['Revolutions']):
            # Plot the line in two segments
            ax.axhline(y=CCF.meta['Phase'].data,
                       xmin=0,
                       xmax=0.25,
                       color='red',
                       ls='solid',
                       linewidth=1.5,
                       path_effects=[pe.withStroke(linewidth=4, foreground='black')]
                       )
            ax.axhline(y=CCF.meta['Phase'].data,
                       xmin=0.75,
                       xmax=1,
                       color='red',
                       ls='solid',
                       linewidth=1.5,
                       path_effects=[pe.withStroke(linewidth=4, foreground='black')]
                       )

    return pcm

def plot_residual_CCF_night(CCF_list, night_num, system_parameters, revolutions_result, ax, 
                           cmap='RdBu_r', vmin=-0.15, vmax=0.15):
    """
    Helper function to plot CCF residual data for a single night.
    """
    CCF_night = sm.get_sublist(CCF_list, 'Night_num', night_num, mode='equal')
    CCF_night = sm.get_sublist(CCF_night, 'Transit_partial', True, mode='equal')

    CCF_residual = _get_residuals(CCF_night, system_parameters, revolutions_result)

    x = CCF_residual[0].spectral_axis.value
    y = [CCF.meta['Phase'].data for CCF in CCF_residual]
    z = np.asarray([CCF.flux for CCF in CCF_residual])
    
    pcm = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_facecolor("black")
    system_parameters.plot_contact_points(ax, ls='--')

    
    for CCF in CCF_night:
        if not(CCF.meta['Revolutions']):
            # Plot the line in two segments
            ax.axhline(y=CCF.meta['Phase'].data,
                       xmin=0,
                       xmax=0.25,
                       color='red',
                       ls='solid',
                       linewidth=1.5,
                       path_effects=[pe.withStroke(linewidth=4, foreground='black')]
                       )
            ax.axhline(y=CCF.meta['Phase'].data,
                       xmin=0.75,
                       xmax=1,
                       color='red',
                       ls='solid',
                       linewidth=1.5,
                       path_effects=[pe.withStroke(linewidth=4, foreground='black')]
                       )

    return pcm

def plot_instrinsic_CCF(CCF_intrinsic,
                        system_parameters,
                        fig:plt.Figure | None = None,
                        axs:plt.Axes | None | np.ndarray = None):
    num_night = len(np.unique([CCF.meta['Night_num'] for CCF in CCF_intrinsic]))

    if fig is None:
        fig, axs = plt.subplots(num_night, 1, sharex=True, sharey=True)

    pcm_list = []
    for ni in range(num_night):
        pcm = plot_CCF_night(CCF_intrinsic, ni+1, system_parameters, axs[ni])
        pcm_list.append(pcm)

    # Add a single colorbar for all subplots
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    plt.colorbar(pcm, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    
    # cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    # fig.colorbar(pcm_list[0], cax=cbar_ax)

    
    fig.supxlabel('Velocity [km/s]')
    fig.supylabel('Phase [1]')
    
    return

def plot_instrinsic_residuals_CCF(CCF_intrinsic,
                                  system_parameters,
                                  revolutions_result,
                                  fig:plt.Figure | None = None,
                                  axs: plt.Axes | None | np.ndarray = None):
    num_night = len(np.unique([CCF.meta['Night_num'] for CCF in CCF_intrinsic]))

    if fig is None:
        fig, axs = plt.subplots(num_night, 2, figsize=(16, 4*num_night),squeeze=False)

    # Plot left column (intrinsic CCF)
    pcm_list_left = []
    for ni in range(num_night):
        pcm = plot_CCF_night(CCF_intrinsic, ni+1, system_parameters, axs[ni,0])
        pcm_list_left.append(pcm)

        aRs = (system_parameters.Planet.semimajor_axis.divide(system_parameters.Star.radius)).convert_unit_to(u.dimensionless_unscaled).data
        veqsini = revolutions_result.posterior.veqsini.mean().to_numpy()
        obliquity = revolutions_result.posterior.obliquity.mean().to_numpy()
        
        phases = np.linspace(axs[ni,0].get_ylim()[0], axs[ni,0].get_ylim()[1], 1000)
        x_p = aRs * np.sin(2*np.pi * phases) # type: ignore
        y_p = aRs * np.cos(2*np.pi * phases) * np.cos(system_parameters.Planet.inclination.data / 180*np.pi) #type: ignore
        x_perpendicular = x_p* np.cos(obliquity/180*np.pi) - y_p * np.sin(obliquity/180*np.pi) #type: ignore
        
        local_stellar_velocity = x_perpendicular * veqsini
        
        axs[ni,0].plot(local_stellar_velocity, phases, color='black', ls='dotted')
        
    # Plot right column (residuals)
    pcm_list_right = []
    for ni in range(num_night):
        pcm = plot_residual_CCF_night(CCF_intrinsic, ni+1, system_parameters, 
                                     revolutions_result, axs[ni,1])
        pcm_list_right.append(pcm)
        axs[ni,1].plot(local_stellar_velocity, phases, color='black', ls='dotted')
    # Adjust layout and add colorbars
    fig.subplots_adjust(right=0.85, wspace=0.4)

    # Left colorbar (for intrinsic CCF)
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes("top", size="5%", pad=0.0)
    plt.colorbar(pcm_list_left[0], cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    
    # Right colorbar (for residuals)
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes("top", size="5%", pad=0.0)
    plt.colorbar(pcm_list_right[0], cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    
    fig.supxlabel('Velocity [km/s]')
    fig.supylabel('Phase [1]')
    # cbar_ax_right = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # fig.colorbar(pcm_list_right[0], cax=cbar_ax_right)

    return fig, axs

class SpectraViewer:
    def __init__(self, CCF_list, system_parameters):
        self.CCF_list = CCF_list
        self.system_parameters = system_parameters
        self.current_index = 0
        self.n_spectra = len(CCF_list)

        # Create the main figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.2)  # Make room for buttons

        # Create buttons
        self.ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')

        self.x_min = min(min(ccf.spectral_axis.value) for ccf in CCF_list)
        self.x_max = max(max(ccf.spectral_axis.value) for ccf in CCF_list)

        self.text_artist = self.fig.text(0.05, 0.05, '', transform=self.fig.transFigure)

        # Connect button click events
        self.btn_prev.on_clicked(self.previous_spectrum)
        self.btn_next.on_clicked(self.next_spectrum)

        # Plot initial spectrum
        self.update_plot()

    def update_text(self, new_text):
        # Update the text
        self.text_artist.set_text(new_text)

    def update_plot(self):
        self.ax.clear()
        CCF = self.CCF_list[self.current_index]

        # Plot CCF with error bars
        self.ax.errorbar(CCF.spectral_axis, CCF.flux, CCF.uncertainty.array, fmt='.')
        self.ax.set_ylim(0 , 1.1)
        self.ax.set_xlim(self.x_min, self.x_max)
        
        match (CCF.meta['Transit_partial'], CCF.meta['Transit_full']):
            case (True, True):
                self.ax.set_facecolor('#A8DAB5')
            case (True, False):
                self.ax.set_facecolor('#FFF59D')
            case (False, False):
                self.ax.set_facecolor('#F8BBD0')
        # Update title
        self.ax.set_title(f'Spectrum {self.current_index}/{self.n_spectra -1}')
        
        # Update text (example: showing current spectrum info)
        self.update_text(f'Spectrum index: {self.current_index}\n Night: {CCF.meta.get("Night", "N/A")} \n Revolutions fit: {CCF.meta.get("Revolutions", "N/A")}')

        # Refresh the plot
        self.fig.canvas.draw_idle()

    def next_spectrum(self, event):
        if self.current_index < self.n_spectra - 1:
            self.current_index += 1
            self.update_plot()

    def previous_spectrum(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot()
            
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation
import numpy as np

class RevolutionsSlides:
    def __init__(self, CCF_list, system_parameters, result, figsize=(19.2, 10.8)):
        self.CCF_list = CCF_list
        self.result = result
        self.system_parameters = system_parameters
        self.current_index = 0
        self.n_spectra = len(CCF_list)
        self.figsize = figsize
        self.animation_running = False
        self.anim = None

        # Initialize Dopplergram parameters
        self.star_params = {
            'radius': system_parameters.Star.radius.data,
            'vsini': system_parameters.Star.vsini.data,
            'inclination': system_parameters.Planet.inclination.data,
            'aRs': system_parameters.Planet.a_rs_ratio.data,
            'Rp_Rs': system_parameters.Planet.rprs.data,
            'obliquity': result.posterior.obliquity.mean().to_numpy(),
            'b': system_parameters.Planet.impact_parameter.data
        }

        # Create custom colormap for Dopplergram
        self.doppler_cmap = 'seismic_r'

        # Create figure with two subplots
        self.fig, self.axs = plt.subplots(2, 1, figsize=self.figsize, 
                                         height_ratios=[2, 1],
                                         gridspec_kw={'hspace': 0.3})
        plt.subplots_adjust(bottom=0.2)

        # Create buttons for interactive mode
        self.ax_prev = plt.axes([0.2, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.31, 0.05, 0.1, 0.075])
        self.ax_play = plt.axes([0.42, 0.05, 0.1, 0.075])
        self.ax_pause = plt.axes([0.53, 0.05, 0.1, 0.075])

        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_play = Button(self.ax_play, 'Play')
        self.btn_pause = Button(self.ax_pause, 'Pause')

        # Initialize text artist with proper position
        self.text_artist = self.fig.text(0.05, 0.02, '', 
                                        transform=self.fig.transFigure,
                                        verticalalignment='bottom')

        # Connect button click events
        self.btn_prev.on_clicked(self.previous_spectrum)
        self.btn_next.on_clicked(self.next_spectrum)
        self.btn_play.on_clicked(self.start_animation)
        self.btn_pause.on_clicked(self.pause_animation)

        # Calculate x-axis limits from all spectra
        self.x_min = min(min(ccf.spectral_axis.value) for ccf in CCF_list)
        self.x_max = max(max(ccf.spectral_axis.value) for ccf in CCF_list)

        # Plot initial spectrum
        self.update_plot()

    def calculate_orbit_points(self, n_points=1000):
        """Calculate orbit points for visualization"""
        phases = np.linspace(0, 1, n_points)
        aRs = self.star_params['aRs']
        inc = np.radians(self.star_params['inclination'])
        obl = np.radians(self.star_params['obliquity'])

        # Calculate orbital positions
        x_p = aRs * np.sin(2*np.pi * phases)
        y_p = aRs * np.cos(2*np.pi * phases) * np.cos(inc)

        # Transform to perpendicular coordinates
        x_perp = x_p * np.cos(obl) - y_p * np.sin(obl)
        y_perp = x_p * np.sin(obl) + y_p * np.cos(obl)

        return x_perp, y_perp

    def calculate_planet_position(self, phase):
        """Calculate current planet position"""
        aRs = self.star_params['aRs']
        inc = np.radians(self.star_params['inclination'])
        obl = np.radians(self.star_params['obliquity'])

        # Calculate orbital position
        x_p = aRs * np.sin(2*np.pi * phase)
        y_p = aRs * np.cos(2*np.pi * phase) * np.cos(inc)

        # Transform to perpendicular coordinates
        x_perp = x_p * np.cos(obl) - y_p * np.sin(obl)
        y_perp = -(x_p * np.sin(obl) + y_p * np.cos(obl))

        return x_perp, y_perp

    def create_dopplergram(self, resolution=1000):
        """Generate Dopplergram data"""
        x = np.linspace(-1.2, 1.2, resolution)
        y = np.linspace(-1.2, 1.2, resolution)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        mask = R <= 1.0

        vsini = self.star_params['vsini']
        inc = np.radians(self.star_params['inclination'])

        doppler_map = np.zeros_like(X)
        doppler_map[mask] = -vsini * X[mask] * np.sin(inc)

        return X, Y, doppler_map, mask
    
    def plot_dopplergram(self, ax, resolution=1000):
        """Plot Dopplergram on given axis"""
        X, Y, doppler_map, mask = self.create_dopplergram(resolution)
        masked_doppler = np.ma.array(doppler_map, mask=~mask)

        vmin, vmax = np.min(masked_doppler), np.max(masked_doppler)
        # Get current planet position
        CCF = self.CCF_list[self.current_index]
        phase = CCF.meta.get('Phase', 0).data
        x_planet, y_planet = self.calculate_planet_position(phase)
        planet_radius = self.star_params['Rp_Rs']

        # Calculate distance from planet center to star center
        planet_center_distance = np.sqrt(x_planet**2 + y_planet**2)

        # First plot the full dopplergram with dim alpha
        im = ax.imshow(masked_doppler, 
                    cmap=self.doppler_cmap,
                    origin='lower',
                    extent=[-1.2, 1.2, -1.2, 1.2],
                    aspect='auto',
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.3)  # Set base alpha to 0.3 for dimming

        # Add stellar disk outline
        stellar_outline = plt.Circle((0, 0), 1.0, 
                                color='black', 
                                fill=False, 
                                linewidth=0.5)  # thin line
        ax.add_patch(stellar_outline)

        # If planet is in front
        if y_planet < 0:
            # Create planet mask
            xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, resolution), 
                                np.linspace(-1.2, 1.2, resolution))
            planet_mask = ((xx - x_planet)**2 + (yy - y_planet)**2 <= planet_radius**2)

            # Create stellar disk mask
            stellar_mask = np.sqrt(xx**2 + yy**2) <= 1.0

            # Region where planet overlaps with star
            overlap_mask = planet_mask & stellar_mask

            # Plot the bright region where planet overlaps with star
            masked_planet_doppler = np.ma.array(masked_doppler,
                                            mask=~overlap_mask)
            ax.imshow(masked_planet_doppler,
                    cmap=self.doppler_cmap,
                    origin='lower',
                    extent=[-1.2, 1.2, -1.2, 1.2],
                    aspect='auto',
                    vmin=vmin,
                    vmax=vmax,
                    alpha=1.0)  # Full brightness for overlap region

            # Plot black region where planet is outside star
            outside_mask = planet_mask & ~stellar_mask
            if outside_mask.any():  # Only if there's any region outside
                black_region = np.ma.array(np.zeros_like(X),
                                        mask=~outside_mask)
                ax.imshow(black_region,
                        cmap='gray',
                        origin='lower',
                        extent=[-1.2, 1.2, -1.2, 1.2],
                        aspect='auto',
                        alpha=1.0)  # Solid black

        # Plot stellar rotation plane in black
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.8)

        # Calculate and plot orbital path
        x_orbit, y_orbit = self.calculate_orbit_points(n_points=200)
        visible_mask = (y_orbit < 0) | ((x_orbit**2 + y_orbit**2) > 1)
        ax.plot(x_orbit[visible_mask], y_orbit[visible_mask], '--', 
                color='darkgreen', alpha=0.8, label='Orbit')

        # Plot planet outline or filled circle
        planet_outside_star = planet_center_distance > (1.0 + planet_radius)
        if y_planet < 0 and not planet_outside_star:
            planet_outline = plt.Circle((x_planet, y_planet), planet_radius, 
                                    color='black', fill=False, 
                                    linewidth=2)
            ax.add_patch(planet_outline)
        elif planet_outside_star:
            planet = plt.Circle((x_planet, y_planet), planet_radius, 
                            color='black', fill=True)
            ax.add_patch(planet)

        # Set labels and title
        ax.set_xlabel('X (R$_*$)')
        ax.set_ylabel('Y (R$_*$)')
        ax.set_title('RM Revolutions analysis')

        # Get the axes dimensions in figure coordinates (0 to 1)
        bbox = ax.get_position()
        axes_width = bbox.width
        axes_height = bbox.height

        # Calculate aspect ratio of the axes
        aspect_ratio = axes_width / axes_height
        aspect_ratio = 19.2 / 10.8
        # Set limits
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(aspect_ratio * ax.get_ylim()[0], aspect_ratio * ax.get_ylim()[1])

        return im



    def update_plot(self):
        """Update both plots"""
        self.axs[0].clear()
        self.axs[1].clear()

        self.plot_dopplergram(self.axs[0])

        CCF = self.CCF_list[self.current_index]
        result = self.result

        # Set background color based on transit state before plotting
        if CCF.meta.get('Transit_partial', False) and CCF.meta.get('Transit_full', False):
            self.axs[1].set_facecolor('#A8DAB5')
        elif CCF.meta.get('Transit_partial', False):
            self.axs[1].set_facecolor('#FFF59D')
        else:
            self.axs[1].set_facecolor('#F8BBD0')

        self.axs[1].errorbar(CCF.spectral_axis.value, CCF.flux.value, CCF.uncertainty.array, fmt='.')
        self.axs[1].set_ylim(-0.2, 1.4)
        self.axs[1].set_xlim(self.x_min, self.x_max)

        x, y = get_gaussian_fit(CCF, self.system_parameters, result)
        self.axs[1].plot(x, y, color='darkred')

        # Add vertical lines for vsini
        vsini = self.star_params['vsini']
        self.axs[1].axvline(x=-vsini, color='darkblue', linestyle='--', alpha=0.8)
        self.axs[1].axvline(x=0, color='black', linestyle='--', alpha=0.8)
        self.axs[1].axvline(x=vsini, color='darkred', linestyle='--', alpha=0.8)

        self.axs[1].set_title(f'Spectrum {self.current_index}/{self.n_spectra -1}')

        # Update text
        text_content = (f'Spectrum index: {self.current_index}\n'
                    f'Night: {CCF.meta.get("Night", "N/A")}\n'
                    f'Revolutions fit: {CCF.meta.get("Revolutions", "N/A")}')
        self.text_artist.set_text(text_content)

        self.fig.canvas.draw_idle()


    def start_animation(self, event):
        """Start the animation"""
        if not self.animation_running:
            self.anim = animation.FuncAnimation(self.fig, self._animation_update,
                                              frames=self.n_spectra,
                                              interval=500,  # 500ms between frames
                                              repeat=True)
            self.animation_running = True

    def pause_animation(self, event):
        """Pause the animation"""
        if self.animation_running and self.anim is not None:
            self.anim.event_source.stop()
            self.animation_running = False

    def next_spectrum(self, event):
        if self.current_index < self.n_spectra - 1:
            self.current_index += 1
            self.update_plot()

    def previous_spectrum(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot()

    def _animation_update(self, frame):
        self.current_index = frame
        self.update_plot()
        return self.axs[0], self.axs[1], self.text_artist

    def save_video(self, filename='animation.mp4', fps=2):
        self.current_index = 0
        anim = animation.FuncAnimation(self.fig, self._animation_update,
                                     frames=self.n_spectra,
                                     interval=1000/fps)
        anim.save(filename, writer='ffmpeg', fps=fps)

    def save_frames(self, output_dir='presentation_frames', 
                   dpi=300, format='png', transparent=False):
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        original_index = self.current_index

        for i in range(self.n_spectra):
            self.current_index = i
            self.update_plot()

            filename = f"{output_dir}/frame_{i:04d}.{format}"
            self.fig.savefig(filename, 
                            dpi=dpi,
                            bbox_inches='tight',
                            facecolor='white' if not transparent else 'none',
                            transparent=transparent)

            print(f"Saved frame {i+1}/{self.n_spectra}")

        self.current_index = original_index
        self.update_plot()

    def save_html(self, filename='interactive_plot.html'):
        import mpld3
        mpld3.save_html(self.fig, filename)
