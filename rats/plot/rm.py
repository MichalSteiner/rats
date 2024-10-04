#%% Importing libraries
import matplotlib.pyplot as plt
import os
import specutils as sp
import numpy as np
import seaborn as sns
import rats.spectra_manipulation as sm
import arviz as az
import astropy.units as u
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
    
    fig.savefig('./figures/whitemode_normal/RV_plot/RV_plot.pdf')


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
    
#%%
def plot_Gaussian_values(data_chain_list: list,
                         CCF_intrinsic: sp.SpectrumList,
                         system_parameters):
    in_transit = [CCF for CCF in CCF_intrinsic if CCF.meta['Transit_partial']]
    import arviz as az
    colors = sns.color_palette('dark')
    
    fig, axs = plt.subplots(3,2)
    for CCF, data_chain in zip(in_transit, data_chain_list):
        if CCF.meta['Spec_num'] in [6, 22, 48, 49, 62, 78]:
            continue
        
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
        
        # FIXME
        # This is probably wrong formula, Whops! Also, move this to a meta keyword instead
        mu = abs(system_parameters.Planet.a_rs_ratio * np.sin(2*np.pi * CCF.meta['Phase'].data))
        
        
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
    # system_parameters.Star.vsini = astropy.nddata.NDDataArray(
    #     7.4 * u.deg,
    #     uncertainty= astropy.nddata.StdDevUncertainty(0.1)
    # )
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
    axs[2,1].set_xlabel('$\mu$ CHECK FORMULA')
    fig.savefig('./figures/whitemode_normal/Gaussian_values_plot.pdf')
