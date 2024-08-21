#%% Importing libraries
import matplotlib.pyplot as plt
import os
import specutils as sp
import numpy as np
import seaborn as sns
import rats.spectra_manipulation as sm
import arviz as az
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
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0]+1,10, sharex=False, sharey=False,figsize=fig_size)
    else:
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0],10, sharex=False, sharey=False,figsize=fig_size)
    
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
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0]+1,10, sharex=False, sharey=False,figsize=fig_size)
    else:
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0],10, sharex=False, sharey=False,figsize=fig_size)
    
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
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0]+1,10, sharex=False, sharey=False,figsize=fig_size)
    else:
        fig, axs = plt.subplots(divmod(len(all_data_chain),10)[0],10, sharex=False, sharey=False,figsize=fig_size)
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