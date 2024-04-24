# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:11:02 2021

@author: Chamaeleontis
"""

#%% Importing libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import matplotlib.gridspec as gs
import matplotlib.animation
import seaborn as sns
import specutils as sp
import numpy as np
import matplotlib.ticker as mtick
import astropy.io.fits as fits
import astropy.constants as con
import typing
import pandas as pd
import astropy.units as u
import rats.spectra_manipulation as sm
from rats.utilities import time_function, save_and_load, progress_tracker, disable_func, skip_function
#%% Color_pallete
color_pallete_dark = sns.color_palette("dark")
color_pallete_pastel = sns.color_palette("pastel")
color_pallete_muted = sns.color_palette("muted")
color_pallete_deep = sns.color_palette("deep")
color_pallete_bright = sns.color_palette("bright")

#%% plot_transmission
def plot_transmission(trans_list,color_pallete=color_pallete_dark):
    '''
    Plot transmission spectrum
    '''
    num_plots = len(trans_list)
    if num_plots == 2:
        num_plots = 1
    fig,axs = plt.subplots(num_plots,sharex=True, sharey=True)
    
    if num_plots == 1:
        axs = [axs]
        trans_list = [trans_list[0]]
    for ind,tran in enumerate(trans_list):
        axs[ind].errorbar(tran.spectral_axis.value,
                          tran.flux.value,
                          tran.uncertainty.array,
                          color =color_pallete[0],
                          label = 'Transmission',
                          )
    axs[0].set_title('Transmission spectrum')
    return fig,axs
#%% plot_intransit
def plot_intransit(trans_list,bins=25,fig=None,axs=None):
    '''
    Plot transmission spectrum
    '''

    
    num_plots = len(trans_list)
    if num_plots == 2:
        num_plots = 1
    if fig is None:
        fig,axs = plt.subplots(num_plots,sharex=True, sharey=True)
    if num_plots == 1:
        axs = [axs]

    for ii in range(num_plots):
        
        axs[ii].errorbar(trans_list[ii].spectral_axis.value,
                          trans_list[ii].flux.value,
                          trans_list[ii].uncertainty.array,
                          color = color_pallete_pastel[7],
                          label = '_nolegend_',
                          zorder = 500,
                          fmt='.'
                          )
        x,y,yerr = sm.binning_spectrum(trans_list[ii],bins)
        axs[ii].errorbar(x,
                          y,
                          yerr,
                          color =color_pallete_dark[0],
                          label = '_nolegend_',
                          zorder = 501,
                          fmt='.',
                          markersize='20'
                          )
    axs[0].set_title('Transmission spectrum')
    return fig,axs
#%% plot_colormap_masking_regions
def plot_colormap_masking_regions(spec_list_nomask,spec_list_mask,vmin=0,vmax=2,cmap='inferno'):
    '''
    Plot of the stellar mask
    '''
    num_nights = spec_list_nomask[-1].meta['Night_num']
    
    fig, axs = plt.subplots(2,num_nights,sharex=True)
    
    for ni in range(num_nights):
        sub_nomask = sm.get_sublist(spec_list_nomask,'Night_num',ni+1)
        sub_mask = sm.get_sublist(spec_list_mask,'Night_num',ni+1)
        
        x_nomask = sub_nomask[0].spectral_axis.value
        x_mask = sub_mask[0].spectral_axis.value
        
        y_nomask = np.zeros(len(sub_nomask))
        for ind in range(len(sub_nomask)):
            y_nomask[ind]= sub_nomask[ind].meta['Phase']
        y_mask = y_nomask
        
        z_nomask = np.zeros((len(sub_nomask),len(x_nomask)))
        z_mask = np.zeros((len(sub_nomask),len(x_nomask)))
        
        for ind in range(len(sub_nomask)):
            z_nomask[ind,:] = sub_nomask[ind].flux.value
            z_mask[ind,:] = sub_mask[ind].flux.value
        
        axs[0,ni].pcolormesh(x_nomask,y_nomask,z_nomask,vmin=vmin,vmax=vmax,cmap=cmap)
        axs[1,ni].pcolormesh(x_mask,y_mask,z_mask,vmin=vmin,vmax=vmax,cmap=cmap)
    # plt.colorbar(cmap)
    return fig,axs
#%% plot_master_list
def plot_master_list(master_list):
    num_spec = len(master_list)
    if num_spec == 2:
        num_spec = 1
    fig,axs = plt.subplots(num_spec,sharex=True,sharey=True)
    
    if num_spec == 1:
        axs = [axs]
    
    
    for ii in range(num_spec):
        axs[ii].errorbar(master_list[ii].spectral_axis.value,
                         master_list[ii].flux.value,
                         master_list[ii].uncertainty.array,
                         label = master_list[ii].meta['type']
                         )
        axs[ii].set_title('Master:' +
                         master_list[ii].meta['type']+
                         '; Night:'+
                         master_list[ii].meta['night'] +
                         '; #night:' +
                         master_list[ii].meta['Night_num'] +
                         '; RF:' +
                         master_list[ii].meta['RF'])
    return fig,axs

#%% plot_obs_log
def plot_obs_log(obs_log,sys_para,orders):
    '''
    Plots observation log plots given the corresponding class and system parameters
    
    Input:
        obs_log ; observations_log - log of observations 
        sys_para ; system_parameters_class - system parameters (for contact points T14 and T23)
        orders ; list - list of order numbers in the order of obs_log.night[i].SN_array[order]
    Output:
        fig,ax = plt.subplots(num_nights,2)
    '''
    num_nights = len(obs_log.night)
    

    T1 = 0 - sys_para.transit.T14 /sys_para.planet.P/2
    T2 = 0 - sys_para.transit.T23 /sys_para.planet.P/2
    T3 = 0 + sys_para.transit.T23 /sys_para.planet.P/2
    T4 = 0 + sys_para.transit.T14 /sys_para.planet.P/2
    
    fig_list = []
    for ni in range(num_nights):
        fig,axs = plt.subplots(2,1,sharex=True,tight_layout=True)
        axs[0].scatter(obs_log.night[ni].Phase,obs_log.night[ni].Airmass_array, label = 'Airmass')
        axs[0].scatter(obs_log.night[ni].Phase,obs_log.night[ni].Seeing_array, label = 'Seeing')
        for ind in range(len(obs_log.night[ni].SN)):
            print(len(obs_log.night[ni].Phase),len(obs_log.night[ni].SN_array[ind]))
            if len(orders) != 0: 
                axs[1].scatter(obs_log.night[ni].Phase,obs_log.night[ni].SN_array[ind],label = 'Order: '+ str(orders[ind]))
            else:
                axs[1].scatter(obs_log.night[ni].Phase,obs_log.night[ni].SN_array[ind],label = 'Median SNR of all orders')
        axs[0].axvline(T1, label = '_nolegend_')
        axs[0].axvline(T2, label = '_nolegend_')
        axs[0].axvline(T3, label = '_nolegend_')
        axs[0].axvline(T4, label = '_nolegend_')
        axs[1].axvline(T1, label = '_nolegend_')
        axs[1].axvline(T2, label = '_nolegend_')
        axs[1].axvline(T3, label = '_nolegend_')
        axs[1].axvline(T4, label = '_nolegend_')
        
        axs[0].axvspan(-0.5,T1,color = 'red',alpha=0.2)
        axs[0].axvspan(T1,T2,color = 'yellow',alpha=0.2)
        axs[0].axvspan(T2,T3,color = 'green',alpha=0.2)
        axs[0].axvspan(T3,T4,color = 'yellow',alpha=0.2)
        axs[0].axvspan(T4,0.5,color = 'red',alpha=0.2)
        
        axs[1].axvspan(-0.5,T1,color = 'red',alpha=0.2)
        axs[1].axvspan(T1,T2,color = 'yellow',alpha=0.2)
        axs[1].axvspan(T2,T3,color = 'green',alpha=0.2)
        axs[1].axvspan(T3,T4,color = 'yellow',alpha=0.2)
        axs[1].axvspan(T4,0.5,color = 'red',alpha=0.2)
        

        axs[0].set_title('Night:'+str(ni+1)+' Airmass and Seeing')
        axs[1].set_title('Night:'+str(ni+1)+' SNR')

        # axs[0].legend(loc='upper left')
        # axs[1].legend(loc='upper left')
        axs[0].set_xlabel('Phase [1]')
        axs[1].set_xlabel('Phase [1]')
        axs[0].set_xlim(-0.07,0.07)
        axs[0].set_ylim(0,3)
        axs[1].set_ylim(0,50)
        axs[0].set_yticks([0,0.5,1,1.5,2,2.5,3])
        # axs[1].set_xticks([])
        
        fig_list.append([fig,axs])
    return fig_list
#%% extract_colormap_value
def extract_colormap_value(spec_list):
    '''
    Extracts x,y,z values from spec_list for colormap. 
    
    Input:
        spec_list ; sp.SpectrumList - list of spectra from which to extract the values
    Output:
        x,y,z ; np.array - arrays of values to put in colormap
    '''
    x = spec_list[0].spectral_axis.value
    
    y = np.zeros(len(spec_list))
    z = np.zeros((len(spec_list),len(x)))
    for ind in range(len(spec_list)):
        y[ind]= spec_list[ind].meta['Phase']
        z[ind,:] = spec_list[ind].flux.value
    return x,y,z
#%% plot_colormap
def plot_colormap(spec_list,vmin,vmax,cmap='inferno', fig=False,ax =False):
    '''
    Colormap plot of spec list. Needs to be sublist for one night only (to avoid Phase binning problem) and already sublisted to relevant values. 
    Input:
        spec_list ; sp.SpectrumList - list of spectra from which to extract the values
        vmin ; value - value of minimum on the colorscale
        vmax ; value - value of maximum on the colorscale
        cmap ; cmap - which colormap to use 
    Output:
        fig,ax ; plt.subplots(1) - plotted figure
    '''
    x,y,z = extract_colormap_value(spec_list)
    
    if fig == False:
        fig,ax = plt.subplots(1)
    c = ax.pcolormesh(x,y,z,vmin=vmin,vmax=vmax,cmap=cmap)
    cbar = plt.colorbar(c)
    if fig == True:
        return
    else:
        return fig,ax
#%% stellar_remnant
def stellar_remnant(RF_star_list,master_out,RF_star_corrected,line_list,fwhm,sys_para,mask_list):
    '''
    Function for plots based on Julia's HEARTS paper (HEARTS VI.) to show impact of sodium remnant
    Written in way that is not dependent on line masked. For paper necessary to zoom on the feature (as this is dependend on the location of the line)
    
    Input:
        RF_star_list ; sp.SpectrumList - list of spectra from observations before correction of master-out
        master_out ; sp.SpectrumList - master-out list
        RF_star_corrected ; sp.SpectrumList - list of spectra after correcting for master-out in the RF_star
        line_list ; list - list of lines which are still present, requires to be quantity
        fwhm ; value - value to define the core and wings of the planetary signal, reguires to be quantity
        sys_para ; rats.parameters.sys_para class - system parameters to get the contact points
        mask_list ; list - list of lines [l1,r1], left and right edge of mask
    Output:
        fig_night ; list - list of plots for each night
    '''
    num_nights = RF_star_list[-1].meta['Night_num']
    
    color_core,color_wings = 'darkgreen','red'
    color_contact = 'red'
    color_mask = 'white'
    cmap = 'viridis'
    cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    fig_night = []
    for ni in range(num_nights):
        fig = plt.figure(constrained_layout= False)
        spec_grid = gs.GridSpec(ncols=10,nrows=10)
        ax1 = fig.add_subplot(spec_grid[0:4,1:9])
        ax2 = fig.add_subplot(spec_grid[4:6,1:9],sharex=ax1)
        ax3 = fig.add_subplot(spec_grid[6:-1,1:9],sharex=ax1)
        ax2.set_facecolor('black')
        tmp_axs = [ax1,ax2,ax3]
        
        sub_star = sm.get_sublist(RF_star_list,'Night_num',ni+1)
        sub_plan = sm.get_sublist(RF_star_corrected,'Night_num',ni+1)
        
        vmin = 0.8
        vmax = 1.2
        plot_colormap(sub_star,vmin,vmax,cmap, fig=True,ax =tmp_axs[0])
        tmp_axs[0].set_ylabel('Phase [1]',rotation='horizontal')
        tmp_axs[0].set_title('Night #%i '%(ni+1)+'Night:' + sub_star[0].meta['Night'])
        tmp_axs[0].axhline(-(sys_para.transit.T14/2/sys_para.planet.P),ls='--',color=color_contact)
        tmp_axs[0].axhline(sys_para.transit.T14/2/sys_para.planet.P,ls='--',color=color_contact)
        tmp_axs[0].axhline(-(sys_para.transit.T23/2/sys_para.planet.P),ls='--',color=color_contact)
        tmp_axs[0].axhline(sys_para.transit.T23/2/sys_para.planet.P,ls='--',color=color_contact)
        
        tmp_axs[1].plot(master_out[ni].spectral_axis.value,
                        master_out[ni+1].flux.value,
                        color='white',
                        label='master_out')
        tmp_axs[1].set_ylabel('Flux [1]',rotation='horizontal',labelpad=70)
        for line in line_list:
            tmp_axs[1].axvline((line+ fwhm/2).value,color=color_core,label='_nolegend_')
            tmp_axs[1].axvline((line- fwhm/2).value,color=color_core,label='_nolegend_')
        for mask in mask_list:
            tmp_axs[1].axvline(mask[0].value,ls='dashdot',color= color_mask)
            tmp_axs[1].axvline(mask[1].value,ls='dashdot',color= color_mask)
        
        plot_colormap(sub_plan,vmin,vmax,cmap, fig=True,ax =tmp_axs[2])
        
        tmp_axs[2].set_xlabel('Wavelength [%s]'%(sub_plan[0].spectral_axis.unit.to_string('latex_inline')  ))
        tmp_axs[2].set_ylabel('Phase [1]',rotation='horizontal')
        
        vel = np.zeros(len(sub_plan))
        phase = np.zeros(len(sub_plan))
        for ind in range(len(sub_plan)):
            vel[ind] = sub_plan[ind].meta['vel_pl'].to(u.m/u.s).value
            phase[ind] = sub_plan[ind].meta['Phase'].value
        for line in line_list:
            x_vel = (line.value + (line.value * vel[:]/ con.c.value))
            y_vel = phase
            tmp_axs[2].axhline(-(sys_para.transit.T14/2/sys_para.planet.P),ls='--',color=color_contact)
            tmp_axs[2].axhline(sys_para.transit.T14/2/sys_para.planet.P,ls='--',color=color_contact)
            tmp_axs[2].axhline(-(sys_para.transit.T23/2/sys_para.planet.P),ls='--',color=color_contact)
            tmp_axs[2].axhline(sys_para.transit.T23/2/sys_para.planet.P,ls='--',color=color_contact)
            
            tmp_axs[2].plot(x_vel,y_vel,color='black',ls='--',label='_nolegend_')
            tmp_axs[2].plot(x_vel+fwhm.value/2,y_vel,color=color_core,label='_nolegend_')
            tmp_axs[2].plot(x_vel-fwhm.value/2,y_vel,color=color_core,label='_nolegend_')
            
        fig_night.append([fig,tmp_axs])
    return fig_night

#%% ax_percentage
def ax_percentage(ax):
    '''
    Sets ax to percentage units without data recalculation
    '''
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
#%% uncertainty_region
def uncertainty_region(lines,diff_unc,ax,color='orange'):
    '''
    Plots uncertainty regions for sigma estimation
    '''
    for line in lines:
        ax.axvspan((line-diff_unc).value,
                   (line+diff_unc).value,
                   color='orange',
                   label='_nolegend_',
                   alpha=0.2,zorder=1)
        
        
#%% plot_xsigma_nondetection
def plot_xsigma_nondetection(ax,sys_para,unc_list,xloc,sigma=3,color_line = 'black'):
    wl_one =(sys_para.planet.radius**2 / sys_para.star.radius**2).decompose()
    sig = (sys_para.planet.radius**2 / sys_para.star.radius**2).decompose()+(np.mean(unc_list)*sigma)
    ax.axhline((sys_para.planet.radius**2 / sys_para.star.radius**2).decompose()+(np.mean(unc_list)*sigma),ls='--',color=color_line,zorder=999)
    ax.text(xloc,wl_one+(np.mean(unc_list)*sigma),
            '$%i\sigma$ upper limit:\n'%(sigma) +\
            '%.2f'%((sig-wl_one)*100)+ ' [%]'+\
            ' %.2f [$R_p]$'%(sm.RpRs2Rp(sig, sys_para=sys_para)), fontsize=30, va='center', ha='center', backgroundcolor='w',zorder=1003)
    return

#%% save_pdf_paper
@disable_func
def save_pdf_paper(main_dir,name_fig,fig):
    '''
    Save figure into /maindir/Paper/figures directory
    '''
    fig.savefig(main_dir+'/Paper/figures/'+name_fig)
    
#%% plot_molecfit_correction
def plot_molecfit_correction(f):
    '''
    Plots telluric correction of the fits.HDUList file as provided by "run_molecfit_all" module

    Parameters
    ----------
    f : fits.HDUList
        HDU list as provided by "run_molecfit_all". Should have first (with zeroth header) HDU of telluric corrected data, second HDU with telluric correction and third with original spectrum

    Returns
    -------
    fig : plt.figure
        Main figure.
    axs : plt.axes
        Axes with two elements, ax = primary x-axis, ax2 = secondary x-axis .

    '''
    fig,ax = plt.subplots(1)
    ax2 = ax.twinx()
    ax2.set_navigate(False)


    ax.plot(f[3].data['wavelength_air'],f[3].data['flux'],color='blue',alpha=0.5) # Original spectrum
    ax2.plot(f[1].data['wavelength_air'],f[2].data,color='red',alpha =0.3) # Telluric profile
    ax.plot(f[1].data['wavelength_air'],f[1].data['flux'],color='green',alpha=0.5) # Corrected spectrum
    axs = [ax,ax2]
    return fig,axs

#%% calculate_sig_lim
def calculate_sig_lim(spectrum,width):
    '''
    Calculates 1 sigma limits for spectrum given the width (in pixels)

    Parameters
    ----------
    spectrum : sp.Spectrum1D
        Spectrum from which to calculate the 1-sigma array.
    width : int
        Number of pixel of the window size.


    Returns
    -------
    sig_1 : pd.Series
        1-sigma array based on the rolling window.

    '''
    sig_1 = pd.Series(spectrum.uncertainty.array).rolling(width
    ,min_periods=1,center=True).quantile(0.5)
    return sig_1

#%% add_sig_lim
def add_sig_lim(ax,spectrum,sys_para):
    '''
    Adds sigma limits to the transmission spectrum plot. Must be in units of wavelength dependent transit depth

    Parameters
    ----------
    ax : plt.axis
        Artist to which to plot the sigma limits.
    spectrum : sp.Spectrum1D
        Spectrum for which to plot the sigma limits.
    sig_1 : pd.Series
        1-Sigma array as received from calculate_sig_lim function.
    sys_para : para.system_parameters_class
        System parameters to continuum.

    Returns
    -------
    None.

    '''
    sig_1 = calculate_sig_lim(spectrum,width=100)
    
    # 1-sigma region
    ax.fill_between(spectrum.spectral_axis.value,
                         sig_1+sys_para.transit.delta,
                         y2=-sig_1+sys_para.transit.delta,
                         color='red',alpha=0.2,zorder=1)
    
    # # 2-sigma region
    ax.fill_between(spectrum.spectral_axis.value,
                          sig_1*2+sys_para.transit.delta,
                          y2=sig_1*1+sys_para.transit.delta,
                          color='red',alpha=0.15,zorder=1)
    
    ax.fill_between(spectrum.spectral_axis.value,
                          -sig_1*1+sys_para.transit.delta,
                          y2=-sig_1*2+sys_para.transit.delta,
                          color='red',alpha=0.15,zorder=1)
    
    # 3-sigma region
    ax.fill_between(spectrum.spectral_axis.value,
                          sig_1*3+sys_para.transit.delta,
                          y2=sig_1*2+sys_para.transit.delta,
                          color='red',alpha=0.1,zorder=1)
    
    ax.fill_between(spectrum.spectral_axis.value,
                          -sig_1*2+sys_para.transit.delta,
                          y2=-sig_1*3+sys_para.transit.delta,
                          color='red',alpha=0.1,zorder=1)
    
    # 5-sigma region
    ax.fill_between(spectrum.spectral_axis.value,
                          sig_1*5+sys_para.transit.delta,
                          y2=sig_1*3+sys_para.transit.delta,
                          color='orange',alpha=0.2,zorder=1)
    
    ax.fill_between(spectrum.spectral_axis.value,
                          -sig_1*3+sys_para.transit.delta,
                          y2=-sig_1*5+sys_para.transit.delta,
                          color='orange',alpha=0.2,zorder=1)
    
    # Outside of 5-sigma region
    ax.fill_between(spectrum.spectral_axis.value,
                          sig_1*5+sys_para.transit.delta,
                          y2=100+sys_para.transit.delta,
                          color='green',alpha=0.2,zorder=1)
    
    ax.fill_between(spectrum.spectral_axis.value,
                          -sig_1*100+sys_para.transit.delta,
                          y2=-sig_1*5+sys_para.transit.delta,
                          color='green',alpha=0.2,zorder=1)
    return

#%% plot_transmission_spectrum
# =============================================================================
# TODO: Redo plots so they work with a new decorator
# TODO: Documentation once redone
# =============================================================================


# @todo_function
# @save_figure
def plot_transmission_spectrum(transmission_list:list,
                            #    sys_para: rats.parameters.system_parameters_class,
                               line_list:list=[],
                               wave_region:sp.spectra.spectral_region.SpectralRegion = sp.SpectralRegion(5000*u.AA,7000*u.AA),
                               binning_factor:int = 15,
                               unit=u.dimensionless_unscaled,
                               equivalency = None,
                               fig_name = '',
                               ):
    
    figure_list = []
    
    for ni,transmission in enumerate(transmission_list):
        if ni > 0:
            continue
        fig,ax = plt.subplots(1)
        
        # Extract smaller wave regions for faster plotting
        # cut_transmission = transmission
        cut_transmission = sm.extract_region_list([transmission],wave_region)[0]
        
        
        # Plot entire transmission spectrum
        ax.errorbar(cut_transmission.spectral_axis.value,
                    cut_transmission.flux.value,
                    cut_transmission.uncertainty.array,
                    color = 'black',
                    label = '_nolegend_',
                    zorder = 500,
                    fmt='.',
                    markersize=10,
                    elinewidth=1,
                    alpha=0.4
                          )
        
        # Disable binning by having binning factor of 0
        if binning_factor != 0:
            x,y,yerr = sm.binning_spectrum(cut_transmission,binning_factor) 
            ax.errorbar(x,
                        y,
                        yerr,
                        color ='royalblue',
                        ecolor = 'cornflowerblue',
                        label = '_nolegend_',
                        zorder = 501,
                        fmt='.',
                        markersize='20',
                        markeredgecolor='black',
                        markeredgewidth = 1,
                        )
        
        for line in line_list:
            ax.axvline(line,
                       color='darkblue',
                       ls='--',
                       )
            
        # ps.add_sig_lim(ax,spectrum,sys_para)
        # ax.axhline(sys_para.transit.delta,color='white')
        ax_percentage(ax)
        ax.set_ylabel('$R_{p,\lambda}^2/R_s^2$')
        ax.set_xlabel('Wavelength [$\AA%$]',labelpad=0)
        ax.legend(['Night %i'%(ni)])
        ax.set_ylim(.95,1.05)
        
        figure_list.append([fig, [ax]])
    return figure_list
