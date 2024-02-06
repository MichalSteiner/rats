# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:26:21 2021

@author: Chamaeleontis

Plotting functions
"""
#%% Importing libraries
import rats
from typing import List
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import rats.spectra_manipulation as sm
import astropy.constants as con
import astropy.units as u
from matplotlib.lines import Line2D
import matplotlib.patches as  mpatches
import os
from scipy.stats import gaussian_kde
#%% Color_pallete
color_pallete_dark = sns.color_palette("dark")
color_pallete_pastel = sns.color_palette("pastel")
color_pallete_muted = sns.color_palette("muted")
color_pallete_deep = sns.color_palette("deep")
color_pallete_bright = sns.color_palette("bright")
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_pallete_dark) 


def wavelength_to_rgb(wavelength: u.Quantity):
    """
    Convert a wavelength in nanometers to an RGB color.
    """
    
    wavelength = wavelength.to(u.nm)
    if wavelength < 380:
        wavelength = 380
    if wavelength > 750:
        wavelength = 750

    if 380 <= wavelength <= 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength <= 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    else:
        R = 1.0
        G = 0.0
        B = 0.0

    # Adjust intensities for different parts of the spectrum
    if 380 <= wavelength <= 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 645 <= wavelength <= 750:
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
    else:
        factor = 1.0

    R *= factor
    G *= factor
    B *= factor

    return (R, G, B)


#%% alias_nasa_table_comp
def alias_nasa_table_comp(key):
    '''
    Takes key from NASA exoplanet archive - PS composite table, and returns its full name and unit.
    Input:
        key
    Output:
        name,unit
    '''
    if key == 'pl_insol': #Insolation flux
        return 'Insolation flux', '$S_{\oplus}$'
    elif key == 'pl_bmasse': # Mass in Earth masses
        return 'Planetary mass', '$M_{\oplus}$'
    elif key == 'pl_rade': # Radius in Earth radii
        return 'Planetary radius', '$R_{\oplus}$'
    elif key == 'pl_orbper': # Radius in Earth radii
        return 'Period', '$d$'
    elif key == 'pl_dens': # Radius in Earth radii
        return 'Planetary density', '$gcm^{-3}$'
    elif key == 'pl_eqt': # Radius in Earth radii
        return 'Equilibrium temperature', '$K$'
#%% Convert_Earth_to_Jupiter
def rad_earth2jupiter(x): # Convert radius of Earth to Jupiter
    return x * con.R_earth.to(u.R_jup).value

def rad_jupiter2earth(x): # Convert radius Jupiter to Earth
    return x * con.R_jup.to(u.R_earth).value

def mass_earth2jupiter(x): # Convert mass Earth to Jupiter
    return x * con.M_earth.to(u.M_jup).value

def mass_jupiter2earth(x): # Convert mass Jupiter to Earth
    return x * con.M_jup.to(u.M_earth).value
#%% tie_y_axis
def tie_y_axis(ax,ykey):
    if ykey == 'pl_bmasse':
        ax2 = ax.secondary_yaxis('right', functions=(mass_earth2jupiter, mass_jupiter2earth))
        ax2.set_ylabel('Planetary mass $[M_{Jup}]$', fontsize = 36)
    elif ykey == 'pl_rade':
        ax2 = ax.secondary_yaxis('right', functions=(rad_earth2jupiter, rad_jupiter2earth))
        ax2.set_ylabel('Planetary radius $[R_{Jup}]$', fontsize =36 )
    return ax2
    
#%% plot_mass_insolation
def plot_mass_insolation(exoplanet_class,
                         cmap = 'viridis',
                         xkey = 'pl_insol',
                         ykey = 'pl_bmasse',
                         ):
    '''
    Plots figure of all known exoplanets in mass [M_earth] vs Insolation [S_earth] units if sufficient parameters are found
    Input:
        exoplanet_class ; rats.para.observed_exoplannesa
        cmap = colormap ; which colormap to use
        xkey = string ; keyword of nasa table to use for x-axis
        ykey = string ; keyword of nasa table to use for y-axis
    Output:
        fig,ax = plt.subplots(1)
    '''
    # Initiate 
    fig,ax = plt.subplots(1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if ykey == 'pl_bmasse' or ykey=='pl_rade':
        ax2 = tie_y_axis(ax,ykey)
    if xkey == 'pl_insol':
        plt.gca().invert_xaxis()
    if ykey == 'pl_dens':
        
        plt.gca().invert_yaxis()
    
    xname,xunit = alias_nasa_table_comp(xkey)
    yname,yunit = alias_nasa_table_comp(ykey)

    x = exoplanet_class.nasa_table[xkey].values
    y = exoplanet_class.nasa_table[ykey].values
    ind = np.logical_and(np.isfinite(x),np.isfinite(y))
    
    
    x = x[ind] + 0.000001
    y = y[ind]
    
    sns.kdeplot(x=x,
                y=y,
                cmap=cmap,
                fill=True,
                label='_nolegend_',
                bw_adjust=0.6,
                log_scale =[True, True],
                levels = [0,.0001,.001,.01,.1,.3,.5,.8,1],
                ax=ax
                )

    ax.scatter(
            exoplanet_class.nasa_table[xkey],
            exoplanet_class.nasa_table[ykey],
            color = 'black',
            edgecolors = 'black',
            label='_nolegend_',
            alpha = 0.5,
            s = 20,
        )
    ax.set_title(yname +' vs ' + xname +' for known exoplanets')

    ax.set_xlabel(xname + '[' + xunit+ ']', fontsize = 36)
    ax.set_ylabel(yname + ' [' + yunit+']', fontsize = 36)
    return fig,ax
#%% highlight_planet_group
def highlight_planet_group(
        exoplanet_class,
        planet_list,
        color,
        ax,
        xkey = 'pl_insol',
        ykey = 'pl_bmasse',
        label = '',
        marker = 'o',
        s = 240
        ):
    '''
    Highlights a planet(s) in mass/radius vs insolation plot by larger points with set color
    Input:
        exoplanet_class ; rats.para.observed_exoplannesa
        planet_list - list ; names of plannesa to highlight
        xkey = string ; keyword of nasa table to use for x-axis
        ykey = string ; keyword of nasa table to use for y-axis
    Output:
        None
    '''
    
    for ii,planet in enumerate(planet_list):
        data = exoplanet_class.nasa_table[exoplanet_class.nasa_table['pl_name']==planet]
        ax.scatter(
                data[xkey],
                data[ykey],
                color = color,
                marker = marker,
                edgecolor = 'black',
                # label='_nolegend_',
                s = s
                )
    handle = Line2D([0], [0], marker=marker, color='w', label=label,
                          markerfacecolor=color,markersize=s/15)
    # handle = mpatches.Patch(facecolor=color,edgecolor='black', label=label)
    return handle
# #%%
# def radius_insolation_plot(exo_class:rats.parameters.observed_exoplanets,
#                            lists_highlighted_planets:List = [[]],
#                            lists_highlighted_labels:List = [[]],
#                            lists_highlighted_colors:List = sns.color_palette("bright"),
#                            xkey:str = 'pl_insol',
#                            ykey:str = 'pl_rade',
#                            cmap:mpl.colors.LinearSegmentedColormap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
#                            ) -> tuple[mpl.figure.Figure,
#                                 plt.Axes]:
#     '''
#     Create a radius vs insolation flux plot of all exoplanets. Depending on the lists variable input, it highlights selected planet(-s) in the diagram with given color.

#     Parameters
#     ----------
#     exo_class : rats.parameters.observed_exoplanets
#         Class containing all the observed exoplanets to date.
#     lists_highlighted_planets : List, optional
#         Lists of lists of planets to highlight in the diagram. The default is [[]].
#     lists_highlighted_labels : List, optional
#         Lists of lists of labels to highlight in the diagram. The default is [[]].
#     lists_highlighted_colors : List, optional
#         Lists of lists of colors to highlight in the diagram. The default is sns.color_palette("bright").
#     xkey : str, optional
#         Key to use for NASA archive on x-axis. The default is 'pl_insol'.
#     ykey : str, optional
#         Key to use for NASA archive on y-axis. The default is 'pl_rade'.
#     cmap : mpl.colors.LinearSegmentedColormap, optional
#         Colormap to use for the KDE plot. The default is sns.dark_palette("#69d", reverse=False, as_cmap=True).

#     Returns
#     -------
#     fig : mpl.figure.Figure
#         Figure container with the plot.
#     ax : plt.Axes
#         Axes container with the given plot.

#     '''
#     # cmap = sns.dark_palette("#69d", reverse=False, as_cmap=True)
#     # cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
#     fig,ax = plot_mass_insolation(exo_class,cmap,xkey,ykey)
    
#     list_handles = []
#     for planet, label, color in zip(lists_highlighted_planets,
#                                     lists_highlighted_labels,
#                                     lists_highlighted_colors):
#         handle = highlight_planet_group(exo_class,
#                                            planet,
#                                            color, 
#                                            ax,
#                                            xkey,
#                                            ykey,
#                                            label,
#                                            marker='o',
#                                            s=500
#                                            )
#         list_handles.append(handle)
    
#     plt.legend(handles=
#         list_handles
#         )
#     ax.set_xlim(10E4,10E-1)
#     # ax.set_ylim(10E-1,25E0)
#     ax.get_figure().gca().set_title("")

#     return fig,ax

#%% draw_lightcurves
def draw_lightcurves(directory,sys_para,offset= 2450000,flux_step=0.02,phase_fraction=20):
    '''
    Draws all lightcurves in selected directory
    
    TODO: 
        Change the input to CONAN output, so the data are already detrended
        Initialize batman model for every light curve with their limb-darkening coeeficients
    '''
    fig,ax = plt.subplots(1)
    sys_para.define_light_curve_model(c1 = 1.04146619,
                                      c2 = 0.18204625)
    model_time  = np.linspace(
        sys_para.transit.T_C.value-sys_para.system.P.value/phase_fraction,
        sys_para.transit.T_C.value+sys_para.system.P.value/phase_fraction,
        num=1000)*u.day
    model_phase = np.zeros(np.shape(model_time))
    model_flux= np.zeros(np.shape(model_time))
    for ind,value in enumerate(model_time):
        model_phase[ind] = sm.phase_fold(value,sys_para)
        model_flux[ind]= sys_para.get_lc_flux(value)*u.dimensionless_unscaled
    for ni,item in enumerate(os.listdir(directory)):
        array = np.loadtxt(directory + '/' + item,skiprows=1)
        time = (array[:,0] + offset )*u.day 
        flux = array[:,1] *u.dimensionless_unscaled
        flux_err= array[:,2]*u.dimensionless_unscaled
        phase = np.zeros(np.shape(time))
        for ind,value in enumerate(time):
            phase[ind] = sm.phase_fold(value,sys_para)
        ax.errorbar(phase,flux-ni*flux_step,yerr=flux_err,linestyle='none',label=item)
        ax.plot(model_phase,model_flux - ni*flux_step,label='_nolegend_')
    ax.legend()
    return
#%% add_lines_na
def add_lines_na(ax,color='darkgreen',ls='--',label='_nolegend_',zorder=999):
    ax.axvline(5889.950,color=color,ls=ls, label= label,zorder= zorder)
    ax.axvline(5895.924,color=color,ls=ls, label= label,zorder= zorder)
