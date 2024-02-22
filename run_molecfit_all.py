#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:10:25 2022

@author: chamaeleontis

To use this code:
    0. Setup esorex command (needs to be full command, alias won't work)
    1. Setup directory root which should consist of these folders:
        molecfit_input
            input spectra
        molecfit_output
            empty, will contain results
        molecfit_reduced
            empty, will containt input spectra after reduction
        tmp_files
            empty or WAVE_INCLUDE, WAVE_EXCLUDE a
        para_files
            molecfit_model.rc
            molecfit_calctrans.rc
            molecfit_correct.rc
            SOF files will be created automatically
            Make sure the RC parameters are valid for the dataset
    2. Select mode of correction
        force_reference = False
            for each spectrum the molecfit_model and molecfit_calctrans commands will be run
                Works well
            if force_s2d = True then molecfit_correct is run on S2D files
                Not implemented, maybe in the future if necessary
        force_reference = True
            Run the molecfit_model and molecfit_calctrans commands on one reference spectrum (needs to be specified in filename)
            Use this telluric profile to correct ALL spectra in directory. It is not advised to use this mode (also, it might break, not tested enough)

Main issues:
    Molecfit as of right now doesn't provide separate telluric profiles for each molecules (it is implemented in esoreflex, but couldn't find easy version of it in esorex). This means that if wavelength regions don't include lines from fitted molecule, it won't be corrected. For ESPRESSO, the default wavelength range is including both H2O and O2 bands. For other instruments, check Smette et al. 2015, Fig. 1 for telluric lines in spectral region.
    
    
By default, the single_use setup will provide the molecfit folders with correct instrument setup for HARPS, ESPRESSO and NIRPS. The wavelength bands provided are mostly correct, but need checking for possibly stellar contamination inside.

TODO:
    Test that the code works using the run_molecfit_all function.
    Implement wavelength frame
    If necessary, consider multiprocessing on each call of molecfit. This need to disable interactive mode completely. Furthermore, the usage of tmp_directories needs to be implemented.
    
"""

#%% Import libraries
import numpy as np
import astropy.io.fits as fits
import subprocess
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gs
from rats.utilities import default_logger_format
import logging
#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

#%%
# Location of esorex command, alias won't work!
command_esorex = "/home/chamaeleontis/esoreflex/esoreflex/bin/esorex"
force_sgl = False
force_all = False
wlg2mic = 0.0001 # Conversion factor from our units to microns
restframe = 'AIR' # Which wavelength frame are we in, air or vacuum
#%% Settings
if __name__ == '__main__':
    # =============================================================================
    # TODO:
    # Settings to change by default for each run:
        # For ESPRESSO data, its necessary to define the UT in rcparams file
    # =============================================================================
    # Main directory
    # directory_root = '/media/chamaeleontis/Observatory_main/WASP-76/data/spectra/ESPRESSO/2018-09-02/Fiber_A/S1D/molecfit'
    # =============================================================================
    # HEARTS
    # =============================================================================

    # =============================================================================
    # KELT-11b
    # =============================================================================
    directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/KELT-11b/data/spectra/HARPS/2017-02-01/Fiber_A/S1D/molecfit'

    directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/KELT-11b/data/spectra/HARPS/2017-02-14/Fiber_A/S1D/molecfit'
    directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/KELT-11b/data/spectra/HARPS/2017-02-15/Fiber_A/S1D/molecfit'
    directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/KELT-11b/data/spectra/HARPS/2017-02-16/Fiber_A/S1D/molecfit'

    directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/KELT-11b/data/spectra/HARPS/2017-03-05/Fiber_A/S1D/molecfit'
    directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/KELT-11b/data/spectra/HARPS/2017-03-06/Fiber_A/S1D/molecfit'
    directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/KELT-11b/data/spectra/HARPS/2017-03-07/Fiber_A/S1D/molecfit'


    # =============================================================================
    # WASP-121b
    # =============================================================================
    directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/WASP-121b/data/spectra/HARPS/2017-12-31/Fiber_A/S1D/molecfit'
    directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/WASP-121b/data/spectra/HARPS/2018-01-09/Fiber_A/S1D/molecfit'
    # directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/WASP-121b/data/spectra/HARPS/2018-01-14/Fiber_A/S1D/molecfit'

    # =============================================================================
    # WASP-166b
    # =============================================================================
    # directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/WASP-166b/data/spectra/HARPS/2017-01-13/Fiber_A/S1D/molecfit'
    # directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/WASP-166b/data/spectra/HARPS/2017-03-03/Fiber_A/S1D/molecfit'
    # directory_root = '/media/chamaeleontis/Observatory_main/Analysis_dataset/APL_HEARTS_Loukas/WASP-166b/data/spectra/HARPS/2017-03-14/Fiber_A/S1D/molecfit'


    # =============================================================================
    # Valentina SKYSUB data
    # =============================================================================
    # directory_root = '/media/chamaeleontis/Observatory_main/Valentina_molecfit/WASP-189b/2023-04-24/molecfit_HARPS_SKYSUB/molecfit'
    # directory_root = '/media/chamaeleontis/Observatory_main/Valentina_molecfit/WASP-189b/2023-06-04/molecfit_HARPS_SKYSUB/molecfit'
    # WLG_TO_MICRON parameterL
    # =============================================================================
    # Additional directories
    directory_molecfit_input = directory_root +'/molecfit_input'
    directory_molecfit_output = directory_root +'/molecfit_output'
    directory_molecfit_reduced = directory_root +'/molecfit_reduced'
    directory_molecfit_tmp_files = directory_root  +'/tmp_files'
    directory_molecfit_para_files = directory_root  +'/para_files'
    directory_molecfit_input_s2d = directory_root +'/molecfit_input_S2D'
    # regions fits files
    wave_include = directory_molecfit_tmp_files + '/WAVE_INCLUDE.fits'
    wave_exclude = directory_molecfit_tmp_files + '/WAVE_EXCLUDE.fits'
    pixel_exclude = directory_molecfit_tmp_files + '/PIXEL_EXCLUDE.fits'
    # rc params files location
    rc_para_model = directory_molecfit_para_files + '/molecfit_model.rc'
    rc_para_calctrans = directory_molecfit_para_files + '/molecfit_calctrans.rc'
    rc_para_correct = directory_molecfit_para_files + '/molecfit_correct.rc'
    # SOF files location
    sof_para_model = directory_molecfit_para_files + '/molecfit_model.sof'
    sof_para_calctrans = directory_molecfit_para_files + '/molecfit_calctrans.sof'
    sof_para_correct = directory_molecfit_para_files + '/molecfit_correct.sof'
    # Wave region fits files
    wave_include_fits = directory_molecfit_tmp_files + '/WAVE_INCLUDE.fits'
    wave_exclude_fits = directory_molecfit_tmp_files + '/WAVE_EXCLUDE.fits'
    pixel_exclude_fits = directory_molecfit_tmp_files + '/PIXEL_EXCLUDE.fits'
    # molecfit commands 
    command_molecfit_model = [command_esorex,'--recipe-config=%s'%rc_para_model,'molecfit_model',sof_para_model]
    command_molecfit_calctrans = [command_esorex,'--recipe-config=%s'%rc_para_calctrans,'molecfit_calctrans',sof_para_calctrans]
    command_molecfit_correct = [command_esorex,'--recipe-config=%s'%rc_para_correct,'molecfit_correct',sof_para_correct]
    # Switch to directory of temporary files to run molecfit_model there
    os.chdir(directory_molecfit_tmp_files)
#%% Plot settings
    plt.rcParams["figure.figsize"] = 18.52, 11.24

#%% Running commands
def run_molecfit_model(): # Run molecfit_model
    subprocess.run(command_molecfit_model)
    return
def run_molecfit_calctrans(): # Run molecfit calctrans
    subprocess.run(command_molecfit_calctrans)
    return
def run_molecfit_correct(): # Run molecfit_correct
    subprocess.run(command_molecfit_correct)
    return
#%% 
class Button_region:
    '''
    Class for buttons regarding region selections
    '''

    def replot_region(self):
        '''
        Replot the regions in the plot. Should not be necessary

        Returns
        -------
        None.

        '''
        self.vspans = []
        for row in self.fits_table[1].data:
            self.vspans.append(self.main_ax.axvspan(row['LOWER_LIMIT'],row['UPPER_LIMIT'],color=self.color_vspan,alpha=0.5))
        return
    
    def save_region(self):
        '''
        Saves wavelength region to the respective fits file

        Returns
        -------
        None.

        '''
        if self.r1 < self.l1:
            print('Invalid click')
            del(self.l1,self.r1)
            return
        new_table = fits.BinTableHDU.from_columns(self.fits_table[1].columns,
                                                  nrows=self.fits_table[1].header['NAXIS2']+1
                                                  )
        new_table.data[-1]['LOWER_LIMIT'] = self.l1
        new_table.data[-1]['UPPER_LIMIT'] = self.r1
        self.fits_table[1].data = new_table.data 
        
        self.fits_table.writeto(self.fits_file,overwrite=True)
        del(self.l1, self.r1) # Delete the boundaries to avoid buggy behavior
        
        for vspan in self.vspans:
            vspan.remove()
        self.replot_region()
        return
    
    def delete_region(self):
        '''
        Deletes the region from the respective fits file

        Returns
        -------
        onclick : function
            Event for right click.

        '''
        def onclick(event):
            if event.inaxes.get_label() == 'mainplot' and event.button == 3: # Only the mainplot right-button clicks are valid
                ind_array = np.arange(self.fits_table[1].header['NAXIS2'])
                for ind,row in enumerate(self.fits_table[1].data):
                    if row['LOWER_LIMIT']<event.xdata and row['UPPER_LIMIT']>event.xdata:
                        ind_array = ind_array[ind_array[:]!=ind]
                self.fits_table[1].data = self.fits_table[1].data[ind_array]
                self.fits_table.writeto(self.fits_file,overwrite=True)
                self.fits_table = fits.open(self.fits_file)
                for vspan in self.vspans:
                    vspan.remove()
                self.replot_region()
        return onclick
    
    def press_click(self):
        '''
        Defines function to do on click

        Returns
        -------
        onclick : function
            What to do on click.

        '''
        def onclick(event):
            if event.inaxes.get_label() == 'mainplot' and event.button == 1: # Only the mainplot left-button clicks are valid
                self.l1 = event.xdata
        return onclick
    
    def release_click(self):
        '''
        Extract x-position on release of button, and saves it to respective wavelength region

        Returns
        -------
        onclick : function
            Extracts x-position of release and saves it.

        '''
        def onclick(event):
            if event.inaxes.get_label() == 'mainplot'  and event.button == 1: # Only the mainplot left-button clicks are valid
                self.r1 = event.xdata
                self.save_region()
        return onclick
    
    
    def region_click(self):
        '''
        Defines activation of events depending on the button state

        Returns
        -------
        clicked : function
            If turned on, activates several mouse click events.

        '''
        def clicked(event):
            if not(self.active): # Activate
                self.button.color = self.color_on
                self.active = not(self.active)
                self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.press_click())
                self.cid2 = self.fig.canvas.mpl_connect('button_release_event', self.release_click())
                self.cid3 = self.fig.canvas.mpl_connect('button_press_event', self.delete_region())
            else: # Disable
                self.button.color = self.color_off
                self.active = not(self.active)
                try:
                    self.fig.canvas.mpl_disconnect(self.cid1)
                    self.fig.canvas.mpl_disconnect(self.cid2)
                    self.fig.canvas.mpl_disconnect(self.cid3)
                except:
                    pass
                
        return clicked
    
    
    def __init__(self,
                 button,
                 fig,
                 ax,
                 fits_file,
                 color_off,
                 color_on,
                 active,
                 main_ax,
                 color_vspan,
                 ):
        '''
        Main initialization function

        Parameters
        ----------
        button : button
            Button connected to the class.
        fig : fig
            Main figure.
        ax : ax
            ax where the button is located on the figure.
        fits_file : str
            Which filename is asociated with this button.
        color_off : str
            Color of turned off button.
        color_on : str
            Color of turned on button
        active : bool
            Is the button active.
        main_ax : ax
            Main ax where data are plotted.
        color_vspan : str
            Color of the vspan of the given button.

        Returns
        -------
        None.

        '''
        self.button = button
        self.fig = fig
        self.ax = ax
        self.fits_file = fits_file
        self.fits_table = fits.open(self.fits_file)
        self.color_off = color_off
        self.color_on = color_on
        self.active = active
        self.main_ax = main_ax
        self.color_vspan = color_vspan
        self.button.on_clicked(self.region_click())
        self.replot_region()
        return
#%%
class Button_flag_sgl:
    '''
    Button for force_flag_sgl
    '''
    def onclick(self):
        '''
        Turns on and off the flag

        Returns
        -------
        clicked : event
            Turns on/off the flag on click on the button.

        '''
        def clicked(event):
            global force_sgl
            self.flag = not(self.flag)
            force_sgl = not(force_sgl)
            if force_sgl:
                self.button.color = self.color_on
                self.button.label.set_text('Force\nsingle\nflag\nON')
            else:
                self.button.color = self.color_off
                self.button.label.set_text('Force\nsingle\nflag\nOFF')
            print(self.flag,force_sgl)
        return clicked
    
    def __init__(self,
                 button,
                 fig,
                 ax,
                 color_off,
                 color_on,
                 flag,
                 ):
        '''
        Initialize the class

        Parameters
        ----------
        button : Button
            Button associated with the flag.
        fig : fig
            Main figure.
        ax : ax
            Ax associated with the button.
        color_off : str
            Color of turned off button.
        color_on : str
            Color of turned on button.
        flag : bool
            flag associated with the button.


        Returns
        -------
        None.

        '''
        self.button = button
        self.fig = fig
        self.ax = ax
        self.color_off = color_off
        self.color_on = color_on
        self.flag = flag
        self.button.on_clicked(self.onclick())
        return
#%% 
class Button_flag_all:
    '''
    Button for force_flag_all
    '''
    def onclick(self):
        '''
        Turns on and off the flag

        Returns
        -------
        clicked : event
            Turns on/off the flag on click on the button.

        '''
        def clicked(event):
            global force_all
            self.flag = not(self.flag)
            force_all = not(force_all)
            if force_all:
                self.button.color = self.color_on
                self.button.label.set_text('Force\nall\nflag\nON')
            else:
                self.button.color = self.color_off
                self.button.label.set_text('Force\nall\nflag\nOFF')
        return clicked
    
    def __init__(self,
                 button,
                 fig,
                 ax,
                 color_off,
                 color_on,
                 flag,
                 ):
        '''
        Initialize the class

        Parameters
        ----------
        button : Button
            Button associated with the flag.
        fig : fig
            Main figure.
        ax : ax
            Ax associated with the button.
        color_off : str
            Color of turned off button.
        color_on : str
            Color of turned on button.
        flag : bool
            flag associated with the button.


        Returns
        -------
        None.

        '''
        self.button = button
        self.fig = fig
        self.ax = ax
        self.color_off = color_off
        self.color_on = color_on
        self.flag = flag
        self.button.on_clicked(self.onclick())
        return

class Button_replot: # Doesn't do anything
    def __init__(self):
        return
#%% close_plot
def close_plot(fig):
    '''
    Closing plot on middle mouse button
    '''
    def clicked(event):
        if event.button == 2: # Only when middle button is pressed
            plt.close('all')
    return clicked
#%% Limits
class Limits:
    '''
    Manipulation of the plot with keys
    Press ctrl+c or ctrl+v for shifting to the left/right 
    '''
    def zoom(self):
        def on_xlims_change(event_ax):
            self.ax_lims = event_ax.get_xlim()
        return on_xlims_change
    def __init__(self,
                 ax,
                 ax_lims):
        self.ax_lims = ax_lims
        self.ax = ax
        
    def move_plot(self):
        def onclick(event):
            '''
            Left/Right/Up/Down arrow navigation
            '''
            if event.key == 'ctrl+c':
                
                lims = self.ax_lims
                adjust = (lims[1] - lims[0]) * 0.8
                self.ax.set_xlim((lims[0] - adjust, lims[1] - adjust))
                plt.draw()
            elif event.key == 'ctrl+v':
                lims = self.ax_lims
                adjust = (lims[1] - lims[0]) * 0.8
                self.ax.set_xlim((lims[0] + adjust, lims[1] + adjust))
                plt.draw()
        return onclick
#%%
def show_gui():
    '''
    Main plot for interaction with molecfit_model segment
    Divided into two parts:
        main plot:
            input_spectrum - input spectrum
            telluric_model - model of the telluric lines
        buttons:
            wave_include - Include wavelength check
            wave_exclude - Exclude wavelength check
            pixel_exclude - Exclude pixel check
            replot - Replot the regions
            force_sgl - force sgl button
            force_all - force all spectra button
    
    '''
    # Fits files
    spectrum_model = fits.open(directory_molecfit_tmp_files + '/TELLURIC_DATA.fits')
    # Setup colors
    color_off = 'orangered'
    color_on = 'yellowgreen'
    color_replot = 'darkturquoise'
    color_flags = 'goldenrod'
    color_flags_on = 'olive'
    # Creating plot
    fig = plt.figure()
    grid = gs.GridSpec(6, 6, figure=fig)
    ax1 = fig.add_subplot(grid[0:5, :],label='mainplot') # Main plot
    ax1_second = ax1.twinx()
    ax1_second.set_navigate(False)
    ax1_second.set_label('mainplot')
    # Buttons axis
    ax2 = fig.add_subplot(grid[5, 0:1],label='button_wave_include')
    ax3 = fig.add_subplot(grid[5, 1:2],label = 'button_wave_exclude')
    ax4 = fig.add_subplot(grid[5, 2:3],label = 'button_wave_exclude')
    ax5 = fig.add_subplot(grid[5, 3:4],label = 'button_wave_exclude')
    ax6 = fig.add_subplot(grid[5, 4:5],label = 'button_wave_exclude')
    ax7 = fig.add_subplot(grid[5, 5:],label = 'button_wave_exclude')
    # Plot the first plot
    l1, = ax1.plot(spectrum_model[1].data['lambda'],spectrum_model[1].data['flux'],) # Basic spectrum, saved in TELLURIC_DATA.fits file
    l2, = ax1_second.plot(spectrum_model[1].data['mlambda'],spectrum_model[1].data['mtrans'],color='darkred', alpha= 0.5) # Telluric model used
    
    # Navigate with key arrows
    new_lim = Limits(ax1,ax1.get_xlim())
    ax1.callbacks.connect('xlim_changed', new_lim.zoom())
    fig.canvas.mpl_connect('key_press_event', new_lim.move_plot())
    fig.canvas.mpl_connect('button_press_event', close_plot(fig))
    # Creating buttons
    button_wave_include = Button(ax2, 'Wave \ninclude',color=color_off)
    button_wave_exclude = Button(ax3, 'Wave \nexclude',color=color_off)
    button_pixel_exclude = Button(ax4, 'Pixel \nexclude',color=color_off)
    button_replot = Button(ax5, 'Replot',color=color_replot)
    button_force_flag_sgl = Button(ax6, 'Force\nsingle\nflag\nOFF',color=color_flags)
    button_force_flag_all = Button(ax7, 'Force\nall\nflag\nOFF',color=color_flags)
    
    # Button definition
    global Button_wave_include
    Button_wave_include = Button_region(button_wave_include,
                                        fig,
                                        ax2,
                                        wave_include_fits,
                                        color_off,
                                        color_on,
                                        active = False,
                                        main_ax=ax1,
                                        color_vspan = 'green'
                                        )
    
    Button_wave_exclude = Button_region(button_wave_exclude,
                                        fig,
                                        ax3,
                                        wave_exclude_fits,
                                        color_off,
                                        color_on,
                                        active = False,
                                        main_ax=ax1,
                                        color_vspan = 'red'
                                        )
    
    Button_pixel_exclude = Button_region(button_pixel_exclude,
                                        fig,
                                        ax4,
                                        pixel_exclude_fits,
                                        color_off,
                                        color_on,
                                        active = False,
                                        main_ax=ax1,
                                        color_vspan = 'yellow'
                                        )
    
    Button_single_flag = Button_flag_sgl(button_force_flag_sgl,
                                     fig,
                                     ax6,
                                     color_flags,
                                     color_flags_on,
                                     flag= force_sgl
                                     )
    
    Button_all_flag = Button_flag_all(button_force_flag_all,
                                     fig,
                                     ax6,
                                     color_flags,
                                     color_flags_on,
                                     flag= force_all
                                     )
    plt.ginput(-1,timeout=-1) # Stop running code while interacting with plot
    plt.close('all')
    return

#%% create_region_fits_files
def create_region_fits_files(item):
    ''' 
    Create wavelength regions fits files (WAVE_INCLUDE, WAVE_EXCLUDE, PIXEL_EXCLUDE)
    Will only create files if they do not exist already
    
    Sets the initial wave region to last 1000 pixels in the spectrum. This is optimized for ESPRESSO S1D spectra and similar spectrographs
    '''
    try: # WAVE_INCLUDE part
        fits.open(wave_include_fits)
    except:
        spectrum = fits.open(directory_molecfit_input + '/' + item)
        if spectrum[0].header['INSTRUME'] == 'ESPRESSO':
            lower_lim = fits.Column(name= 'LOWER_LIMIT',format='1D',array = [0.68,0.72])
            upper_lim = fits.Column(name= 'UPPER_LIMIT',format='1D',array = [0.69,0.73])
        elif spectrum[0].header['INSTRUME'] == 'HARPS':
            lower_lim = fits.Column(name= 'LOWER_LIMIT',format='1D',array = [0.62,0.64])
            upper_lim = fits.Column(name= 'UPPER_LIMIT',format='1D',array = [0.64,0.68])
        else:
            lower_lim = fits.Column(name= 'LOWER_LIMIT',format='1D',array = [spectrum[1].data['wavelength'][-2000]*wlg2mic])
            upper_lim = fits.Column(name= 'UPPER_LIMIT',format='1D',array = [spectrum[1].data['wavelength'][-1000]*wlg2mic])
        
        new_table_wave_include = fits.BinTableHDU.from_columns([lower_lim,upper_lim])
        hdu = fits.PrimaryHDU()
        hdul = fits.HDUList([hdu,new_table_wave_include])
        hdul.writeto(wave_include_fits)
        pass
    
    try: # WAVE_EXCLUDE part
        fits.open(wave_exclude_fits)
    except:
        lower_lim = fits.Column(name= 'LOWER_LIMIT',format='1D')
        upper_lim = fits.Column(name= 'UPPER_LIMIT',format='1D')
        new_table_wave_exclude = fits.BinTableHDU.from_columns([lower_lim,upper_lim])
        hdu = fits.PrimaryHDU()
        hdul = fits.HDUList([hdu,new_table_wave_exclude])
        hdul.writeto(wave_exclude_fits)
        pass
    
    try: # PIXEL_EXCLUDE part
        fits.open(pixel_exclude_fits)
    except:
        lower_lim = fits.Column(name= 'LOWER_LIMIT',format='1D',)
        upper_lim = fits.Column(name= 'UPPER_LIMIT',format='1D',)
        new_table_pixel_exclude = fits.BinTableHDU.from_columns([lower_lim,upper_lim])
        hdu = fits.PrimaryHDU()
        hdul = fits.HDUList([hdu,new_table_pixel_exclude])
        hdul.writeto(pixel_exclude_fits)
        pass
    
    return
#%% create_sof_file
def create_sof_files(item,all_spec,force_s2d=False):
    '''
    Create SOF file list for molecfit to use for each spectrum separately
    '''
    f1 = open(sof_para_model,'w') # SOF for molecfit_model
    f1_lines = [item + ' SCIENCE\n',
                directory_molecfit_tmp_files + '/' + 'WAVE_INCLUDE.fits WAVE_INCLUDE\n',
                directory_molecfit_tmp_files + '/' + 'WAVE_EXCLUDE.fits WAVE_EXCLUDE\n',
                directory_molecfit_tmp_files + '/' + 'PIXEL_EXCLUDE.fits PIXEL_EXCLUDE\n',
                ]
    f1.writelines(f1_lines)
    f1.close()
    
    f2 = open(sof_para_calctrans,'w') # SOF for molecfit_calctrans
    f2_lines = [item + ' SCIENCE\n',
                directory_molecfit_tmp_files + '/' + 'MODEL_MOLECULES.fits MODEL_MOLECULES\n',
                directory_molecfit_tmp_files + '/' + 'ATM_PARAMETERS.fits ATM_PARAMETERS\n',
                directory_molecfit_tmp_files + '/' + 'BEST_FIT_PARAMETERS.fits BEST_FIT_PARAMETERS\n',
                directory_molecfit_tmp_files + '/' + 'BEST_FIT_MODEL.fits BEST_FIT_MODEL\n',
                ]
    f2.writelines(f2_lines)
    f2.close()
    
    f3 = open(sof_para_correct, 'w') # SOF for molecfit_correct
    if all_spec:
        file_lines = [directory_molecfit_input + '/' +i + ' SCIENCE\n' for i in os.listdir(directory_molecfit_input)]
        file_lines.append(directory_molecfit_tmp_files + '/' + 'MODEL_MOLECULES.fits MODEL_MOLECULES\n')
        file_lines.append(directory_molecfit_tmp_files + '/' + 'ATM_PARAMETERS.fits ATM_PARAMETERS\n')
        file_lines.append(directory_molecfit_tmp_files + '/' + 'BEST_FIT_PARAMETERS.fits BEST_FIT_PARAMETERS\n')
        file_lines.append(directory_molecfit_tmp_files + '/' + 'BEST_FIT_MODEL.fits BEST_FIT_MODEL\n')
        file_lines.append(directory_molecfit_tmp_files + '/' + 'TELLURIC_CORR.fits TELLURIC_CORR\n')
        f3_lines = file_lines
    elif force_s2d ==False:
        f3_lines = [item + ' SCIENCE\n',
                    directory_molecfit_tmp_files + '/' + 'MODEL_MOLECULES.fits MODEL_MOLECULES\n',
                    directory_molecfit_tmp_files + '/' + 'ATM_PARAMETERS.fits ATM_PARAMETERS\n',
                    directory_molecfit_tmp_files + '/' + 'BEST_FIT_PARAMETERS.fits BEST_FIT_PARAMETERS\n',
                    directory_molecfit_tmp_files + '/' + 'BEST_FIT_MODEL.fits BEST_FIT_MODEL\n',
                    directory_molecfit_tmp_files + '/' + 'TELLURIC_CORR.fits TELLURIC_CORR\n',
                    ]
    elif force_s2d == True:
        f3_lines = [item.replace('molecfit_input','molecfit_input_s2d') + ' SCIENCE\n',
                    directory_molecfit_tmp_files + '/' + 'MODEL_MOLECULES.fits MODEL_MOLECULES\n',
                    directory_molecfit_tmp_files + '/' + 'ATM_PARAMETERS.fits ATM_PARAMETERS\n',
                    directory_molecfit_tmp_files + '/' + 'BEST_FIT_PARAMETERS.fits BEST_FIT_PARAMETERS\n',
                    directory_molecfit_tmp_files + '/' + 'BEST_FIT_MODEL.fits BEST_FIT_MODEL\n',
                    directory_molecfit_tmp_files + '/' + 'TELLURIC_CORR.fits TELLURIC_CORR\n',
                    ]
    f3.writelines(f3_lines)
    f3.close()
    return
#%% check_empty_region
def check_empty_region():
    '''
    Checks whether fits files for region selections are empty (and comments them from using if they are)
    '''
    fits_wave_include = fits.open(wave_include_fits)
    fits_wave_exclude = fits.open(wave_exclude_fits)
    fits_pixel_exclude = fits.open(pixel_exclude_fits)
    fits_list = [fits_wave_include,fits_wave_exclude,fits_pixel_exclude]
    sof_file = open(sof_para_model,'r')
    lines_list = sof_file.readlines()
    ind_list = np.arange(3)+1
    for f,l,x in zip(fits_list,lines_list[1:],ind_list): # For each region fits file and their respective line in SOF
        if f[1].header['NAXIS2'] == 0: # Check that table is empty (no region selected)
            if not(l.startswith('# ')): # Check that line is not commented
                lines_list[x] = '# ' + l # Comment the respective line
            else: # Else leave it commented
                l = l
        else: # If table is not empty
            if (l.startswith('# ')): # Check that line not commented 
                lines_list[x] = l[2:] # Un-comment
            else: # Else leave it uncommented
                l = l
    sof_file.close()
    sof_file = open(sof_para_model,'w')
    sof_file.writelines(lines_list)
    sof_file.close()
    return

#%% 
def update_header(item):
    '''
    Updates header of the output files, add telluric profile data to the HDU list and moves it to reduced data

    Parameters
    ----------
    item : str
        Filename of the currently analyzed spectrum

    Returns
    -------
    None.

    '''
    header = fits.open(directory_molecfit_tmp_files + '/BEST_FIT_PARAMETERS.fits')
    
    f = fits.open(item.replace('molecfit_input/','molecfit_output/SCIENCE_TELLURIC_CORR_'),mode='update')
    
    
    # Add telluric profile used in this spectrum to correct the spectrum
    g = fits.open(directory_molecfit_tmp_files + '/TELLURIC_CORR.fits')
    telluric_profile = fits.PrimaryHDU(data= g[1].data)
    g2 = fits.open(item)
    
    f.append(telluric_profile)
    f.append(g2[1])
    f.append(header[1]) # Best fit information 
    f.close()
    
    
    os.replace(item, item.replace('molecfit_input','molecfit_reduced'))
    
    return
#%% Run_molecfit_all
def run_molecfit_night(directory_root: str,
                       force_s2d= False,
                       force_all= False,
                       run_noninteractive: bool = False):
    '''
    Run the entire molecfit pipeline for the given spectrum
    '''
    global force_sgl
    setup_directories(directory_root)
    
    
    if len(os.listdir(directory_molecfit_input)) == 0:
        logger.warning('The input directory is empty:')
        logger.warning('Looking into directory:')
        logger.warning(directory_molecfit_input)
        logger.info('This can happen if molecfit has already been rerun completely.')
        logger.info('In that case no action is needed and this warning can be ignored')
        return
    
    for file in os.listdir(directory_molecfit_input):
        if file.endswith('.fits'):
            create_region_fits_files(file)
            break
            
    create_region_fits_files(os.listdir(directory_molecfit_input)[0]) # Create new region fits file if not existing
    
    for file_item in os.listdir(directory_molecfit_input): # Run through all spectra
        
        
        if not(file_item.endswith('.fits')):
            pass
        item = directory_molecfit_input + '/' + file_item # Path to current spectrum
        create_sof_files(item,all_spec=False,force_s2d=force_s2d) # Create SOF files for current spectrum
        check_empty_region() # Check that region fits files are commented if empty
        
        if force_all: # Run molecfit_model and calctrans in no-GUI mode
            os.chdir(directory_molecfit_tmp_files)
            run_molecfit_model() # molecfit_model
            run_molecfit_calctrans() # molecfit_calctrans
        else: # Run molecfit_model and calctrans with GUI
            while not(force_sgl) and not(force_all): # Run model and calctrans as long as the force_sgl is not turned on
                os.chdir(directory_molecfit_tmp_files)
                run_molecfit_model() # molecfit_model
                run_molecfit_calctrans() # molecfit_calctrans
                show_gui()
                check_empty_region()
                
        # Once proper regions were selected, move to output directory and start molecfit_correct
        os.chdir(directory_molecfit_output)
        run_molecfit_correct() # molecfit_correct
        update_header(item) # Update header of resulting fits file

        logger.info('Finished reducing spectrum')
        logger.info(f'There is still {len(os.listdir(directory_molecfit_input))} spectra in current night directory')
        
        if not(force_all): # Restart the force_sgl flag
            force_sgl = False
    if not(run_noninteractive):
        force_all, force_sgl = False, False
    return
#%%
def setup_directories(directory_root: str):
    """
    Setup global variables to be used for given dataset.
    
    All the filepaths, commands and folder paths are output as a global variable, so they can be used within this file without passing through function calls.

    Parameters
    ----------
    directory_root : str
        Directory of the root molecfit dataset.
    """
    global directory_molecfit_input, directory_molecfit_output, directory_molecfit_reduced, directory_molecfit_tmp_files, directory_molecfit_para_files
    global directory_molecfit_input_s2d
    global wave_include, wave_exclude, pixel_exclude
    global rc_para_model, rc_para_calctrans, rc_para_correct
    global sof_para_model, sof_para_calctrans, sof_para_correct
    global wave_include_fits, wave_exclude_fits, pixel_exclude_fits
    global command_molecfit_model, command_molecfit_calctrans, command_molecfit_correct
    
    # Additional directories
    directory_molecfit_input = directory_root +'/molecfit_input'
    directory_molecfit_output = directory_root +'/molecfit_output'
    directory_molecfit_reduced = directory_root +'/molecfit_reduced'
    directory_molecfit_tmp_files = directory_root  +'/tmp_files'
    directory_molecfit_para_files = directory_root  +'/para_files'
    directory_molecfit_input_s2d = directory_root +'/molecfit_input_S2D'
    # regions fits files
    wave_include = directory_molecfit_tmp_files + '/WAVE_INCLUDE.fits'
    wave_exclude = directory_molecfit_tmp_files + '/WAVE_EXCLUDE.fits'
    pixel_exclude = directory_molecfit_tmp_files + '/PIXEL_EXCLUDE.fits'
    # rc params files location
    rc_para_model = directory_molecfit_para_files + '/molecfit_model.rc'
    rc_para_calctrans = directory_molecfit_para_files + '/molecfit_calctrans.rc'
    rc_para_correct = directory_molecfit_para_files + '/molecfit_correct.rc'
    # SOF files location
    sof_para_model = directory_molecfit_para_files + '/molecfit_model.sof'
    sof_para_calctrans = directory_molecfit_para_files + '/molecfit_calctrans.sof'
    sof_para_correct = directory_molecfit_para_files + '/molecfit_correct.sof'
    # Wave region fits files
    wave_include_fits = directory_molecfit_tmp_files + '/WAVE_INCLUDE.fits'
    wave_exclude_fits = directory_molecfit_tmp_files + '/WAVE_EXCLUDE.fits'
    pixel_exclude_fits = directory_molecfit_tmp_files + '/PIXEL_EXCLUDE.fits'
    # molecfit commands 
    command_molecfit_model = [command_esorex,'--recipe-config=%s'%rc_para_model,'molecfit_model',sof_para_model]
    command_molecfit_calctrans = [command_esorex,'--recipe-config=%s'%rc_para_calctrans,'molecfit_calctrans',sof_para_calctrans]
    command_molecfit_correct = [command_esorex,'--recipe-config=%s'%rc_para_correct,'molecfit_correct',sof_para_correct]
    return
#%%
def run_molecfit_all(main_directory: str,
                     force_all: bool = False,
                     run_noninteractive: bool = False):
    for root, subdir, filename in os.walk(main_directory + '/spectroscopy_data'):  
        if root.endswith('molecfit'):
            logger.info('Running molecfit on folder:')
            logger.info(root)
            run_molecfit_night(
                directory_root= root,
                force_s2d= False,
                force_all= force_all,
                run_noninteractive= run_noninteractive
            )
        else:
            logger.debug(f'Ignoring direcotry {subdir}')
        
    return
