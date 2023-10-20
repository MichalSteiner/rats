#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:04:14 2022

Template for transmission spectroscopy pipeline using the RATS (Rapid Analysis of Transmission Spectra) package.

@author: chamaeleontis
"""
# =============================================================================
# TODO:
# plots
# =============================================================================
#%% Importing libraries
# chaos library - custom TS pipeline
import rats
import rats.eso as eso # Loading ESO instruments
import rats.parameters as para # System parameters
import rats.single_use as su # Single use functions
import rats.plot_spectra as ps # Spectra plotting
import rats.spectra_manipulation as sm # Spectra manipulation
import rats.table as tab # Table creation
import rats.plot_functions as pf # Plot functions
import rats.molecfit as mol # Molecfit functions
import rats.spectra_ccf as ccf # CCF functions
# Matplotlib and seaborn
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns
# Astropy and specutils
import specutils as sp
import astropy.units as u
import astropy.constants as con
import astropy.io.fits as fits
# Others
import numpy as np
import os

#%% Select type of plots (normal vs dark mode)
# import rats.matplotlib_style_sheets.dark_mode_presentation
#%% Flags to force loads and skips of functions
force_load = False
force_skip = False
#%% Setup of directories and movement of files from downloaded folder
# TODO: Change the filepaths
data_directory = 'Add directory of the data (as extracted from DACE)'
main_directory = 'Add main directory of the project'

figure_directory = main_directory + '/figures'
directory_spectra = main_directory + '/data/spectra/ESPRESSO'
# For multiple instruments, need to define multiple directories
#%% Movement of data to predefined directory tree
# TODO: Run once, then comment/remove this cell
instruments = ['ESPRESSO']
su.setup_routine(data_directory, main_directory,instruments=instruments)
path_main, path_spectra, path_photometry, path_figures, path_figures_spectra, path_figures_photometry, path_tables = su.setup_directory_paths(main_directory,relative_path=True)
#%% Change the working directory
os.chdir(main_directory)

#%% Loading parameters
# TODO: Change the name of the system
sys_para = para.system_parameters_class('Name of the system')
sys_para.load_nasa_parameters('Name of the planet as defined by NASA archive',force_load=True)
#%% Check the original references!
# TODO: Check that the system values are correct and correct those that are not manually
#     Keep in mind, the code will run as long as you define the name of the planet above, 
#     but it doesn't take necessarilly the best values for your system!
# Use this as a first quick check only.
# =============================================================================
# Defining stellar parameters
# =============================================================================
# sys_para.star.radius = 0. * u.R_sun
# sys_para.star.radiusherr = 0. * u.R_sun
# sys_para.star.radiuslerr = 0. * u.R_sun
# sys_para.star.radiusref = ''
# sys_para.star.mass = 0. * u.M_sun
# sys_para.star.massherr = 0. * u.M_sun
# sys_para.star.masslerr = 0. * u.M_sun
# sys_para.star.massref = ''

# =============================================================================
# Defining planetary parameters
# =============================================================================
# sys_para.planet.radius = 0 * u.R_jup
# sys_para.planet.mass = 0. * u.M_jup

# =============================================================================
# Defining system parameters
# =============================================================================
# sys_para.system.a = 0. * u.au
# sys_para.system.b = 0. 
# sys_para.system.i = 0  * u.deg
# sys_para.system.e = 0
# sys_para.system.P = 0 * u.day
# sys_para.system.omega = 0 * u.deg
# sys_para.system.omega_bar = np.radians(sys_para.system.omega)
# sys_para.system.vel_s = 0 * u.km / u.s 
# sys_para.system.a_rs_ratio = 0
# sys_para.system.rs_a_ratio = 1/0

# =============================================================================
# Defining transit parameters
# =============================================================================
# sys_para.transit.semiamp_s = 0 * u.m / u.s
# sys_para.transit.T_C = 0 * u.day
# sys_para.transit.delta = 0 *u.dimensionless_unscaled # Not in percentages, but raw number (opposed to NASA archive)
# sys_para.update_contact_points()
# sys_para.transit.T14 = 0 *u.hour
# sys_para.transit.T23 = 0 *u.hour

# =============================================================================
# Print the system parameter values
# =============================================================================
sys_para.print_values()
# =============================================================================
# Create system parameters table
# This will print out latex code in the console to copy-paste into your papere
# =============================================================================
# sys_para.create_system_table()

#%% Get equivalency and custom units for given system
equivalency_transmission, F_lam, R, R_plam, delta_lam, H_num = sm.custom_transmission_units(sys_para)

#%% Loading exoplanet table 
# =============================================================================
# Load the observed exoplanet class, calculate scale height for them
# =============================================================================
exo_class = para.observed_exoplanets()
exo_class.calculate_scale_height()

#%% Plotting exoplanet figure Mass vs Insolation flux
# =============================================================================
# Radius-insolation plot
# =============================================================================
# TODO: Change the handles of planets highlighted
# TODO: Change the systems highlighted
fig, ax = pf.radius_insolation_plot(
    exo_class,
    [['Enter name(-s) of the planet(-s) to highlight']],
    [['Enter label(-s) of the planet(-s) to highlight']],
    )

#%% Loading data
# TODO: Decide on S1D or S2D spectra
# =============================================================================
# S1D - simplest to use, by default
# =============================================================================
data_deblaze_s1d_A = eso.load_all(directory_spectra,
                                  'Fiber_A',
                                  'S1D_SKYSUB',
                                  force_skip = force_skip
                                  )
data_deblaze_s1d_B = eso.load_all(directory_spectra,
                                  'Fiber_B',
                                  'S1D',
                                  force_skip = force_skip
                                  )
# =============================================================================
# S2D - easiest to handle after order extraction, molecfit correction annoying
# Good for single species check
# =============================================================================
# data_raw_s2d_A, data_deblaze_s2d_A = eso.load_all(directory_spectra, 'Fiber_A', 'S2D')
# data_raw_s2d_B, data_deblaze_s2d_B = eso.load_all(directory_spectra, 'Fiber_B', 'S2D')
# =============================================================================
# Molecfit corrected spectra
# =============================================================================
# data_deblaze_s1d_A, telluric_profiles, data_uncorrected = mol.molecfit_new_output(
#                         instrument_directory = directory_spectra,
#                         spec_type='S1D')
#%% Telluric quality control
# =============================================================================
# Calculates master of uncorrected and telluric corrected spectra
# Overplot it with telluric profile to check for potential spurious feature
# =============================================================================


#%% Define phases, velocities and transit values
# =============================================================================
# TODO: Replace the S1D name with S2D if using it instead
# =============================================================================
eso.update_phase_and_vel(data_deblaze_s1d_A,sys_para)
para.update_transit(data_deblaze_s1d_A,sys_para)
#%% Calculate master A and B spectrum
# =============================================================================
# Calculate master fiber A and B spectrum
# =============================================================================
fiber_A_master = sm.calculate_master_list(data_deblaze_s1d_A,key = None,)
fiber_B_master = sm.calculate_master_list(data_deblaze_s1d_B,key = None,)
#%% Plot both fibers combined
# =============================================================================
# Plot both masters combined
# TODO: Create a function for this and define it here
# =============================================================================

#%% Removing low SNR spectrum
# =============================================================================
# Filtering bad spectra
# Common issues are: 
# Low SNR (<10)
# High airmass (>2.2 for ESPRESSO, or >2)
# Removing entire night (weather conditions)
# See documentation of sm.get_sublist() function for more information 
# =============================================================================
# data_deblaze_s1d_A = sm.get_sublist(data_deblaze_s1d_A,'S_N',10,mode='more')

#%% Create observational log + table
# =============================================================================
# Observation log - create table and plot
# TODO: Define orders at which to take SNR
# =============================================================================
'''Table already saved in corresponding directory'''
# plt.rcParams["figure.figsize"] = 9.26, 11.24
# obs_log = para.observations_log(data_deblaze_s1d,orders = [])
# #%%
# table = tab.table_obs_log(obs_log,[])
# fig_list = ps.plot_obs_log(obs_log,sys_para,orders=[])
# plt.rcParams["figure.figsize"] = 18.52, 11.24
# for ii in range(2):
#     fig_list[ii][0].savefig(figure_directory+'obs_log%i.pdf'%(ii))

#%% Check master of Fiber A and B
# =============================================================================
# Master of Fiber A and B - comparison
# TODO: 
    # Add telluric profile to this plot
    # Write as a function
# =============================================================================
# master_A = sm.calculate_master_list(data_deblaze_s1d,sn_type='None')
# master_B = sm.calculate_master_list(data_deblaze_B,sn_type='None')

# fig,axs = plt.subplots(2,sharex=True)
# ni = 0
# axs[0].plot(master_A[ni].spectral_axis,master_A[ni].flux)
# axs[1].plot(master_B[ni].spectral_axis,master_B[ni].flux)

# fig,axs = plt.subplots(2,sharex=True)
# ni = 1
# axs[0].plot(master_A[ni].spectral_axis,master_A[ni].flux)
# axs[1].plot(master_B[ni].spectral_axis,master_B[ni].flux)

#%% Rebinning and normalizing spectra
# =============================================================================
# Rebinning to common wavelength frame and normalizing spectra
# Replace the input of sm.binning list for S2D spectra
# TODO: Replace these functions with save/load-able pickles for the code to run faster
# =============================================================================
data_deblaze_s1d_A_sorted = sm.sort_spec_list(data_deblaze_s1d_A,
                                              force_skip = force_skip
                                              )
data_binned = sm.binning_list(data_deblaze_s1d_A_sorted,
                              force_skip = force_skip
                              )
data_normalized = sm.normalize_list(data_binned,
                                    force_load = force_load,
                                    force_skip = force_skip
                                    )

#%% Correction of cosmic rays
# =============================================================================
# OPTIMIZE: Check for this function and see if it can't be optimized
# =============================================================================
data_cosmic_corrected = sm.cosmic_correction_all(data_normalized,
                                                 force_load = force_load,
                                                 force_skip = force_skip
                                                 )
#%% Moving to rest frame of star
# =============================================================================
# Move from rest frame of Earth to rest frame of Star
# Careful for BERV value, different format work differently
# New DRS tend to correct BERV by themselves
# TODO: Replace these functions with save/load-able pickles for the code to run faster
# =============================================================================
data_SRF = sm.shift_list(data_cosmic_corrected,
                                 shift_BERV=0,
                                 shift_v_sys = 1,
                                 shift_v_star = 1,
                                 shift_v_planet = 0,
                                 force_load = force_load,
                                 force_skip = force_skip,
                                 pkl_name = 'data_SRF.pkl'
                                 )
# data_SRF = sm.normalize_list(data_SRF)

#%% Master out calculation and correction
# =============================================================================
# Calculate the master out and correct for it
# If RM effect is significant, include the obliquity and vsini values
# =============================================================================
master_SRF_out = sm.calculate_master_list(data_SRF,
                                          key = 'Transit',
                                          value =False,
                                          force_load = force_load,
                                          force_skip = force_skip,
                                          pkl_name = 'master_out_SRF.pkl'
                                          )
master_SRF_int = sm.calculate_master_list(data_SRF,
                                          key = 'Transit',
                                          value =True,
                                          force_load = force_load,
                                          force_skip = force_skip,
                                          pkl_name = 'master_in_SRF.pkl'
                                          )
data_out_corrected = sm.spec_list_master_correct(data_SRF,
                                                 master_SRF_out,
                                                 force_load = force_load,
                                                 force_skip = force_skip,
                                                 pkl_name = 'star_corrected_noRM.pkl'
                                                 )
# RM_star_56,RM_star_56_all = sm.rm_list_correct(star_56,
#                                 master_RF_star_oot_56,
#                                 sys_para)
#%% Shifting to RF planet
# =============================================================================
# Shift the data to planetary rest frame
# =============================================================================
data_PRF = sm.shift_list(data_out_corrected,
                                 shift_BERV=0,
                                 shift_v_sys = 0,
                                 shift_v_star = -1,
                                 shift_v_planet = 1,
                                 force_load = force_load,
                                 force_skip = force_skip,
                                 pkl_name = 'data_PRF.pkl'
                                 )

#%% Calculation of transmission spectrum
# =============================================================================
# Calculate the transmission spectrum and out-of-transit master
# transmission spectrum should have only planetary atmosphere signal (and uncorrected effects like RM and CLV)
# Out of transit spectrum should be flat 
# Until now all the spectra are in units of excess absorption,
# but both transmission_spectrum and out_spectrum with custom units that can be 
# transfered with .to(unit) command (F, R_plam, H_num, delta_lam) 
# =============================================================================
transmission_spectrum = sm.calculate_master_list(data_PRF,
                                                 key = 'Transit',
                                                 value =True,
                                                 sn_type='quadratic',
                                                 force_load = force_load,
                                                 force_skip = force_skip,
                                                 pkl_name = 'transmission_spectrum.pkl'
                                                 )
out_spectrum = sm.calculate_master_list(data_PRF,
                                        key = 'Transit',
                                        value =False,
                                        sn_type='quadratic',
                                        force_load = force_load,
                                        force_skip = force_skip,
                                        pkl_name = 'out_corrected_out_PRF_spectrum.pkl'
                                        )
transmission_spectrum = sm.replace_flux_units_transmission(transmission_spectrum, R)
out_spectrum = sm.replace_flux_units_transmission(out_spectrum, R)

#%% Plot transmission spectrum
# =============================================================================
# Plot the transmission spectrum with rough estimates on the uncertainties
# =============================================================================
ps.plot_transmission_spectrum(transmission_spectrum,
                              sys_para,
                              rats.lists.sodium_doublet,
                              )

