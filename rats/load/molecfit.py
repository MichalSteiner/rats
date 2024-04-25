# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:13:38 2021

@author: Chamaeleontis
"""

#%% Importing libraries
import numpy as np 
import astropy.io.fits as fits
import astropy.units as u
import specutils as sp
import os as os
import astropy
import rats.load.eso as eso
import rats.spectra_manipulation as sm
from rats.utilities import time_function, save_and_load, progress_tracker, disable_func, skip_function, default_logger_format
import logging

logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

#%%
@progress_tracker
def molecfit_output(main_directory: str,
                    spectrum_type: str = 'S1D',
                    mask_threshold: float = 0,
                    ) -> [sp.SpectrumList, sp.SpectrumList, sp.SpectrumList]:
    """
    Loads output of molecfit correction, as run by "run_molecfit_all" module.

    Parameters
    ----------
    main_directory : str
        Main directory of the project.
    spectrum_type : str, optional
        Type of spectrum, by default 'S1D'. Currently, no other modes are usable, but expansion to S2D format is planned.
    mask_threshold : float, optional
        Masking threshold, by default 0. Telluric profile is value between 0 and 1, with 0 being full absorption and 1 being no telluric absorption. As such, higher value masks more data. This is useful to avoid regions with strong contamination where the model is too unprecise and the uncertainty is not truthful to the data. 

    Returns
    -------
    corrected_spectra : sp.SpectrumList
        Molecfit corrected spectra.
    telluric_profiles : sp.SpectrumList
        Telluric profiles used for correction. Used for masking with threshold.
    uncorrected_spectra : sp.SpectrumList
        Original uncorrected spectra.
    """
    
    spectrum_directory = main_directory + '/spectroscopy_data/'
    
    corrected_spectra = sp.SpectrumList()
    telluric_profiles = sp.SpectrumList()
    uncorrected_spectra = sp.SpectrumList()
    
    logger.info('Loading molecfit output:')
    logger.info('='*50)
    for instrument in os.listdir(spectrum_directory):
        instrument_directory = spectrum_directory + '/' + instrument
        logger.info(f'Loading instrument: {instrument}')
        for night in os.listdir(instrument_directory):
            logger.info(f'    Loading night: {night}')
            molecfit_output_path = instrument_directory + '/' + night + '/Fiber_A/S1D/molecfit/molecfit_output/'
            
            for item in os.listdir(molecfit_output_path):
                if not(item.endswith('.fits')) or not(item.startswith('SCIENCE')):
                    continue
                
                corrected_spectrum, telluric_profile, uncorrected_spectrum = _load_molecfit_output_single_spectrum(molecfit_output_path + item, spectrum_type, mask_threshold)
                corrected_spectra.append(corrected_spectrum)
                telluric_profiles.append(telluric_profile)
                uncorrected_spectra.append(uncorrected_spectrum)
                
    
    eso._numbering_nights(corrected_spectra)
    eso._numbering_nights(uncorrected_spectra)
    eso._numbering_nights(telluric_profiles)
    
    return corrected_spectra, telluric_profiles, uncorrected_spectra

def _load_molecfit_output_single_spectrum(filename: str,
                                          spectrum_type: str = 'S1D',
                                          mask_threshold: float = 0) -> [sp.Spectrum1D | sp.SpectrumCollection, sp.Spectrum1D | sp.SpectrumCollection, sp.Spectrum1D | sp.SpectrumCollection]:
    """
    Load a single molecfit corrected spectrum.

    Parameters
    ----------
    filename : str
        Filename to the fits file to open.
    spectrum_type : str, optional
        Type of spectrum, by default 'S1D'. Currently, no other modes are usable, but expansion to S2D format is planned.
    mask_threshold : float, optional
        Masking threshold, by default 0. Telluric profile is value between 0 and 1, with 0 being full absorption and 1 being no telluric absorption. As such, higher value masks more data. This is useful to avoid regions with strong contamination where the model is too unprecise and the uncertainty is not truthful to the data. 


    Returns
    -------
    corrected_spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Corrected spectrum by molecfit
    telluric_profile : sp.Spectrum1D | sp.SpectrumCollection
        Telluric profile used by molecfit
    uncorrected_spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Uncorrected spectrum before molecfit correction

    Raises
    ------
    NotImplementedError
        For non S1D data, no implementation is done yet.
    """
    
    
    if spectrum_type != 'S1D':
        raise NotImplementedError('It is not possible to load S2D files yet.')
    
    f = fits.open(filename) 
    
    meta = eso._basic_meta_parameters()
    meta.update({
        'header': f[0].header,
        'BERV_corrected': True,
        'RF_Barycenter': True,
        'RF': 'Barycenter_Sol',
        'vacuum': False,
        'air': True
        })
    meta.update(eso._load_meta_from_header(f[0].header))
    
    telluric_profile = sp.Spectrum1D( # Define the spectrum
        spectral_axis = f[1].data['wavelength_air']*u.AA,
        flux = f[2].data*u.dimensionless_unscaled,
        uncertainty = astropy.nddata.StdDevUncertainty(np.zeros_like(f[2].data)),
        meta= meta,
        mask= np.isnan(f[2].data)
        )
    
    # from rats.spectra_manipulation import _shift_spectrum
    
    # telluric_profile = _shift_spectrum(telluric_profile,
    #                                    velocities= [telluric_profile.meta['velocity_BERV']])
    
    mask_ind = np.where(telluric_profile.flux.value < mask_threshold)
    
    flux = f[1].data['flux']
    error = f[1].data['error']
    flux[mask_ind] = np.nan
    error[mask_ind] = np.nan
    
    
    corrected_spectrum = sp.Spectrum1D( # Define the spectrum
        spectral_axis = f[1].data['wavelength_air']*u.AA,
        flux = flux*u.ct,
        uncertainty = astropy.nddata.StdDevUncertainty(error),
        meta = meta,
        mask = np.isnan(flux),
        )
    
    uncorrected_spectrum = sp.Spectrum1D( # Define the spectrum
        spectral_axis = f[3].data['wavelength_air']*u.AA,
        flux = f[3].data['flux']*u.ct,
        uncertainty = astropy.nddata.StdDevUncertainty(f[3].data['error']),
        meta = meta,
        mask = np.isnan(f[3].data['flux']),
        )
    
    return corrected_spectrum, telluric_profile, uncorrected_spectrum

# def _masking_flux_based_on_threshold(tellr,
#                                      uncertainty,
#                                      profile
#                                      mask_threshold):
#     return


#%% molecfit_new_output
# =============================================================================
# Likely bugged with telluric profiles being in different rest frame
# =============================================================================
@progress_tracker
def molecfit_new_output(main_directory: str,
                        spec_type: str = 'S1D',
                        mask_threshold: float = 0,
                        )->[sp.SpectrumList, sp.SpectrumList, sp.SpectrumList]:
    '''
    Load molecfit output files.

    Parameters
    ----------
    instrument_directory : str
        Location of data for instrument
    spec_type : str, optional
        Type of the spectra (defines the spec_type directory). The default is 'S1D'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    spec_list = sp.SpectrumList()
    spec_list_profile = sp.SpectrumList()
    spec_list_noncorrected = sp.SpectrumList()
    
    
    
    
    for instrument in os.listdir(main_directory+'/spectroscopy_data'):
        for ind, night in enumerate(os.listdir(main_directory+'/spectroscopy_data/'+instrument)): # Looping through each night
            molecfit_output_directory = main_directory+'/spectroscopy_data/'+instrument + '/' + night + '/Fiber_A/S1D/molecfit/molecfit_output'
            
            for ii,item in enumerate(os.listdir(molecfit_output_directory)): # Looping through each file
                if item.startswith('SCIENCE'): # Only SCIENCE fits files
                
                    f = fits.open(molecfit_output_directory + '/' + item) 
                    meta = {'header':f[0].header}
                    
                    
                    
                    meta.update(eso.load_meta_from_header_new_pipeline(f[0].header))
                    meta.update(eso.set_meta_parameters())
                    meta.update(eso.set_meta_velocity_corrections())
                    meta.update({
                                    'Night':night,
                                    'Night_num': ind+1,
                                    'Night_spec_num':ii+1,
                        })
                    
                    eso.set_filename_header(meta,molecfit_output_directory + '/' + item)
                    meta['blaze_corrected'] = True
                    meta['telluric_corrected'] = True
                    meta['S_N']= meta['Average S_N']
                    meta.update({
                                    'Night':night,
                                    'Night_num': ind+1,
                                    'Night_spec_num':ii+1,
                        })
                    mask_ind = np.where(f[2].data<mask_threshold)
                    flux = f[1].data['flux']
                    error = f[1].data['error']
                    flux[mask_ind] = np.nan
                    error[mask_ind] = np.nan
                    
                    tmp_spec = sp.Spectrum1D( # Define the spectrum
                        spectral_axis = f[1].data['wavelength_air']*u.AA,
                        flux = flux*u.ct,
                        uncertainty = astropy.nddata.StdDevUncertainty(error),
                        meta = meta,
                        mask = np.isnan(flux),
                        wcs = sm.add_spam_wcs(),
                        )
                    spec_list.append(tmp_spec) # Append the spectrum
                    
                    tmp_spec_profile = sp.Spectrum1D( # Define the spectrum
                        spectral_axis = f[1].data['wavelength_air']*u.AA,
                        flux = f[2].data*u.ct,
                        wcs = sm.add_spam_wcs(),
                        )
                    spec_list_profile.append(tmp_spec_profile) # Append the spectrum
                    
                    tmp_spec_noncorrected = sp.Spectrum1D( # Define the spectrum
                        spectral_axis = f[3].data['wavelength_air']*u.AA,
                        flux = f[3].data['flux']*u.ct,
                        uncertainty = astropy.nddata.StdDevUncertainty(f[3].data['error']),
                        meta = meta,
                        mask = np.isnan(f[3].data['flux']),
                        wcs = sm.add_spam_wcs(),
                        )
                    spec_list_noncorrected.append(tmp_spec_noncorrected) # Append the spectrum
                    
                    del(tmp_spec) # Delete the variable
                    del(tmp_spec_profile)
                
    eso.set_meta_numbering(spec_list)
    eso.set_meta_numbering(spec_list_noncorrected)
    return spec_list, spec_list_profile,spec_list_noncorrected # Returns


#%%
'''Shifting air to vac and vac to air wavelength'''
def airtovac(wlnm):
    wlA=wlnm*10.0
    s = 1e4 / wlA
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return(wlA*n/10.0)

def vactoair(wlnm):
    wlA = wlnm*10.0
    s = 1e4/wlA
    f = 1.0 + 5.792105e-2/(238.0185e0 - s**2) + 1.67917e-3/( 57.362e0 - s**2)
    return(wlA/f/10.0)

#%% import_spectra_output
def import_spectra_output_e2ds(root_directory_spec,order):
    '''
    Input:
        root_directory_spec ; directory of spectra data
    Output:
        spec_list ; sp.SpectrumList - telluric corrected
    '''
    # Allocation of spectrum lists
    # All spectra
    spec_list = sp.SpectrumList()
    # Spectra from one night
    tmp_spec_list = sp.SpectrumList()
    profile_list = sp.SpectrumList()
    # For cycle
    for instrument in os.listdir(root_directory_spec):
        # Going through all instrument, assigning night directory
        night_directory = root_directory_spec + '/' + instrument
        for ind,night in enumerate(os.listdir(night_directory)):
            # Going through spectra from one night
            # Where is data saved
            telluric_corrected_directory = night_directory + '/' + night + '/Fiber_A/e2ds/telluric_corrected/%i'%(order) 
            # Cleaning one night spec_list
            tmp_spec_list = sp.SpectrumList()
            try:
                os.listdir(telluric_corrected_directory)
            except:
                break

            for spectra in os.listdir(telluric_corrected_directory):
                # For cycle for all spectra in directory
                filename = telluric_corrected_directory + '/' + spectra
                # Open table
                telluric_table_out = fits.open(filename)
                # Get flux, wavelength and error
                tmp_flux = telluric_table_out[1].data['tacflux'] * u.ct
                tmp_spectral_axis = telluric_table_out[1].data['wavelength'] * u.AA
                tmp_uncertainty = astropy.nddata.StdDevUncertainty(telluric_table_out[1].data['err_flux'] ,unit='ct')
                # Copy uncertainty * 0 for telluric profile (assuming perfect telluric profile)
                tmp_uncertainty_profile = astropy.nddata.StdDevUncertainty(telluric_table_out[1].data['err_flux'] *0 ,unit=u.dimensionless_unscaled)
                # Updating header for the spectrum
                header = telluric_table_out[0].header
                meta  = eso.load_meta_from_header_old_pipeline(header)
                meta.update(eso.set_meta_parameters())
                meta.update({'header':header})
                meta['blaze_corrected'] = True
                meta['telluric_corrected'] = True
                # Appending to one night spec_list
                tmp_spec_list.append(sp.Spectrum1D(
                spectral_axis = tmp_spectral_axis,
                flux = tmp_flux,
                uncertainty = tmp_uncertainty,
                meta = meta,
                mask = np.isnan(tmp_flux),
                wcs = sm.add_spam_wcs(),
                ))
                # Appending telluric profile list 
                profile_list.append(sp.Spectrum1D(
                flux = telluric_table_out[1].data['mtrans'] * u.dimensionless_unscaled,
                spectral_axis = telluric_table_out[1].data['wavelength'] * u.AA,
                uncertainty = tmp_uncertainty_profile, 
                mask = None,
                meta = None,
                wcs = sm.add_spam_wcs(),
                ))
            print(tmp_spectral_axis)
            # Update one night meta data
            eso.set_meta_night_parameters(tmp_spec_list, night, (ind+1))
            # Append to resulting spectrum list
            spec_list.extend(tmp_spec_list)
    # Update spectrum numbering
    eso.set_meta_numbering(spec_list)
    return spec_list,profile_list

