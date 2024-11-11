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
                    ) -> tuple[sp.SpectrumList, sp.SpectrumList, sp.SpectrumList]:
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
                
                corrected_spectrum.meta['Night'] = night #type: ignore
                telluric_profile.meta['Night'] = night #type: ignore
                uncorrected_spectrum.meta['Night'] = night #type: ignore
    
    eso._numbering_nights(corrected_spectra)
    eso._numbering_nights(uncorrected_spectra)
    eso._numbering_nights(telluric_profiles)
    
    return corrected_spectra, telluric_profiles, uncorrected_spectra

def _load_molecfit_output_single_spectrum(filename: str,
                                          spectrum_type: str = 'S1D',
                                          mask_threshold: float = 0) -> tuple[sp.Spectrum1D | sp.SpectrumCollection, sp.Spectrum1D | sp.SpectrumCollection, sp.Spectrum1D | sp.SpectrumCollection]:
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
        'header': f[0].header, #type: ignore
        'BERV_corrected': True,
        'RF_Barycenter': True,
        'RF': 'Barycenter_Sol',
        'vacuum': False,
        'air': True
        })
    meta.update(eso._load_meta_from_header(f[0].header)) #type: ignore
    
    telluric_profile = sp.Spectrum1D( # Define the spectrum
        spectral_axis = f[1].data['wavelength_air']*u.AA, #type: ignore
        flux = f[2].data*u.dimensionless_unscaled, #type: ignore
        uncertainty = astropy.nddata.StdDevUncertainty(np.zeros_like(f[2].data)),#type: ignore
        meta= meta,
        mask= np.isnan(f[2].data)#type: ignore
        )
    
    mask_ind = np.where(telluric_profile.flux.value < mask_threshold)
    
    flux = f[1].data['flux']#type: ignore
    error = f[1].data['error']#type: ignore
    flux[mask_ind] = np.nan
    error[mask_ind] = np.nan
    
    
    corrected_spectrum = sp.Spectrum1D( # Define the spectrum
        spectral_axis = f[1].data['wavelength_air']*u.AA,#type: ignore
        flux = flux*u.ct,#type: ignore
        uncertainty = astropy.nddata.StdDevUncertainty(error),#type: ignore
        meta = meta,
        mask = np.isnan(flux),
        )
    
    uncorrected_spectrum = sp.Spectrum1D( # Define the spectrum
        spectral_axis = f[3].data['wavelength_air']*u.AA,#type: ignore
        flux = f[3].data['flux']*u.ct,#type: ignore
        uncertainty = astropy.nddata.StdDevUncertainty(f[3].data['error']),#type: ignore
        meta = meta,
        mask = np.isnan(f[3].data['flux']),#type: ignore
        )
    
    return corrected_spectrum, telluric_profile, uncorrected_spectrum


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

