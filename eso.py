# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 06:39:17 2021

@author: Michal Steiner

Functions used with HARPS, ESPRESSO and NIRPS spectrographs

ESO provides three main formats for these instruments:
    S2D : The echelle 2D spectrum as detected on the CCD.
    S1D : Rebinned, interpolated data to 1D spectrum. This format is further corrected for several effects, including dispersion effect.
    CCF : CCF calculated using a stellar template.
    
To use this code, there are several options:
Easiest is to use the load_all() function, which assumes directory setup by single_use functions.
There exist also load_night and load_instrument functions to load a given night or given instrument.
Finally, a single load_spectrum function can be used on individiual filename to load. 
The S1D and S2D spectra are providing masking pixels with negative flux and nans using the Spectrum1D.mask | SpectrumCollection.mask attributes. However, these are not applied by default, and actual masking of these pixels is requested by the user.

Currently the tested version of DRS pipeline are 3.0.0 of ESPRESSO DRS. If different format is used, will raise an exception. You can try and test other version of DRS by commenting/removing the DRS check, but the pipeline is not tested there.


"""

#%% Importing libraries
import numpy as np
import astropy.io.fits as fits
import astropy
import astropy.units as u
import specutils as sp
import os
from enum import Enum
import logging
from rats.utilities import time_function, save_and_load, progress_tracker, skip_function, disable_func, default_logger_format, todo_function

#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

def _replace_nonpositive_with_nan(spectrum: sp.Spectrum1D | sp.SpectrumCollection) -> sp.Spectrum1D | sp.SpectrumCollection:
    """
    Replace non-positive values in spectrum with NaNs.

    Parameters
    ----------
    spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Input spectrum

    Returns
    -------
    sp.Spectrum1D | sp.SpectrumCollection
        Output spectrum.
    """
    
    
    new_flux = np.where(spectrum.flux > 0, spectrum.flux, np.nan)
    new_uncertainty = np.where(spectrum.flux > 0, spectrum.uncertainty.array, np.nan)
    new_uncertainty = astropy.nddata.StdDevUncertainty(new_uncertainty)
    
    match type(spectrum):
        case sp.Spectrum1D:
            new_spectrum = sp.Spectrum1D(
                spectral_axis= spectrum.spectral_axis,
                flux= new_flux * spectrum.flux.unit,
                uncertainty= new_uncertainty,
                meta= spectrum.meta.copy(),
                mask = np.isnan(new_flux)
            )
        case sp.SpectrumCollection:
            new_spectrum = sp.SpectrumCollection(
                spectral_axis= spectrum.spectral_axis,
                flux= new_flux * spectrum.flux.unit,
                uncertainty= new_uncertainty,
                meta= spectrum.meta.copy(),
                mask = np.isnan(new_flux)
            )
        case _:
            raise NotImplementedError('Requested format is not available')
    return new_spectrum

#%% Load spectrum
def load_spectrum(filename: str) -> sp.Spectrum1D:
    """
    Load a single spectrum based on filename.

    Parameters
    ----------
    filename : str
        Location of the fits file.

    Returns
    -------
    sp.Spectrum1D
        Loaded spectrum.

    Raises
    ------
    KeyError
        If spectral type format is not available.
    """
    fits_file = fits.open(filename)
    header = fits_file[0].header
    _check_DRS_version(header['HIERARCH ESO PRO REC1 PIPE ID'])
    logger.debug('Opening file:')
    logger.debug('    ' + filename) 
    
    type_spec = header['HIERARCH ESO PRO CATG']
    if 'S1D' in type_spec:
        logger.debug('Detected S1D spectrum, trying to open.')
        spectrum = load_S1D_spectrum(fits_hdulist= fits_file)
    elif 'S2D' in type_spec:
        logger.debug('Detected S2D spectrum, trying to open.')
        spectrum = load_S2D_spectrum(fits_hdulist= fits_file)
    elif 'CCF' in type_spec:
        logger.debug('Detected CCF spectrum, trying to open.')
        spectrum = load_CCF_spectrum(fits_hdulist= fits_file)
    else:
        raise KeyError("Not a viable format of spectra.")
    
    spectrum = _replace_nonpositive_with_nan(spectrum)
    return spectrum

#%% Load S1D spectrum
def load_S1D_spectrum(fits_hdulist: fits.hdu.hdulist.HDUList) -> sp.Spectrum1D:
    """
    Load a S1D spectrum.

    Parameters
    ----------
    fits_hdulist : fits.hdu.hdulist.HDUList
        Opened fits HDUList.

    Returns
    -------
    sp.Spectrum1D
        Loaded spectrum
    """
    
    data = fits_hdulist[1].data
    main_header = fits_hdulist[0].header
    data_header = fits_hdulist[1].header
    
    # Load wavelength, flux and flux error fields with correct units based on TUNIT and TTYPE fields.
    for ii in range(data_header['TFIELDS']):
        if data_header['TTYPE%s'%(ii+1)] == 'wavelength_air':
            wavelength_unit = _find_unit(data_header['TUNIT%s'%(ii+1)])
            wavelength = data['wavelength_air'] * wavelength_unit
        
        elif data_header['TTYPE%s'%(ii+1)] == 'flux':
            flux_unit = _find_unit(data_header['TUNIT%s'%(ii+1)])
            flux = data['flux'] * flux_unit
            
        elif data_header['TTYPE%s'%(ii+1)] == 'error':
            error_unit = _find_unit(data_header['TUNIT%s'%(ii+1)])
            error = data['error']
    
    meta = _basic_meta_parameters()
    meta.update({
        'header': main_header,
        'BERV_corrected': True,
        'RF_Barycenter': True,
        'RF': 'Barycenter_Sol',
        'vacuum': False,
        'air': True
        })
    meta.update(_load_meta_from_header(main_header))
    
    spectrum = sp.Spectrum1D(
        spectral_axis= wavelength,
        flux= flux,
        uncertainty= astropy.nddata.StdDevUncertainty(error, copy=True),
        mask = _mask_flux_array(flux),
        meta = meta,
        )
    return spectrum


#%% Load S2D spectrum
def load_S2D_spectrum(fits_hdulist: fits.hdu.hdulist.HDUList) -> sp.SpectrumCollection:
    """
    Load S2D spectrum.

    Parameters
    ----------
    fits_hdulist : fits.hdu.hdulist.HDUList
        Fits HDUList including the S2D spectrum, as loaded from .fits file provided by DACE.

    Returns
    -------
    sp.SpectrumCollection
        SpectrumCollection (S2D format) with n_orders x n_pixels shape. It is corrected for the dispersion effect on the S2D spectra using the eso._correct_dispersion_S2D() method.
    """
    main_header = fits_hdulist[0].header # Preparing header of spectra
    flux = fits_hdulist['SCIDATA'].data # Flux array
    flux_err = fits_hdulist['ERRDATA'].data # Flux error array
    wavelength_air = fits_hdulist['WAVEDATA_AIR_BARY'].data # Wavedata - air (ground instruments)
    nb_orders = fits_hdulist['SCIDATA'].header['NAXIS2'] # Number of orders 
    
    flux = _correct_dispersion_S2D(wavelength_air,
                                   flux,
                                   flux_err)
    meta = _basic_meta_parameters()
    meta.update({
        'header': main_header,
        'BERV_corrected': True,
        'RF_Barycenter': True,
        'RF': 'Barycenter_Sol',
        'vacuum': False,
        'air': True
        })
    
    meta.update(_load_meta_from_header(main_header))
    
    spectrum = sp.SpectrumCollection(
        flux = flux * u.ct,
        spectral_axis = wavelength_air * u.AA,
        uncertainty = astropy.nddata.StdDevUncertainty(flux_err, copy= True),
        meta = meta,
        mask = _mask_flux_array(flux)
        )
    return spectrum

#%% Load CCF spectrum
@todo_function
def load_CCF_spectrum(fits_hdulist: fits.hdu.hdulist.HDUList):
    #TODO Implement CCF
    raise NotImplementedError("Not implemented yet")

    return
  
#%% Load all spectra from project
@save_and_load
@progress_tracker
@time_function
def load_all(main_directory: str,
             spectra_format: str,
             fiber: str,
             instrument_list: list | None = None,
             force_load: bool= False,
             force_skip: bool= False,
             pkl_name: str | None = None,
             ) -> sp.SpectrumList:
    """
    Load all spectra from all instruments and all nights in one spectrum list.

    Parameters
    ----------
    main_directory : str
        Directory of the project. Spectra are then saved in /maindirectory/data/spectra folder.
    spectra_format : str
        Format of the spectra. Options are based on _SpectraFormat class:
            'S1D_SKYSUB'
            'S1D'
            'S1D_SKYSUB'
            'S2D_SKYSUB'
            'S2D_BLAZE'
            'CCF'
            'CCF_SKYSUB'
    fiber : str
        Fiber of the spectra to use. Options are based on _Fiber class:
            'A'
            'B'
    instrument_list : list | None, optional
        List of instrument to filter through, by default None.
    force_load : bool, optional
        Force loading the output, instead of running the function, by default False
    force_skip : bool, optional
        Force skipping the function, instead of running the function, by default False
    pkl_name : str | None, optional
        Where to save the data as pickle file, by default None

    Returns
    -------
    sp.SpectrumList
        List of spectra of all requested instruments and all nights. 

    Raises
    ------
    ValueError
        Raises error on wrong combination of fiber and format of spectra.
    """
    if (fiber == 'B' and
        'SKYSUB' in spectra_format):
        raise ValueError('Invalid combination of fiber and spectra format. There are no SKYSUB_B files.')
    
    spectra_list = sp.SpectrumList()
    for instrument in os.listdir(main_directory + '/spectroscopy_data/'):
        if (instrument_list is not None) and (instrument not in instrument_list):
            logger.info('Ignoring false instrument in folder: ' + instrument)
            continue
        elif instrument.startswith('.'):
            logger.info('Ignoring false instrument in folder: ' + instrument)
            continue
        else:
            logger.info('Loading instrument: ' + instrument)
            instrument_directory = main_directory + '/spectroscopy_data/' + instrument
            spectra_list.extend(
                load_instrument(
                    instrument_directory= instrument_directory,
                    spectra_format= spectra_format,
                    fiber= fiber,
                    )
                )
    _numbering_nights(spectra_list)
    
    return spectra_list

#%% Load instrument folder
@progress_tracker
def load_instrument(instrument_directory: str,
                    spectra_format: str,
                    fiber: str,
                    ) -> sp.SpectrumList:
    """
    Load all spectra observed with a given instrument.

    Parameters
    ----------
    instrument_directory : str
        Directory of all spectra for given instrument.
    spectra_format : str
        Spectra format which to load.
    fiber : str
        Fiber which to load.

    Returns
    -------
    sp.SpectrumList
        List of spectra of given instruments and all nights.
    """
    spectra_list = sp.SpectrumList()
    for night in os.listdir(instrument_directory):
        if night.startswith('.'):
            logger.info('Ignoring false night: '+ night)
            continue
        else:
            logger.debug('Loading instrument directory:')
            logger.debug(instrument_directory)
            logger.info('Loading night: ' + instrument_directory + '/' + night)
            night_directory = instrument_directory + '/' + night
            spectra_list.extend(
                load_night(
                    night_directory= night_directory,
                    spectra_format= spectra_format,
                    fiber= fiber
                    )
                )
    return spectra_list
#%% Load night of a given instrument
@progress_tracker
def load_night(night_directory: str,
               spectra_format: str,
               fiber: str
               ) -> sp.SpectrumList:
    """
    Load all spectra from single night of a single instrument observations.

    Parameters
    ----------
    night_directory : str
        Directory of given night and instrument.
    spectra_format : str
        Spectra format which to load. 
    fiber : str
        Fiber which to load

    Returns
    -------
    sp.SpectrumList
        List of spectra of given instrument and given night. 
    """
    spectra_list = sp.SpectrumList()
    spectra_directory = (night_directory + '/' + 
                         _Fiber[fiber].value[1] + '/' + 
                         _SpectraFormat[spectra_format].value[0] + '/' +  
                         _SpectraFormat[spectra_format].value[1]
                         )
    
    if not(os.path.exists(spectra_directory)):
        logger.critical('The directory does not exist')
        logger.critical('    ' + spectra_directory)
        return spectra_list
    elif len(os.listdir(spectra_directory)) == 0:
        logger.critical('The directory %s is empty'%(spectra_directory))
        logger.critical('    ' + spectra_directory)
        return spectra_list
    
    for filename in os.listdir(spectra_directory):
        if not(filename.endswith('.fits')):
            logger.info('Ignoring file with no fits extension: '+ filename)
            continue
        logger.debug('Opening spectrum with filename:')
        logger.debug(spectra_directory + '/' + filename)
        spectra_list.append(
            load_spectrum(
                filename= spectra_directory + '/' + filename
                )
            )
    return spectra_list

#%% SpectraFormat
class _SpectraFormat(Enum):
    """
    Viable spectra formats to load. 
    
    Raises KeyError if wrong format has been used.
    """
    S1D = ['S1D', 'raw']
    S1D_SKYSUB = ['S1D', 'S1D_SKYSUB']
    S2D = ['S2D', 'raw']
    S2D_SKYSUB = ['S2D', 'S2D_SKYSUB']
    S2D_BLAZE = ['S2D', 'S2D_BLAZE']
    CCF = ['CCF', 'raw']
    CCF_SKYSUB = ['CCF', 'CCF_SKYSUB']
    
#%% Fiber
class _Fiber(Enum):
    """
    Viable fibers to load.
    
    Raises KeyError if wrong fiber has been used
    """
    A = ['A', 'Fiber_A']
    B = ['B', 'Fiber_B']

#%%
def _numbering_nights(spectrum_list:sp.SpectrumList):
    """
    Index spectrum list with Night indices (start = 1), Spectrum indices within single night (start = 1) and spectra indices (start = 1)

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list loaded from all instruments and nights as given by load_all() function.
    """
    if len(spectrum_list) == 0:
        logger.warning('No spectra loaded, cannot index:')
        logger.warning('    Returning')
        return
    
    number_night = 1
    number_spec = 1
    last_night =  spectrum_list[0].meta['Night']
    
    for ind, item in enumerate(spectrum_list):
        item.meta['Spec_num'] = ind + 1
        
        if item.meta['Night'] == last_night:
            item.meta['Night_num'] = number_night
            item.meta['Night_spec_num'] = number_spec
            number_spec += 1
        else:
            number_night += 1
            number_spec = 1
            last_night = item.meta['Night']
            item.meta['Night_num'] = number_night
            item.meta['Night_spec_num'] = number_spec
    return

#%% correct_dispersion_S2D
def _correct_dispersion_S2D(spectral_axis: sp.spectra.spectral_axis.SpectralAxis,
                           flux: astropy.units.quantity.Quantity,
                           uncertainty: astropy.nddata.StdDevUncertainty) -> astropy.units.quantity.Quantity:
    """
    Corrects for the dispersion effect on the S2D spectra.
    
    TODO: Add uncertainty propagation.

    Parameters
    ----------
    spectral_axis : sp.spectra.spectral_axis.SpectralAxis
        Spectral axis (2-dimensional) of the spectrum
    flux : astropy.units.quantity.Quantity
        Flux (2-dimensional) of the spectrum
    uncertainty : 
        Uncertainty (2-dimensional) of the spectrum flux

    Returns
    -------
    astropy.units.quantity.Quantity
        Corrected flux array
    """
    # Get a difference between adjacent pixels, and add one more element to have correct shape for division
    difference = np.append(np.diff(spectral_axis),
                           np.diff(spectral_axis)[:,-1]).reshape(
                               spectral_axis.shape
                               )
    # Need to divide by a mean to have the same SNR in the spectrum.
    difference /= difference.mean(axis = 1)[:,np.newaxis]
    
    flux = flux / difference
    return flux
#%% find_UT
def _find_UT(header: np.ndarray) -> int:
    '''
    Convenience function returning which UT was used for current spectrum of ESPRESSO. Does not work for 4-UT mode yet.
    #TODO Add 4-UT

    Parameters
    ----------
    header : np.ndarray
        header of the spectrum.

    Returns
    -------
    int
        Which UT was used.

    '''
    return int(header['TELESCOP'][-1:])
#%% find_nb_orders
def _find_nb_orders(header: np.ndarray) -> int:
    '''
    Convenience function returning number of orders for current spectrum (HARPS, ESPRESSO), new DRS pipeline

    Parameters
    ----------
    header : Array
        header of the spectrum.

    Returns
    -------
    int
        number of orders in spectra.

    '''
    if header['INSTRUME'] == 'ESPRESSO':
         nb_orders = 170
    elif header['INSTRUME'] == 'HARPS':
        nb_orders = 71
    elif header['INSTRUME'] == 'NIRPS':
        nb_orders = 70  
    else:
        raise ValueError("Instrument is not supported")
    return nb_orders

#%% Check DRS version
def _check_DRS_version(DRS:str):
    """
    Check version of DRS used.

    Parameters
    ----------
    DRS : str
        Version of DRS used using the Pipe ID header keyword 'HIERARCH ESO PRO REC1 PIPE ID'.
    """
    
    if (DRS != 'espdr/3.0.0'):
        logger.warning('The current version of DRS has not been tested. Please check for errors:')
        logger.info('    Current version:'+DRS)
        logger.info('    Tested versions:')
        logger.info('        ESPRESSO (+HARPS/NIRPS): 3.0.0')
    else:
        logger.debug('    Test of DRS version passed.')
    return

#%% Create basic meta parameters
def _basic_meta_parameters() -> dict:
    """
    Provide full set of meta parameters (with some undefined).

    Returns
    -------
    dict
        Meta dictionary with various information.
    """
    meta = {
        'normalization': False,
        'blaze_corrected': False,
        'telluric_corrected': False,
        'RM_velocity' : 'undefined',
        'Phase': 'undefined',
        'RM_corrected': False,
        'RF_Earth': False,
        'RF_Barycenter': False,
        'RF_Star': False,
        'RF_Planet': False,
        'RF': 'undefined',
        'v_sys_corrected':False,
        'v_star_corrected':False,
        'v_planet_corrected':False,
        'Night': 'undefined',
        'Night_num': 'undefined',
        'Night_spec_num': 'undefined',
        'velocity_planet': 'undefined',
        'velocity_star': 'undefined',
        'BJD': 'undefined',
        'velocity_BERV': 'undefined',
        'velocity_system': 'undefined',
        'Seeing': 'undefined',
        'Airmass': 'undefined',
        'S_N': 'undefined',
        'Exptime': 'undefined',
        'Average_S_N': 'undefined',
        'vacuum': 'undefined',
        'air': 'undefined',
        'instrument': 'undefined'
    }
    return meta

#%% Find units based on keyword in header
def _find_unit(header_TUNITx_value: str) -> u.Unit:
    """
    Find unit based on a TUNITx keyword in the header of the data table.

    Parameters
    ----------
    header_TUNITx_value : str
        String of the value in the header under the TUNITx keyword.

    Returns
    -------
    u.Unit
        Resulting unit.

    Raises
    ------
    ValueError
        In case unit is not recognized, raise a ValueError
    """
    if header_TUNITx_value == 'angstrom':
        result = u.AA
    elif header_TUNITx_value == 'e-':
        result = u.ct
    else:
        raise ValueError('Type of unit not recognized.')
    return result
#%% Load header parameters
def _load_meta_from_header(header: fits.header.Header) -> dict:
    """
    Load meta parameters from header.

    Parameters
    ----------
    header : fits.header.Header
        Main header of the fits file

    Returns
    -------
    dict
        The meta parameters to add to the spectrum.
    """
    
    nb_orders = _find_nb_orders(header)
    sn = np.zeros(nb_orders+1)
    
    # SNR is saved as an array, with average SNR saved as zero-th element
    for ii in range(nb_orders):
        sn[ii+1] = header['HIERARCH ESO QC ORDER%i SNR' %(ii+1)]
    sn[0] = np.mean(sn[1:])
    
    # As ESPRESSO header keywords change based on UT used
    if header['INSTRUME'] == 'ESPRESSO':
        UT = str(_find_UT(header))
    else:
        UT = ''
    
    # Create dictionary
    meta = {
            'BJD':header['HIERARCH ESO QC BJD'] * u.day, # BJD
            'velocity_BERV':header['HIERARCH ESO QC BERV'] * 1000 * u.m / u.s , # BERV velocity
            'Airmass':np.mean([header['HIERARCH ESO TEL%s AIRM START'%(UT)],
                               header['HIERARCH ESO TEL%s AIRM END'%(UT)]]), # Airmass
            'Seeing':np.mean([header['HIERARCH ESO TEL%s AMBI FWHM START'%(UT)],
                              header['HIERARCH ESO TEL%s AMBI FWHM END'%(UT)]]), # Seeing
            'S_N_all':sn, # Signal-to-noise,
            'Average_S_N':sn[0],
            'Exptime':header['EXPTIME'] * u.s, #Exposure time,
            'instrument': header['INSTRUME'],
            'Night': header['DATE-OBS'][:10],
            }
    
    return meta
#%% 
def _mask_flux_array(flux: u.Quantity) -> np.ndarray:
    """
    Creates a mask of flux array assuming finite and positive (non-zero) values.

    Parameters
    ----------
    flux : u.Quantity
        Flux on which to create mask.

    Returns
    -------
    np.ndarray
        Mask for the flux array. True if pixel should be masked, False if not, as defined by the sp.Spectrum1D documentation.
    """
    mask = np.logical_or(
        np.isnan(flux),
        flux <= 0
        )
    
    return mask

#%% Testing function
if __name__ == '__main__':
    logger.info('Testing setup for rats.eso module.')
    main_directory = '/media/chamaeleontis/Observatory_main/Analysis_dataset/rats_test'
    
    for spectra_member in _SpectraFormat:
        if 'CCF' in spectra_member.name:
            logger.critical('CCF format are not implemented yet. After implementing remove this line and test.')
            continue
        
        for fiber_member in _Fiber:
            if ((fiber_member.name == 'B') and 
                ('SKYSUB' in spectra_member.name)):
                continue
            
            logger.info('Trying to load all data from folder: ' + main_directory)
            logger.info('    Currently testing format:' + spectra_member.name + ' and fiber: ' + fiber_member.name)
            spectra_list = load_all(
                main_directory= main_directory,
                spectra_format= spectra_member.name,
                fiber= fiber_member.name,
                instrument_list= None,
                force_load= False,
                force_skip= False,
                pkl_name= spectra_member.name + fiber_member.name + '.pkl'
                )
            logger.info('Loaded '+ str(len(spectra_list)) + ' number of spectra')
            logger.info('    Succesfully loaded format:' + spectra_member.name + ' and fiber: ' + fiber_member.name)
    logger.info('Test succesful. Check logs for issues.')