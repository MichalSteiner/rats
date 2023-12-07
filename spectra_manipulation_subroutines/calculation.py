from itertools import pairwise
import specutils as sp
import numpy as np
from astropy.nddata import StdDevUncertainty
#%% Masking of non-photon noise dominated pixels
def _mask_non_photon_noise_dominated_pixels_in_spectrum(spectrum: sp.Spectrum1D,
                                                        threshold: float= 0.5) -> sp.Spectrum1D:
    """
    Give a spectrum an updated mask to mask the non-photon noise dominated pixels.

    Parameters
    ----------
    spectrum : sp.Spectrum1D
        Spectrum for which to find the mask.
    threshold : float, optional
        Threshold by which to mask, by default 0.5. This masks pixels that have 50% structure of non-photon noise. The factor is a percentage of how much "domination" should the non-photon noise sources have. For example, to get a 10% mask (pixels with non-photon noise on the order of magnitude as the photon noise), threshold = 0.1.
        
    Returns
    -------
    spectrum : sp.Spectrum1D
        Spectrum object with updated mask attribute.
    """
    
    photon_noise = np.sqrt(spectrum.flux.value)
    uncertainty = spectrum.uncertainty.array
    
    # Filter based on a threshold
    mask = np.where(uncertainty * (1 - threshold) > photon_noise)
    
    spectrum.mask[mask] = True
    return spectrum
#%% Execute masks in spectrum
def _execute_mask_in_spectrum(spectrum: sp.Spectrum1D | sp.SpectrumCollection) -> sp.Spectrum1D | sp.SpectrumCollection:
    """
    Executes mask assigned in spectrum by filling with nans. Mask is applied where mask == True

    Parameters
    ----------
    spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Spectrum for which to execute the mask.

    Returns
    -------
    new_spectrum : sp.Spectrum1D | sp.SpectrumCollection
        New spectrum with masked pixels based on mask.
    """
    
    # TESTME
    mask = spectrum.mask
    
    masked_flux = spectrum.flux
    masked_flux[mask] = np.nan
    
    masked_uncertainty = spectrum.uncertainty
    masked_uncertainty[mask] = np.nan

    if type(spectrum) == sp.Spectrum1D:
        new_spectrum = sp.Spectrum1D(
            spectral_axis= spectrum.spectral_axis,
            flux= masked_flux,
            uncertainty= masked_uncertainty,
            mask= spectrum.mask,
            meta= spectrum.meta,
            )
    elif type(spectrum) == sp.SpectrumCollection:
        new_spectrum = sp.SpectrumCollection(
            spectral_axis= spectrum.spectral_axis,
            flux= masked_flux,
            uncertainty= masked_uncertainty,
            mask= spectrum.mask,
            meta= spectrum.meta,
            )
    
    return new_spectrum
#%% error_bin
def _error_bin(array: np.ndarray | list) -> float:
    """
    Calculation of error within single bin.

    Parameters
    ----------
    array : np.ndarray | list
        Array of values within the bin.

    Returns
    -------
    value : float
        Error of the bin
    """
    # Change list to array
    if isinstance(array,list):
        array = np.asarray(array)
    # For arrays longer than 1 value
    if len(array) != 1:
        value = np.sqrt(np.sum(array**2)/(len(array))**2)
    else:
        value = array[0]
        
    return value

#%% Get spectrum type
def get_spectrum_type(key: str, value: str | None) -> str:
    """
    Supplementary function giving spec_type value for each key,value used
    Used for labeling plots
    Input:
        key ; key of meta dictionary
        value ; value of meta dictionary
    Output:
        spec_type ; Type of master spectrum (eg. 'out-of-Transit master')
    """
    # Type of master based on in/out of transit
    if (key == 'Transit') or (key == 'Transit_full') or (key == 'Preingress') or (key == 'Postegress'):
        if value == False:
            spectrum_type = 'Out-of-transit'
        else:
            spectrum_type = 'In-transit (transmission)'
            
    # Type of master based on before/after telluric correction
    if key == 'telluric_corrected':
        if value == True:
            spectrum_type = 'After-telluric-correction'
        else:
            spectrum_type = 'Before-telluric-correction'
    # Set None master type (for debugging)
    if key == None:
        spectrum_type = 'None'
    return spectrum_type
#%% Check order of nights
def _check_night_ordering(spectrum_list: sp.SpectrumList) -> None:
    """
    Checks whether the nights are ordered in spectrum list.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum_list to check the order in

    Raises
    ------
    ValueError
        In case the nights are not ordered, raises a ValueError.
    """
    for first_spectrum, second_spectrum in pairwise(spectrum_list):
        if first_spectrum.meta['Night_num'] > second_spectrum.meta['Night_num']:
            logger.critical('The spectrum list night indices are not ordered.')
            logger.critical('This is assumed for several functions.')
            raise ValueError('Spectrum list meta parameter "Night_num" is not ordered')
    return None

#%% cosmic_correction_night
def cosmic_correction_night(sublist):
    """
    Correction for cosmics in given night

    Parameters
    ----------
    sublist : sp.SpectrumList
        Sublist of given night to correct for.

    Returns
    -------
    new_spec_list : sp.SpectrumList
        Corrected spectrum sublist.

    """
    new_spec_list = sp.SpectrumList()
    
    flux_all = np.zeros((len(sublist),len(sublist[0].spectral_axis)))
    for ind,item in enumerate(sublist):
        flux_all[ind,:] = item.flux
    median = np.median(flux_all,axis=0)
    
    for ii, item in enumerate(sublist):
        ind = np.where((item.flux - median) > item.uncertainty.array * 5)
        flux = item.flux
        flux[ind] = median[ind]
        
        new_spec_list.append(
            sp.Spectrum1D(
                spectral_axis = item.spectral_axis ,
                flux = flux * item.flux.unit,
                uncertainty = item.uncertainty, # Seems like a weird step
                mask =  item.mask.copy(),
                meta = item.meta.copy(),
                wcs = item.wcs,
                )
            )
    return new_spec_list

#%% Calculate master (average) spectrum
def calculate_master(spectrum_list,
               spectrum_type = '',
               night = '',
               num_night = '',
               rf = '',
               sn_type='quadratic'
               ):
    """
    Calculates master spectrum of spectrum list
    Input:
        spec_list ; sp.SpectrumList
        spec_type ; Type of resulting master spectrum (for automatic labels)
        night ; number of night/ 'all' nights (for automatic labels)
        rf ; rest frame of spectrum (for automatic labels)
        sn_type ; type of weighting
            possible options are:
                'S_N' ; linear S_N weight
                'quadratic' ; quadratic S_N weight
                'quadratic_combined' ; quadratic S_N and light curve flux
                'None' ; fill with 1
    Output:
        master ; sp.Spectrum1D - master spectrum
    """
    # Unit of flux
    unit_flux = spectrum_list[0].flux.unit
    # Allocate wavelength, flux and flux_err arrays
    spectral_axis = spectrum_list[0].spectral_axis
    flux = np.zeros(spectrum_list[0].spectral_axis.shape)
    flux_err = np.zeros(spectrum_list[0].spectral_axis.shape)
    # Allocate weighting
    # Since some pixels might be masked, its necessary to weight by pixel
    weights_total = np.zeros(spectrum_list[0].spectral_axis.shape)
    
    # For cycle through spec_list
    for item in spectrum_list:
        mask_flux = np.isnan(item.flux) # Masking NaNs
        mask_err = np.isnan(item.uncertainty.array) # Masking NaNs
        mask = mask_flux + mask_err # Getting combined mask (zero is for correct pixels)
        mask != 0
        # Taking flux and error from spectrum
        tmp_flux = item.flux.value
        tmp_err = item.uncertainty.array
        # Assigning weights according to type of weighting
        if sn_type == 'S_N':
            weights = [item.meta['Average_S_N']] * ~mask 
        elif sn_type == 'quadratic':
            weights = (np.asarray([item.meta['Average_S_N']])**2) * ~mask
        elif sn_type =='quadratic_error':
            weights = ((np.asarray([item.meta['Average_S_N']])**2*np.asarray([item.uncertainty.array])**2) * ~mask).flatten()
        elif sn_type == 'quadratic_combined':
            weights = (np.asarray([item.meta['Average_S_N']])**2 +\
                       np.asarray(item.meta['delta']**2)) * ~mask
        elif sn_type == 'None':
            weights = [1] * len(flux)
        elif sn_type == 'quadratic_error':
            weights = item.uncertainty.array**(-2) * ~mask
        # Erasing NaN values with 0
        tmp_flux = np.where(mask == False,tmp_flux,0)
        tmp_err = np.where(mask == False,tmp_err,0)
        weights = np.where(mask == False,weights,0)
        # Suming flux, flux_err and weights for averaging
        flux += tmp_flux * weights
        flux_err += (tmp_err*weights)**2
        weights_total += weights
    # Averaging flux and flux_err
    flux = flux / weights_total
    flux_err = np.sqrt(flux_err) / weights_total
    # Creation of master spectrum
    master = sp.Spectrum1D(
        flux = flux * unit_flux,
        spectral_axis = spectral_axis,
        uncertainty =  astropy.nddata.StdDevUncertainty(flux_err),
        mask = np.isnan(flux),
        wcs = add_spam_wcs(),
        )
    # Updating type of master
    master.meta = {'type':spectrum_type,
                   'night':night,
                   'Night_num':num_night,
                   'RF': rf
        }
    return master

def calculate_master_test(
    spectrum_list: sp.SpectrumList,
    spectrum_type: str = '',
    night: str = '',
    num_night: str = '',
    rf: str = '',
    sn_type: str | None = None,
    method: str = 'addition',
    ) -> sp.Spectrum1D:
    
    
    
    master = _empty_spectrum_like(spectrum_list[0])
    
    for spectrum in spectrum_list:
        _execute_mask_in_spectrum(spectrum)
        weights = _gain_weights(spectrum= spectrum,
                                sn_type= sn_type)
        master = master.add(spectrum.multiply(weights))
    
    if method == 'average' or method == 'mean':
        master = master.divide(len(spectrum_list) * u.dimensionless_unscaled)
    return master

def _gain_weights(spectrum: sp.Spectrum1D,
                  sn_type: str | None = None) -> sp.Spectrum1D:
    """
    Generate weight spectrum for the master calculation.

    Parameters
    ----------
    spectrum : sp.Spectrum1D
        Spectrum for which to generate the weights array.
    sn_type : str | None
        Type of weighting, by default None.
        Options are:
            None:
                No weights assumed
            'Average_S_N':
                Weights are scaled by average SNR.
            'quadratic':
                Weights are scaled by average SNR**2
            'quadratic_error':
                Weights are scaled by flux_error ** 2.
            'quadratic_combined':
                Weights are scaled by flux_error ** 2 * delta(phase) ** 2
                Delta is the time dependent transit depth.
            
    Returns
    -------
    weights : sp.Spectrum1D
        Weights for the spectrum.
    """
    weights = _empty_spectrum_like(spectrum= spectrum,
                                   constant= 1)
    
    match sn_type:
        case None:
            pass
        case 'Average_S_N':
            weights = weights.multiply(spectrum.meta['Average_S_N'])
        case 'quadratic':
            weights = weights.multiply(spectrum.meta['Average_S_N'] ** 2)
        case 'quadratic_error':
            # TODO Check if this doesn't break for uncertainty = 0
            weights = weights.multiply(spectrum.uncertainty.array**2)
        case 'quadratic_combined':
            weights = weights.multiply(spectrum.uncertainty.array**2)
            weights = weights.multiply(spectrum.meta['delta']**2)
    return weights



def _empty_spectrum_like(spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                         constant: float | int = 0,
                         ) -> sp.Spectrum1D | sp.SpectrumCollection:
    """
    Create an empty spectrum of the same shape as the input. The type of spectrum is inherited by input (either sp.Spectrum1D or sp.SpectrumCollection)

    Parameters
    ----------
    spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Input spectrum of the desired shape.
    constant : float | int
        Constant which is assigned to the flux values, by default 0.

    Returns
    -------
    new_spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Empty spectrum with the desired shape.

    Raises
    ------
    TypeError
        If input spectrum is not sp.Spectrum1D or sp.SpectrumCollection object, raise TypeError.
    """
    
    if type(spectrum) == sp.Spectrum1D:
        new_spectrum = sp.Spectrum1D(
            spectral_axis= spectrum.spectral_axis,
            flux= np.zeros_like(spectrum.flux) + constant,
            uncertainty= StdDevUncertainty(np.zeros_like(spectrum.flux))
            )
    elif type(spectrum) == sp.SpectrumCollection:
        new_spectrum = sp.SpectrumCollection(
            spectral_axis= spectrum.spectral_axis,
            flux= np.zeros_like(spectrum.flux) + constant,
            uncertainty= StdDevUncertainty(np.zeros_like(spectrum.flux))
            )
    else:
        logger.critical('Spectrum type not found.')
        raise TypeError('Spectrum type is not valid. Only sp.Spectrum1D and sp.SpectrumCollection can be requested.')
    return new_spectrum
