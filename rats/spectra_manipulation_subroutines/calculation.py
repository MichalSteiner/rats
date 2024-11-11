from itertools import pairwise
import specutils as sp
import numpy as np
import astropy
from astropy.nddata import StdDevUncertainty, NDDataRef, NDDataArray
import astropy.units as u
from rats.utilities import default_logger_format
import logging

logger = logging.getLogger(__name__)
logger = default_logger_format(logger)


#%% Masking of flux below threshold in spectrum
def _mask_flux_below_threshold(spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                               threshold: float = 100,
                               polynomial_order: int = 6):
    
    match type(spectrum):
        case sp.Spectrum1D:
            spectral_axis = spectrum.spectral_axis.value
            flux = spectrum.flux.value
            weights = 1/(spectrum.uncertainty.array**2)
            mask_flux = ~np.isfinite(flux)
            mask_err = ~np.isfinite(weights)
            mask = np.ma.mask_or(mask_flux, mask_err)
            
            polynomial_fit = np.polyfit(spectral_axis, flux, polynomial_order, w= weights)
            flux_polynomial = np.poly1d(polynomial_fit)(spectral_axis)
            
            mask_flux_in_order = np.where(flux_polynomial > threshold)
            spectrum_order.mask[mask_flux_in_order] = np.nan
        
        case sp.SpectrumCollection:
            for spectrum_order in spectrum:
                spectral_axis = spectrum_order.spectral_axis.value
                flux = spectrum_order.flux.value
                weights = 1#/(spectrum_order.uncertainty.array**2)
                mask_flux = ~np.isfinite(flux)
                mask_err = ~np.isfinite(weights)
                mask = np.logical_or(mask_flux, mask_err)
                
                polynomial_fit = np.polyfit(spectral_axis[~mask], flux[~mask], polynomial_order,
                                            #w= weights[~mask]
                                            )
                flux_polynomial = np.poly1d(polynomial_fit)(spectral_axis)
                
                mask_flux_in_order = np.where(flux_polynomial < threshold)
                spectrum_order.mask[mask_flux_in_order] = True
    
    # fig,ax = plt.subplots(1)
    # ax.plot(spectral_axis, flux, color = 'darkblue', alpha=0.6)
    # ax.plot(spectral_axis[~spectrum_order.mask],flux[~spectrum_order.mask], color = 'darkgreen', alpha=0.6)
    # ax.plot(spectral_axis[~mask], np.poly1d(polynomial_fit)(spectral_axis[~mask]), color = 'darkred')
    # ax.axhline(100,ls='--', color='black')
    
    return spectrum 

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
    
    masked_uncertainty = spectrum.uncertainty.array
    masked_uncertainty[mask] = np.nan
    masked_uncertainty = StdDevUncertainty(masked_uncertainty)

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
        value = np.sqrt(np.nansum(array**2)/(sum(np.isfinite(array)))**2)
    else:
        value = array[0]
        
    return value

#%% Get spectrum type
def _get_spectrum_type(key: str | None, value: str | None) -> str:
    """
    Supplementary function giving spec_type value for each key,value used.
    Used for labeling plots.

    Parameters
    ----------
    key : str | None
        Key of meta dictionary.
    value : str | None
        Value of meta dictionary.

    Returns
    -------
    spectrum_type : str
        Type of master spectrum (e.g., 'Out-of-Transit master').

    Notes
    -----
    The function determines the type of master spectrum based on the provided key and value.
    - If the key is 'Transit_partial' or 'Transit_full':
        - If the value is False, the spectrum type is 'Out-of-transit'.
        - If the value is True, the spectrum type is 'In-transit (transmission)'.
    - If the key is 'Preingress', the spectrum type is 'Pre-ingress'.
    - If the key is 'Postegress', the spectrum type is 'Post-egress'.
    - If the key is 'telluric_corrected':
        - If the value is True, the spectrum type is 'After-telluric-correction'.
        - If the value is False, the spectrum type is 'Before-telluric-correction'.
    - If the key is None, the spectrum type is 'None' (for debugging).
    """
    # TODO rewrite so it gives always a type
    # Type of master based on in/out of transit
    if (key == 'Transit_partial') or (key == 'Transit_full'):
        if value == False:
            spectrum_type = 'Out-of-transit'
        else:
            spectrum_type = 'In-transit (transmission)'
    if (key == 'Preingress'):
        spectrum_type = 'Pre-ingress'
    if (key == 'Postegress'):
        spectrum_type = 'Post-egress'
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
def _cosmic_correction_night(sublist):
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
    flux_all = NDDataArray([item.flux for item in sublist], unit= sublist[0].flux.unit)
    median = np.nanmedian(flux_all, axis=0)
    
    for ii, item in enumerate(sublist):
        ind = abs(item.flux.value - median) > item.uncertainty.array * 5
        flux = item.flux.value
        flux[ind] = np.nan
        
        new_spec_list.append(
            sp.Spectrum1D(
                spectral_axis = item.spectral_axis ,
                flux = flux * item.flux.unit,
                uncertainty = item.uncertainty, # Seems like a weird step
                mask =  item.mask.copy(),
                meta = item.meta.copy(),
                )
            )
    return new_spec_list
#%% Calculate master from filtered sublist
def _calculate_master(
    spectrum_list: sp.SpectrumList,
    spectrum_type: str = '',
    night: str = '',
    num_night: str = '',
    rf: str = '',
    sn_type: str | None = None,
    method: str = 'average',
    ) -> sp.Spectrum1D | sp.SpectrumCollection:
    """
    Calculates a master from filtered spectrum list. The method of master calculation is passed as a keyword.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list from which to calculate the master. The spectrum list is already filtered to only relevant spectra.
    spectrum_type : str, optional
        Which type of master is being calculated, by default ''
    night : str, optional
        For which night is the master calculated, by default ''
    num_night : str, optional
        What is the number of night for which the master has been calculated, by default ''
    rf : str, optional
        Rest frame in which the master is calculated, by default ''
    sn_type : str | None, optional
        Weighting option, by default None.
    method : str, optional
        Method for calculation of the master, by default 'average'. Options are:
            'average' or 'mean'
                Classical average master, with weights defined by sn_type
            'addition'
                Master created by adding all relevant spectra together. No weighting is assumed.
            'median'
                Master created by calculation of median spectrum. No weighting is assumed.
        

    Returns
    -------
    master : sp.Spectrum1D | sp.SpectrumCollection
        Master spectrum from the spectrum_list
    """
    
    # FIXME Change the implementation to work with NDDataArray instead
    # CLEANME This can be simplified. 
    master = _empty_spectrum_like(spectrum_list[0],
                                  unit = spectrum_list[0].flux.unit)
    weight_sum = _empty_spectrum_like(spectrum_list[0],
                                      unit = None)
    spectrum_format = type(master)
    
    
    for spectrum in spectrum_list:
        spectrum = _remove_NaNs_with_constant(spectrum)
    # CLEANME - this can be simplified
    
    match (method, spectrum_format):
        case 'addition', _:
            master = _calculate_addition_master(
                spectrum_list= spectrum_list,
                master= master,
                spectrum_type= spectrum_type,
                night= night,
                num_night= num_night,
                rf= rf,
                sn_type= sn_type
                )
        case 'median', _:
            #TODO
            logger.critical('Median master is not implemented yet! Returning')
            raise NotImplementedError
            return
        case 'mean' | 'average', sp.Spectrum1D:
            master = _calculate_mean_master(
                spectrum_list= spectrum_list,
                master= master,
                weight_sum= weight_sum,
                spectrum_type= spectrum_type,
                night= night,
                num_night= num_night,
                rf= rf,
                sn_type= sn_type
                )
        case 'mean' | 'average', sp.SpectrumCollection:
            master = _calculate_mean_master_collection(
                spectrum_list= spectrum_list,
                master= master,
                weight_sum= weight_sum,
                spectrum_type= spectrum_type,
                night= night,
                num_night= num_night,
                rf= rf,
                sn_type= sn_type
                )
        case _, _:
            raise ValueError('Requested method is not valid.')
        
    return master

def _test_equality_of_spectral_axis(spectrum_list: sp.SpectrumList) -> bool:
    """
    Tests the equality of spectral axes in spectrum list. This is necessary for any mathematical operation on the spectrum list, like building master.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list on which to test the condition.

    Returns
    -------
    bool
        _description_
    """
    return np.asarray([(item1.spectral_axis ==
                    item2.spectral_axis
                    ).all() for item1, item2 in pairwise(spectrum_list)]
                    ).all()
    
    

def _calculate_addition_master(
    spectrum_list: sp.SpectrumList,
    master: sp.Spectrum1D | sp.SpectrumCollection,
    spectrum_type: str = '',
    night: str = '',
    num_night: str = '',
    rf: str = '',
    sn_type: str | None = None,
    ) -> sp.Spectrum1D | sp.SpectrumCollection:
    """
    Calculates master by addition.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list to calculate master from.
    master : sp.Spectrum1D | sp.SpectrumCollection
        Base master spectrum empty-object prepared before-hand to hold the result
    spectrum_type : str, optional
        Which type of master is being calculated, by default ''
    night : str, optional
        For which night is the master calculated, by default ''
    num_night : str, optional
        What is the number of night for which the master has been calculated, by default ''
    rf : str, optional
        Rest frame in which the master is calculated, by default ''
    sn_type : str | None, optional
        Weighting option, by default None.

    Returns
    -------
    master : sp.Spectrum1D | sp.SpectrumCollection
        Calculated master. Format is the same as input master.
    """
    # Test equality of wavelength grid
    assert _test_equality_of_spectral_axis(spectrum_list), 'Spectral axis is not same for the spectrum list'
    spectral_axis = spectrum_list[0].spectral_axis
    
    weights = _gain_weights_list(spectrum_list, sn_type=sn_type)
    
    flux_array = astropy.nddata.NDDataArray(
        data = [item.flux for item in spectrum_list],
        uncertainty = astropy.nddata.StdDevUncertainty([(item.uncertainty.array) for item in spectrum_list]),
        unit= spectrum_list[0].unit
        )
    flux_array = _remove_NaNs_with_constant_NDData(flux_array,
                                                   constant=0)
    weighted_flux = flux_array.multiply(weights)
    
    for item in weighted_flux:
        master = master.add(item)

    
    master.meta = {
        'rf': rf,
        'num_night': num_night,
        'night': night,
        'sn_type': sn_type,
        'method': 'addition',
        'type': spectrum_type,
        }
    
    return master

def _calculate_mean_master(spectrum_list: sp.SpectrumList,
                           master: sp.Spectrum1D,
                           weight_sum: sp.Spectrum1D,
                           spectrum_type: str = '',
                           night: str = '',
                           num_night: str = '',
                           rf: str = '',
                           sn_type: str | None = None,
                           ) -> sp.Spectrum1D:
    """
    Calculates a mean master from a spectrum list of sp.Spectrum1D.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list which holds all the Spectrum1D spectra.
    master : sp.Spectrum1D
        Empty master spectrum reference
    weight_sum : sp.Spectrum1D
        Empty weight sum spectrum reference
    spectrum_type : str, optional
        Which type of master is being calculated, by default ''
    night : str, optional
        For which night is the master calculated, by default ''
    num_night : str, optional
        What is the number of night for which the master has been calculated, by default ''
    rf : str, optional
        Rest frame in which the master is calculated, by default ''
    sn_type : str | None, optional
        Weighting option, by default None.

    Returns
    -------
    master : sp.Spectrum1D
        Average master spectrum.
    """
    
    for spectrum in spectrum_list:
        weights = _gain_weights(spectrum= spectrum,
                                sn_type= sn_type)
        
        # Handles masked and NaNs values in the spectrum
        weights.mask = np.logical_or(np.isnan(spectrum.flux),
                             np.isnan(spectrum.uncertainty.array)
                             )
        # TODO: This is weird....
        weights = _execute_mask_in_spectrum(weights)
        weights = _remove_NaNs_with_constant(weights, constant= 0)
        
        spectrum = _execute_mask_in_spectrum(spectrum)
        spectrum = _remove_NaNs_with_constant(spectrum, constant= 0)

        master = master.add(spectrum.multiply(weights))
        weight_sum = weight_sum.add(weights)
        
    master = master.divide(weight_sum)
    
    master.meta = {
        'rf': rf,
        'num_night': num_night,
        'night': night,
        'sn_type': sn_type,
        'method': 'average',
        'type': spectrum_type,
        }
    return master
#%% 
def _calculate_mean_master_collection(
    spectrum_list: sp.SpectrumList,
    master: sp.SpectrumCollection,
    weight_sum: sp.SpectrumCollection,
    spectrum_type: str = '',
    night: str = '',
    num_night: str = '',
    rf: str = '',
    sn_type: str | None = None
    ) -> sp.SpectrumCollection:
    #DOCUMENTME
    master_flux = NDDataRef(
            data= master.flux,
            uncertainty= master.uncertainty
            )
    weight_sum_flux = NDDataRef(
        data= weight_sum.flux
        )
    
    for spectrum in spectrum_list:
        flux_2D = NDDataRef(
            data= spectrum.flux,
            uncertainty= spectrum.uncertainty)
        
        weights_flux = _gain_weights(spectrum= spectrum,
                                     sn_type= sn_type)
        master_flux = master_flux.add(flux_2D.multiply(weights_flux))
        weight_sum_flux = weight_sum_flux.add(weights_flux)

    master_flux = master_flux.divide(weight_sum_flux)
    meta = {
        'rf': rf,
        'num_night': num_night,
        'night': night,
        'sn_type': sn_type,
        'method': 'average',
        'type': spectrum_type,
        }
    master = sp.SpectrumCollection(
        spectral_axis= master.spectral_axis,
        flux = master_flux.data * master_flux.unit,
        uncertainty= master_flux.uncertainty,
        mask = np.isnan(master_flux.data),
        meta= meta
        )
    
    return master

def _remove_NaNs_with_constant_NDData(input_data: astropy.nddata.NDDataArray,
                                      constant: float = 0) -> astropy.nddata.NDDataArray:
    """
    Remove NaNs with a constant value from NDDataArray, by default 0. This is useful to handle NaNs propagation with master calculation (e.g., one bad pixel in single spectrum would propagate the NaN to a master spectrum, without this handling).

    Parameters
    ----------
    input_data : astropy.nddata.NDDataArray
        Input data to remove NaNs from.
    constant : float, optional
        Constant value to fill the NDData for, by default 0. Sensible values are 0's and 1's.

    Returns
    -------
    astropy.nddata.NDDataArray
        _description_
    """
    new_data = astropy.nddata.NDDataArray(
        data = np.where(np.isfinite(input_data.data),
                           input_data.data,
                           constant) * input_data.unit,
        uncertainty = astropy.nddata.StdDevUncertainty(
            np.where(
                np.isfinite(input_data.uncertainty.array),
                input_data.uncertainty.array,
                0))
        )
    return new_data

#%%
def _remove_NaNs_with_constant(spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                               constant: float = 0) -> sp.Spectrum1D | sp.SpectrumCollection:
    """
    Remove the NaNs values from the spectrum and replaces them with constant. Used for calculation of several types of masters, where the constant is defined such that the values do not interact with the goal (e.g, constant = 0 when adding).

    Parameters
    ----------
    spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Spectrum for which to remove NaNs.
    constant : float, optional
        Constant which to use to replace the NaNs value, by default 0

    Returns
    -------
    sp.Spectrum1D | sp.SpectrumCollection
        Spectrum with replaced flux and uncertainty values where NaNs were present.
    """
    
    if type(spectrum) == sp.Spectrum1D:
        new_spectrum = sp.Spectrum1D(
            spectral_axis= spectrum.spectral_axis,
            flux= np.where(np.isfinite(spectrum.flux),
                           spectrum.flux,
                           constant * spectrum.flux.unit),
            uncertainty = StdDevUncertainty(np.where(np.isfinite(spectrum.uncertainty.array),
                                                     spectrum.uncertainty.array,
                                                     0)),
            meta = spectrum.meta,
            mask = spectrum.mask,
            )
            
    elif type(spectrum) == sp.SpectrumCollection:
        new_spectrum = sp.SpectrumCollection(
            spectral_axis= spectrum.spectral_axis,
            flux= np.where(np.isfinite(spectrum.flux),
                           spectrum.flux,
                           constant * spectrum.flux.unit
                           ),
            uncertainty = StdDevUncertainty(np.where(np.isfinite(spectrum.uncertainty.array),
                                                     spectrum.uncertainty.array,
                                                     0)),
            meta = spectrum.meta,
            mask = spectrum.mask,
            )
    return new_spectrum

def _gain_weights(spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                  sn_type: str | None = None) -> sp.Spectrum1D | astropy.nddata.NDDataRef:
    """
    Generate weight spectrum for the master calculation.

    Parameters
    ----------
    spectrum : sp.Spectrum1D | sp.SpectrumCollection
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
    weights : sp.Spectrum1D | astropy.nddata.NDDataRef
        Weights for the spectrum. If spectrum collection is used, a NDDataRef object is passed back, as the calculation is done on this object instead of the collections themselves. This is to avoid slow for-looping over orders.
    """
    # Change this to exclude NaNs
    if type(spectrum) == sp.SpectrumCollection:
        weights = NDDataRef(
            data = np.ones_like(spectrum.flux.value)
        )
    elif type(spectrum) == sp.Spectrum1D:
        weights = _empty_spectrum_like(spectrum= spectrum,
                                       constant= 1)
    else:
        logger.critical('Spectrum type not found.')
        raise ValueError('Spectrum type is not valid. Only sp.Spectrum1D and sp.SpectrumCollection can be requested.')
    
    match sn_type:
        case None:
            pass
        case 'Average_S_N':
            weights = weights.multiply(spectrum.meta['Average_S_N'] * u.dimensionless_unscaled)
        case 'quadratic':
            weights = weights.multiply(spectrum.meta['Average_S_N'] ** 2 * u.dimensionless_unscaled)
        case 'quadratic_error':
            # TODO Check if this doesn't break for uncertainty = 0
            weights = weights.multiply(spectrum.uncertainty.array**2 * u.dimensionless_unscaled)
        case 'quadratic_combined':
            weights = weights.multiply(spectrum.uncertainty.array**2 * u.dimensionless_unscaled)
            weights = weights.multiply(spectrum.meta['delta']**2 * u.dimensionless_unscaled)
    
    weights = weights.multiply(np.isfinite(spectrum.flux)*u.dimensionless_unscaled)
    
    return weights

def _gain_weights_list(spectrum_list: sp.SpectrumList,
                       sn_type: str | None = None) -> astropy.nddata.NDDataArray:
    """
    Generate weight spectrum for the master calculation.

    Parameters
    ----------
    spectrum : sp.SpectrumList
        Spectrum list with either sp.Spectrum1D or sp.SpectrumCollection objects.
    sn_type : str | None
        Type of weighting, by default None.
        Options are:
            None:
                No weights assumed. This will return weights with np.isfinite(flux) value, to handle NaN values. 
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
    weights : astropy.nddata.NDDataArray
        Weights for the spectrum list.
    """
    weights = NDDataArray(data = ([np.logical_and(np.isfinite(item.flux),
                                                 np.isfinite(item.uncertainty.array)
                                                  ) for item in spectrum_list]))
    
    # TESTME
    match sn_type:
        case None:
            return weights
        case 'Average_S_N':
            scale = [item.meta['Average_S_N'] for item in spectrum_list]* u.dimensionless_unscaled
        case 'quadratic':
            scale = [item.meta['Average_S_N']**2 for item in spectrum_list]* u.dimensionless_unscaled
        case 'quadratic_error':
            # TODO Check if this doesn't break for uncertainty = 0
            scale = [item.uncertainty.array**(-2) for item in spectrum_list]* u.dimensionless_unscaled
        case 'quadratic_combined':
            scale = [item.uncertainty.array**(-2) for item in spectrum_list]* u.dimensionless_unscaled
            scale *= [item.meta['delta']**2 for item in spectrum_list]* u.dimensionless_unscaled
    weights = weights.multiply(scale, axis=0)
    
    return weights


def _empty_spectrum_like(spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                         constant: float | int = 0,
                         unit: None | u.Unit = None,
                         ) -> sp.Spectrum1D | sp.SpectrumCollection:
    """
    Create an empty spectrum of the same shape as the input. The type of spectrum is inherited by input (either sp.Spectrum1D or sp.SpectrumCollection)

    Parameters
    ----------
    spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Input spectrum of the desired shape.
    constant : float | int
        Constant which is assigned to the flux values, by default 0.
    unit : u.Unit | None
        Unit which to add to the flux array, by default None.

    Returns
    -------
    new_spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Empty spectrum with the desired shape.

    Raises
    ------
    TypeError
        If input spectrum is not sp.Spectrum1D or sp.SpectrumCollection object, raise TypeError.
    """
    if unit is None:
        unit = u.dimensionless_unscaled
    
    if type(spectrum) == sp.Spectrum1D:
        new_spectrum = sp.Spectrum1D(
            spectral_axis= spectrum.spectral_axis,
            flux= (np.zeros_like(spectrum.flux.value) + constant) * unit,
            uncertainty= StdDevUncertainty(np.zeros_like(spectrum.flux.value))
            )
    elif type(spectrum) == sp.SpectrumCollection:
        new_spectrum = sp.SpectrumCollection(
            spectral_axis= spectrum.spectral_axis,
            flux= (np.zeros_like(spectrum.flux.value) + constant) * unit,
            uncertainty= StdDevUncertainty(np.zeros_like(spectrum.flux.value))
            )
    else:
        logger.critical('Spectrum type not found.')
        raise TypeError('Spectrum type is not valid. Only sp.Spectrum1D and sp.SpectrumCollection can be requested.')
    return new_spectrum
