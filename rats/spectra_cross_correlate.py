"""
Cross-correlation functions of the RATS package.

"""

import astropy
import specutils as sp
import numpy as np
from functools import singledispatch
import astropy.units as u
import rats.spectra_manipulation as sm
import rats.spectra_manipulation_subroutines.calculation as smcalc
import logging
import astropy.nddata as nd
from rats.utilities import default_logger_format, time_function
from itertools import repeat
from mpire import WorkerPool, cpu_count

NUM_CORES = cpu_count() - 1
#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)



@time_function
def cross_correlate_list(
    spectrum_list: sp.SpectrumList,
    model: sp.Spectrum1D,
    velocities: u.Quantity = np.arange(-200, 200.5, 0.5) * u.km/u.s,
    sn_type: str | None = None,
    force_multiprocessing: bool = True) -> sp.SpectrumList:
    """
    Cross-correlate branch for sp.Spectrum1D objects. 

    Parameters
    ----------
    spectral_data : sp.SpectrumList
        Spectrum to cross-correlate with model. Each spectrum must be of the same shape as model.
    model : sp.Spectrum1D | sp.SpectrumCollection
        Model template that is cross-correlated with spectrum. Must be same type and shape as spectrum, unless spectrum is a SpectrumList. In that case, each spectrum must be of the same shape as the model.
    velocities : u.Quantity, optional
        Range of velocities to cross-correlate over, by default np.arange(-200,200,0.5)*u.km/u.s
    sn_type : str | None, optional
        Type of weights to use, by default None. Check smcalc._gain_weights_list() for options.

    Returns
    -------
    CCF : sp.SpectrumList
        CCF of spectrum list and model. Length of the list is the same as the spectral data. Shape of individual spectrum is the same as the model and each spectrum of spectral data.
    """
    
    for spectrum in spectrum_list:
        assert (spectrum.spectral_axis == model.spectral_axis).all(), 'Spectrum and model have different spectral axis.'
    
    CCF_flux = np.zeros_like([velocities.value]*len(spectrum_list))
    
    if force_multiprocessing:
        logger.warning('Starting multiprocessing - CCF calculation | This is extremely memory heavy.')
        weights = smcalc._gain_weights_list(spectrum_list, sn_type= sn_type)
        flux_list = astropy.nddata.NDDataArray([item.flux for item in spectrum_list])
        flux_list = np.nan_to_num(flux_list.data, nan=0.0)
        
        with WorkerPool(n_jobs=NUM_CORES, shared_objects=(flux_list, weights, model)) as pool:
            CCF_flux = pool.map(_CCF_multiprocessing_wrapper,
                                velocities,
                                progress_bar=True,
                                chunk_size=1)
        CCF_flux = CCF_flux.reshape(len(velocities), len(spectrum_list)).T
        logger.warning('Finished multiprocessing - CCF calculation')
    else:
        for ind, velocity in enumerate(velocities):
            logger.info(f'Calculation of velocity {velocity} | progress: {(ind/len(velocities)*100):.2f}%')
            # We are shifting model, not the spectrum list, hence the reverse sign
            shifted_model = sm._shift_spectrum(
                spectrum= model,
                velocities= [-velocity]
                )
            flux_list = astropy.nddata.NDDataArray([item.flux for item in spectrum_list])
            
            CCF_flux[:, ind] = _calculate_CCF_single_velocity(
                spectrum_list= spectrum_list,
                shifted_model= shifted_model,
                flux_list= flux_list,
                sn_type= sn_type)
        
    CCF_list = sp.SpectrumList()
    
    for (row, spectrum) in zip(CCF_flux, spectrum_list):
        CCF_list.append(
            sp.Spectrum1D(
                spectral_axis = velocities,
                flux = row * u.dimensionless_unscaled,
                meta= spectrum.meta,
                )
            )
    return CCF_list

def _CCF_multiprocessing_wrapper(shared_objects: tuple,
                                 velocity: u.Quantity, # Iterated
                                 ) -> astropy.nddata.NDDataArray:
    """
    CCF calculation wrapper for multiprocessing function. It is a 2-step function of first shifting the template model and then calculation of the CCF array.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list which to cross-correlate with the template
    model : sp.Spectrum1D | sp.SpectrumCollection
        Template model which is being cross-correlated with the spectrum_list.
    velocity : u.Quantity
        Velocity by which to shift the template.
    sn_type : str | None
        Type of weighting used. For options, check smcalc._gain_weights_list() function.

    Returns
    -------
    CCF_flux : astropy.nddata.NDDataArray
        The CCF values for all spectra in spectrum list.
    """
    # flux_list = shared_objects[0]
    # weights= shared_objects[1]
    # model =shared_objects[2]
    shifted_model = sm._shift_spectrum(
        spectrum= shared_objects[2],
        velocities= [-velocity]
        )
    CCF_flux = _calculate_CCF_single_velocity_test(
        weights= shared_objects[1],
        shifted_model= shifted_model,
        flux_list= shared_objects[0],
        )
    return CCF_flux

#%%
def _calculate_CCF_single_velocity(spectrum_list: sp.SpectrumList,
                                   shifted_model: sp.Spectrum1D | sp.SpectrumCollection,
                                   flux_list: astropy.nddata.NDDataArray,
                                   sn_type: str | None = None) -> astropy.nddata.NDDataArray:
    """
    Calculate CCF values for single velocity, provided spectrum list and shifted model.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list on which to cross-correlate the model.
    shifted_model : sp.Spectrum1D | sp.SpectrumCollection
        Shifted model which to cross-correlate
    sn_type : str | None, optional
        Type of weights to use, by default None. Check smcalc._gain_weights_list() for options.

    Returns
    -------
    CCF_value : astropy.nddata.NDDataArray
        The value of CCF of all spectra for given velocity. The shape is the length of the spectrum_list.
        
    """

    assert type(spectrum_list[0]) == type(shifted_model), "Type of spectrum and shifted model is not same"

    # Allocate arrays
    weights = smcalc._gain_weights_list(spectrum_list, sn_type= sn_type)
    model_flux = np.repeat(shifted_model.flux[np.newaxis, :], len(spectrum_list), axis=0)

    # Run the calculation of weighted CCF
    match type(shifted_model):
        case sp.Spectrum1D:
            true_weight = weights.multiply(model_flux)
            CCF_Flux = flux_list.multiply(true_weight)
            CCF_value = np.nansum(CCF_Flux, axis=1) / np.nansum(true_weight, axis=1)
        case sp.SpectrumCollection:
            true_weight = weights.multiply(model_flux)
            CCF_Flux = flux_list.multiply(true_weight)
            CCF_value = np.apply_over_axes(np.nansum, CCF_Flux, [1,2]) / np.apply_over_axes(np.nansum, true_weight, [1,2])
    return CCF_value

def _calculate_CCF_single_velocity_test(weights: astropy.nddata.NDDataArray,
                                   shifted_model: sp.Spectrum1D | sp.SpectrumCollection,
                                   flux_list: astropy.nddata.NDDataArray
                                   ) -> astropy.nddata.NDDataArray:

    if type(shifted_model) == sp.SpectrumCollection:
        raise NotImplementedError('Not supported yet!')

    template = shifted_model.flux[:, np.newaxis] * weights.data.T
    template = template / np.nansum(template,axis=0)
    template = np.nan_to_num(template, nan=0.0)
    
    return np.diag(flux_list @ template)
