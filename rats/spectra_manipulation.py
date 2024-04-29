# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:34:34 2021

@author: Chamaeleontis
"""

#%% Importing libraries
import specutils as sp
import astropy
import numpy as np
import rats.parameters as para
import rats.spectra_manipulation_subroutines.calculation as smcalc
import astropy.units as u
from functools import singledispatch 
import astropy.io.fits as fits
from pathos.multiprocessing import Pool
from itertools import repeat
from copy import deepcopy
import astropy.constants as con
import pickle
# import polars as pl
import warnings
import math
import specutils.fitting as fitting
import scipy as sci
import pandas as pd
import multiprocessing
# from joblib.externals.loky import set_loky_pickler
# from joblib import parallel_backend
# from joblib import Parallel, delayed
# from joblib import wrap_non_picklable_objects
import sys
import time
import traceback
from astropy.wcs import WCS
from rats.utilities import time_function, save_and_load, progress_tracker, disable_func, skip_function, default_logger_format
from typing import Any
import logging

logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

@progress_tracker
@time_function
def sigma_clipping_list(spectrum_list: sp.SpectrumList,
                        num_of_sigma: float = 5,
                        window_size: int = 1000
                        ) -> sp.SpectrumList:
    """
    Clips the spectrum list by number of sigma.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list to sigma-clip.
    num_of_sigma : float, optional
        Number of sigma to clip by, by default 5
    window_size : int, optional
        Size of the rolling window, by default 1000.

    Returns
    -------
    new_spectrum_list : sp.SpectrumList
        Sigma-clipped spectrum list.
    """
    new_spectrum_list = sp.SpectrumList()
    
    for item in spectrum_list:
        new_spectrum = _sigma_clipping_spectrum(item,
                                            num_of_sigma=num_of_sigma)
        new_spectrum_list.append(new_spectrum)

    return new_spectrum_list

def _sigma_clipping_spectrum(spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                             num_of_sigma:float,
                             window_size: int = 500,
                             ) -> sp.Spectrum1D | sp.SpectrumCollection:
    """
    Sigma clipping of single spectrum. The rolling window is of size 1000 by default.

    Parameters
    ----------
    spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Spectrum to sigma-clip.
    num_of_sigma : float
        Number of sigma to sigma clip by.
    window_size : int, optional
        Size of the rolling window, by default 1000.

    Returns
    -------
    new_spectrum : sp.Spectrum1D | sp.SpectrumCollection
        New spectrum which has been sigma clipped by.
    """
    flux = pl.Series('flux', spectrum.flux.value).fill_nan(None)
    while True:
        rolling_std = flux.rolling_std(window_size= window_size, center= True , min_periods= 1
                                   ) * num_of_sigma
        rolling_median = flux.rolling_median(window_size= window_size, center= True , min_periods= 1
                                      )
        new_flux = np.where(abs(flux - rolling_median) < rolling_std,
                            flux,
                            np.nan
                            )
        
        if sum(np.isnan(flux.to_numpy())) == sum(np.isnan(new_flux)):
            break
        else:
            flux = pl.Series('flux', new_flux).fill_nan(None)
    
    if type(spectrum) == sp.Spectrum1D:
        new_spectrum = sp.Spectrum1D(
                spectral_axis= spectrum.spectral_axis,
                flux= new_flux * spectrum.flux.unit,
                uncertainty= spectrum.uncertainty,
                mask = np.isnan(new_flux),
                meta= spectrum.meta,
            )
    if type(spectrum) == sp.SpectrumCollection:
        raise NotImplementedError('Please test whether this implementation work.')
        new_spectrum = sp.SpectrumCollection(
                spectral_axis= spectrum.spectral_axis,
                flux= new_flux,
                uncertainty= spectrum.uncertainty,
                mask = np.isnan(new_flux),
                meta= spectrum.meta,
            )
    return new_spectrum


#%% Masking values below a flux threshold
def mask_flux_below_threshold_in_list(spectrum_list:sp.SpectrumList,
                                      threshold: float = 100,
                                      polynomial_order: int = 6,
                                      execute: bool = True) -> sp.SpectrumList:
    new_spectrum_list = sp.SpectrumList()
    
    for spectrum in spectrum_list:
        new_spectrum_list.append(
            smcalc._mask_flux_below_threshold(spectrum= spectrum,
                                              threshold= threshold,
                                              polynomial_order= polynomial_order)
            )
        print('Hello')
    if execute:
        new_spectrum_list = execute_mask_in_list(new_spectrum_list)
    return

#%% Masking non-photon noise dominated pixels
def mask_non_photon_noise_dominated_pixels_in_list(spectrum_list: sp.SpectrumList,
                                                   threshold: float= 0.5,
                                                   execute: bool = True) -> sp.SpectrumList:
    """
    Give a spectra list an updated mask to mask the non-photon noise dominated pixels.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectra list for which to find the mask.
    threshold : float, optional
        Threshold by which to mask, by default 0.5. This masks pixels that have 50% structure of non-photon noise. The factor is a percentage of how much "domination" should the non-photon noise sources have. For example, to get a 10% mask (pixels with non-photon noise on the order of magnitude as the photon noise), threshold = 0.1.
    execute : bool, optional
        Whether to execute the mask, by default True. Otherwise the mask is only saved in the mask keyword
        
    Returns
    -------
    spectrum_list : sp.SpectrumList
        Spectra list object with updated mask attribute.
    """
    new_spectrum_list = sp.SpectrumList()
    
    for spectrum in spectrum_list:
        new_spectrum_list.append(
            smcalc._mask_non_photon_noise_dominated_pixels_in_spectrum(spectrum= spectrum,
                                                                threshold= threshold)
            )
    if execute:
        new_spectrum_list = execute_mask_in_list(new_spectrum_list)
    return new_spectrum_list

#%% Execute masks in spectrum list
def execute_mask_in_list(spectrum_list: sp.SpectrumList) -> sp.SpectrumList:
    # TODO
    # DOCUMENTME
    
    new_spectrum_list = sp.SpectrumList()
    for spectrum in spectrum_list:
        new_spectrum_list.append(
            smcalc._execute_mask_in_spectrum(spectrum= spectrum)
        )
    
    return new_spectrum_list


#%% custom_transmission_units
def custom_transmission_units(system_parameters: para.SystemParametersComposite
                              ) -> [u.Equivalency, u.Unit, u.Unit, u.Unit, u.Unit, u.Unit]:
    """
    Defines transmission spectrum specific units for given planet. Conversion equations are available at https://www.overleaf.com/read/gkdtkqvwffzn

    Parameters
    ----------
    sys_para : SystemParameters
        System parameters of given system..

    Returns
    -------
    equivalency : Equivalency
        Equivalency between the unit system (see below definitions).
    F_lam : u.Unit
        Standard way of representing transmission spectrum as (1- atmospheric absorption).
    R : u.Unit
        Atmospheric absorption only. No signal should be centered at 0, absorption signal should go up.
    R_plam : u.Unit
        Absorption in units of planetary radius. No atmosphere should be centered at Rp value, atmospheric signal is aiming up
    delta_lam : u.Unit
        Transit depth like units. No atmosphere should be centered at transit depth (delta) value (represented as number, not percentage). Atmospheric signal is aiming up.
    H_num : u.Unit
        Number of atmospheric scale heights

    """
    # Constants for given system
    Rp, Rs = system_parameters.Planet.radius.data * system_parameters.Planet.radius.unit, system_parameters.Star.radius.data * system_parameters.Star.radius.unit
    system_parameters._calculate_atmospheric_scale_height()
    H = (system_parameters.Planet.atmospheric_scale_height.convert_unit_to(u.km).data) # In km
    rat = (Rs/Rp).decompose().value # There is a rat in my code, oh no!
    Rp_km = Rp.to(u.km).value # Planet radius in km
    
    # Definition of units
    F_lam = u.def_unit(['','T','Transmitance','Transmitted flux'])
    R = u.def_unit(['','R','Excess atmospheric absorption', 'A'])
    R_plam =  u.def_unit(['Rp','Planetary radius'])
    delta_lam =  u.def_unit(['','Wavelength dependent transit depth'])
    H_num =  u.def_unit(['','Number of scale heights'])
    
    equivalency_transmission = u.Equivalency(
        [
            # Planetary radius (white-light radius assumed)
            (F_lam, R_plam,
                lambda x: rat* (1-x+x/rat**2)**(1/2),
                lambda x: (x**2 - rat**2)/(1-rat**2)
              ),
            # R = (1- Flam)
            (F_lam, R,
              lambda x: 1-x,
              lambda x: 1-x
              ),
            # Transit depth
            (F_lam, delta_lam,
              lambda x: 1-x+x*rat**(-2),
              lambda x: (x-1)/(-1+rat**(-2))
              ),
            # R to planetary radius
            (R, R_plam,
              lambda x: rat * np.sqrt(x + rat**(-2) - x*rat**(-2)),
              lambda x: (x**2-1 ) / (rat**2 -1)
              ),
            # R to transit depth
            (R, delta_lam,
              lambda x: x + rat**(-2) - rat**(-2)*x,
              lambda x: (x-rat**(-2)) / (1-rat**(-2))
              ),
            # Rp_lam to transit depth 
            (R_plam, delta_lam,
              lambda x: x**2 * (rat**(-2) -1) / (1-rat**(2)) ,
              lambda x: np.sqrt( (x*(1-rat**2) + rat**2 - rat**(-2)) / (rat**(-2)-1) )
              ),
            # H_num
            (F_lam, H_num,
              lambda x: (rat* (1-x+x/rat**2)**(1/2)-1)/H,
              lambda x: ((x*H+1)**2*rat**(-2)-(1))/(-1+rat**(-2))
              ),
            (R, H_num,
              lambda x: 1-((rat* (1-x+x/rat**2)**(1/2)-1)/H),
              lambda x: 1-((x*H+1)**2*rat**(-2)-(1))/(-1+rat**(-2)) 
              ),
            # Use this equivalency, please
            (R_plam, H_num,
              lambda x: (x-1)*Rp_km/H,
              lambda x: (x*H / Rp_km)+1
              ),
            (delta_lam, H_num,
              lambda x: (rat*x**(1/2)-1)/H,
              lambda x: (x*H+1)**2*rat**(-2)
              ),
        ],
        "Transmission",
    )
    
    return equivalency_transmission, F_lam, R, R_plam, delta_lam, H_num


#%% extract_subregion
def _extract_subregion(spectrum: sp.Spectrum1D,
                       subregion: sp.SpectralRegion) -> np.ndarray:
    """
    Convenience function that extracts indices of given subregion

    Parameters
    ----------
    spectrum : sp.Spectrum1D
        Spectrum from which to extract the subspectrum.
    subregion : sp.SpectralRegion
        Subregion of sp.SpectralRegion with size of 1.

    Returns
    -------
    ind : np.ndarray
        Indices of which pixels to include.

    """
    ind = np.where(
        np.logical_and(
        spectrum.spectral_axis > subregion.lower,
        spectrum.spectral_axis < subregion.upper,
        )
        )
    return ind

#%% extract_region
def extract_region_in_spectrum(spectrum: sp.Spectrum1D,
                               spectral_region: sp.SpectralRegion) -> sp.Spectrum1D:
    """
    Extract region from spectrum

    Parameters
    ----------
    spectrum : sp.Spectrum1D
        Spectrum from which to extract the subspectrum..
    spectral_region : sp.SpectralRegion
        Spectral region from which to take the subspectrum.

    Returns
    -------
    cut_spectrum : sp.Spectrum1D
        Cut spectrum with spectral_region.

    """
    ind = np.array([],dtype =int)
    
    for subregion in spectral_region: # Extract all indices 
        ind = np.append(ind,(_extract_subregion(spectrum,subregion)))
    cut_spectrum = sp.Spectrum1D( # Create new spectrum with old parameters
        spectral_axis =spectrum.spectral_axis[ind],
        flux = spectrum.flux[ind],
        uncertainty = spectrum.uncertainty[ind],
        mask = spectrum.mask[ind].copy(),
        meta = spectrum.meta.copy(),
        )
    
    return cut_spectrum
#%% extract_region_list
def extract_region_in_list(spectrum_list: sp.SpectrumList,
                           spectral_region: sp.SpectralRegion) -> sp.SpectrumList:
    """
    Extracts region of wavelength from list of spectra

    Parameters
    ----------
    spec_list : sp.SpectrumList
        Spectra list to extract region from.
    spectral_region : sp.SpectralRegion
        Spectral region which to include in the spectrum.

    Returns
    -------
    new_spectrum_list : sp.SpectrumList
        Cutted spectrum list based on spectral region.

    """
    new_spectrum_list = sp.SpectrumList()
    for item in spectrum_list:
        new_spectrum_list.append(
            extract_region_in_spectrum(item,spectral_region)
            )
    return new_spectrum_list

#%% extract_sgl_order
def extract_sgl_order(spec_coll,order):
    """
    Extract specific order from spectrum collection
    
    Input:
        spec_coll ; sp.SpectrumCollection
    Output:
        spec_list ; sp.SpectrumList
    """
    spec_list = sp.SpectrumList()
    for item in spec_coll:
        item.meta.update({'S_N':item.meta['S_N_all'][order]})
        spec_list.append(item[order])
    return spec_list

#%% extract_dbl_order
def extract_dbl_order(spec_coll_list,orders):
    """
    Extract specific two orders from spectrum collection (For ESPRESSO double orders)
    
    Input:
        spec_coll ; sp.SpectrumCollection
        orders ; order to extract
            WARNING: ESPRESSO orders are from 1 to 170, while sp.SpectrumCollection works
                     from 0 to 169
    Output:
        spec_list ; sp.SpectrumList
    """
    spec_list = sp.SpectrumList()
    for item in spec_coll_list:
        for order in orders:
            item.meta.update({'S_N':item.meta['S_N_all'][order]})
            spec_list.append(item[order-1])
    return spec_list

#%% determine_rest_frame
def _determine_rest_frame(spectrum: sp.Spectrum1D | sp.SpectrumCollection):
    """
    Determine rest frame for given rest frame based on the velocities corrected
    Input:
        spectrum
    Change:
        meta['RF'] - changed to estimated RF (Earth, Barycenter_Sol ,Sun, Barycenter_Star, Star, Planet)
    """
    test_meta = [spectrum.meta['BERV_corrected'],
                 spectrum.meta['v_sys_corrected'],
                 spectrum.meta['v_star_corrected'],
                 spectrum.meta['v_planet_corrected']
                 ]
    if test_meta == [True,False,False,False]:
        spectrum.meta['RF'] = 'Earth'
    elif test_meta == [False,False,False,False]:
        spectrum.meta['RF'] = 'Barycentrum_Sol'
    elif test_meta == [False,True,False,False]:
        spectrum.meta['RF'] = 'Barycenter_Star'
    elif test_meta == [False,False,True,False] or test_meta == [False,True,True,False]:
        spectrum.meta['RF'] = 'Star'
    elif test_meta == [False,False,False,True] or test_meta == [False,True,False,True]:
        spectrum.meta['RF'] = 'Planet'
    else:
        logger.warning('Automatic rest frame finding routine failed.')
        logger.warning('    RF keyword changed to "undefined"')
        spectrum.meta['RF'] = 'undefined'
    return

#%% print_info
def print_info_spectrum_list(spectrum_list: sp.SpectrumList) -> None:
    """
    Print number indexes information about the spectra in the list.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list for which to print info.

    Returns
    -------
    None
    """
    logger.print('='*30)
    for item in spectrum_list:
        logger.print(f'Night: {item.meta["Night"]} | #Night: {item.meta["Night_num"]} | #Spectrum_in_night: {item.meta["Night_spec_num"]} | #Spectrum: {item.meta["Spec_num"]}')
    logger.print('='*30)
    return None

#%% tie_wlg
def tie_wlg(model):
    """
    Ties wavelengths between Na D1 and D2 lines given a model
    Input:
        model ; astropy.modeling.models.Gaussian1D
    Output:
        model tied together
    """
    # TODO Move to spectra_manipulation_routines.models
    return model.mean_0 + (5895.924-5889.950)

#%% get_sublist
def get_sublist(spectrum_list: sp.SpectrumList,
                key: str,
                value: Any,
                mode: str ='normal'
                ) -> sp.SpectrumList:
    """
    Returns a filtered sublist. Depending on the mode, it will return spectra with:
        mode= 'normal' (or 'equal'):
            spectrum.meta[key] == value
        mode= 'less':
            spectrum.meta[key] < value
        mode= 'more':
            spectrum.meta[key] > value
        mode= 'non-equal'
            spectrum.meta[key] != value

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list to filter through
    key : str
        Key of the meta parameter to filter with.
    value : Any
        Which value to filter by in the spectrum.meta[key]
    mode : str, optional
        Mode of the condition filter, by default 'normal'. Options are 'normal', 'less', 'more', 'non-equal'

    Returns
    -------
    new_spectrum_list : sp.SpectrumList
        Filtered spectrum list.
    """
    # CLEANME
    
    new_spectrum_list = spectrum_list.copy()
    
    # For single value extraction
    if (mode == 'normal') or (mode == 'equal'):
        for item in spectrum_list:
            if item.meta[key] != value:
                new_spectrum_list.remove(item)
    # For values smaller than value
    elif mode == 'less':
        for item in spectrum_list:
            if item.meta[key] > value:
                new_spectrum_list.remove(item)
    # For values higher than value
    elif mode == 'more':
        for item in spectrum_list:
            if item.meta[key] < value:
                new_spectrum_list.remove(item)
    # For values that are non-equal to value
    elif mode == 'non-equal':
        for item in spectrum_list:
            if item.meta[key] == value:
                new_spectrum_list.remove(item)
    return new_spectrum_list


#%% binning_spectrum
def binning_spectrum(spectrum: sp.Spectrum1D,
                     bin_factor: int = 10):
    """
    Bins a input spectrum by bin_factor*pixels

    Parameters
    ----------
    spectrum : sp.Spectrum1D
        Input spectrum to bin.
    bin_factor : int, optional
        How many pixels we want to bin by. The default is 10.

    Returns
    -------
    x
        x values of the binned spectrum.
    y
        y values of the binned spectrum.
    y_err
        y_err values of the binned spectrum.
    
    """
    
    num_bins = round(spectrum.spectral_axis.shape[0] / bin_factor)
    
    x = sci.stats.binned_statistic(spectrum.spectral_axis, spectrum.spectral_axis, statistic=np.nanmean, bins=num_bins)
    
    y = sci.stats.binned_statistic(spectrum.spectral_axis, spectrum.flux, statistic=np.nanmean, bins=num_bins)
    
    y_err = sci.stats.binned_statistic(spectrum.spectral_axis, spectrum.uncertainty.array, statistic= smcalc._error_bin, bins=num_bins)
    # Shouldn't this be returned as spectrum? On one side, it makes more sense to return it as sp.Spectrum1D, on other, it will be tricky to use rebinning animations as such. 
    return x.statistic,y.statistic,y_err.statistic

#%% shift_spectrum
def _shift_spectrum(spectrum: sp.Spectrum1D | sp.SpectrumCollection, 
                    velocities: list
                    ) -> sp.Spectrum1D | sp.SpectrumCollection:
    """
    Shifts spectrum by a list of velocities.

    Parameters
    ----------
    spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Spectrum to shift. The output will be the same as input
    velocities : list
        Velocity list to shift by. Must be a list of astropy Quantities in the units of velocity.

    Returns
    -------
    new_spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Shifted spectrum, interpolated to the old wavelength grid.
    """
    if spectrum.spectral_axis.unit.is_equivalent(u.km/u.s): # If spectral axis is in velocity space
        new_x_axis = spectrum.spectral_axis.to_value(u.km/u.s) *u.km/u.s # This seems weird
        for velocity in velocities:
            new_x_axis = new_x_axis + velocity
    elif spectrum.spectral_axis.unit.is_equivalent(u.AA): # If spectral axis is in wavelength space
        term = 1 # Initiation of the term (1 since we are dividing/multiplying the term)
        for velocity in velocities: # To factor all velocities (replace with relativistic one?)
            term = term / (1+velocity.to(u.m/u.s)/con.c) # Divide as the velocities are in opposite sign
        new_x_axis = spectrum.spectral_axis * term # Apply Doppler shift
    else: 
        raise ValueError(f'The unit of velocities is wrong. The units of velocities is {[velocity.unit.is_equivalent(u.AA) for velocity in velocities]}')
    # Mask handling
    # Combine the mask arrays of flux and errors
    mask_flux = ~np.isfinite(spectrum.flux)
    mask_err = ~np.isfinite(spectrum.uncertainty.array)
    mask = np.logical_or(mask_flux, mask_err)

    # Interpolation function for flux - cubic spline with no extrapolation
    interpolate_flux = sci.interpolate.CubicSpline(new_x_axis[~mask],
                                           spectrum.flux[~mask],
                                           extrapolate= False)
    # Interpolation function for uncertainty - cubic spline with no extrapolation
    
    # Calculated with square of uncertainty, than final uncertainty is np.sqrt()
    interpolate_error = sci.interpolate.CubicSpline(new_x_axis[~mask],
                                           spectrum.uncertainty.array[~mask]**2,
                                           extrapolate= False)
    # Applying interpolation functions to the old wave_grid
    # mask = ~mask # Indices of good pixels (both flux and error)
    new_flux = interpolate_flux(spectrum.spectral_axis) # Interpolate on the old wave_grid
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_uncertainty = np.sqrt(interpolate_error(spectrum.spectral_axis)) # Interpolate on the old wave_grid
        
    # Masking values that were NaN
    new_flux[mask] = np.nan
    new_uncertainty[mask] = np.nan
    
    new_uncertainty = astropy.nddata.StdDevUncertainty(new_uncertainty) # Interpolate on the old wave_grid
    
    # Create new spectrum
    if type(spectrum) == sp.Spectrum1D:
        new_spectrum = sp.Spectrum1D(
            spectral_axis = spectrum.spectral_axis,
            flux = new_flux * spectrum.flux.unit,
            uncertainty = new_uncertainty,
            meta = spectrum.meta.copy(),
            mask = mask,
            wcs = spectrum.wcs,
            )
    elif type(spectrum) == sp.SpectrumCollection:
        logger.warning('Spectral Collection format has been untested, please verify')
        new_spectrum = sp.SpectrumCollection(
            spectral_axis = spectrum.spectral_axis,
            flux = new_flux * spectrum.flux.unit,
            uncertainty = new_uncertainty,
            meta = spectrum.meta.copy(),
            mask = mask,
            wcs = spectrum.wcs,
            )
        logger.info('Finished correctly')
    
    return new_spectrum
#%% interpolate2commonframe
@singledispatch
def interpolate2commonframe(spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                            new_spectral_axis: sp.spectra.spectral_axis.SpectralAxis) -> sp.Spectrum1D | sp.SpectrumCollection:
    """
    Interpolate spectrum to new spectral axis.

    Parameters
    ----------
    spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Spectrum to interpolate to new spectral axis
    new_spectral_axis : sp.Spectrum1D.spectral_axis 
        Spectral axis to intepolate the spectrum to

    Returns
    -------
    new_spectrum : sp.Spectrum1D | sp.SpectrumCollection
        New spectrum interpolated to given spectral_axis
    """
    logger.critical('Spectra format is invalid.')
    raise ValueError('The spectrum format is invalid')

@interpolate2commonframe.register(sp.Spectrum1D)
def _interpolate2commonframe_1D(spectrum: sp.Spectrum1D, 
                               new_spectral_axis: sp.spectra.spectral_axis.SpectralAxis
                               ) -> sp.Spectrum1D:
    """
    Interpolate spectrum to new spectral axis.

    Parameters
    ----------
    spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Spectrum to interpolate to new spectral axis
    new_spectral_axis : sp.Spectrum1D.spectral_axis 
        Spectral axis to intepolate the spectrum to

    Returns
    -------
    new_spectrum : sp.Spectrum1D
        New spectrum interpolated to given spectral_axis
    """
    # Don't run when the spectrum already has the desired spectral axis.
    if len(spectrum.spectral_axis) == len(new_spectral_axis):
        if (spectrum.spectral_axis == new_spectral_axis).all(): 
            logger.debug('Skipping interpolation as the desired spectral axis is the same as of the spectrum.')
            return spectrum
    
    # Combine the mask arrays of flux and errors
    mask_flux = ~np.isfinite(spectrum.flux)
    mask_err = ~np.isfinite(spectrum.uncertainty.array)
    mask = np.logical_or(mask_flux, mask_err)
    
    flux_interpolate = sci.interpolate.CubicSpline(
        spectrum.spectral_axis[~mask],
        spectrum.flux[~mask],
        extrapolate= False
        )
    # Error is interpolated as square, then taken a square-root from it
    error_interpolate = sci.interpolate.CubicSpline(
        spectrum.spectral_axis[~mask],
        spectrum.uncertainty.array[~mask]**2,
        extrapolate= False
        )
    
    new_flux = flux_interpolate(new_spectral_axis) # Interpolate on the old wave_grid

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_uncertainty = np.sqrt(error_interpolate(new_spectral_axis)) # Interpolate on the old wave_grid
    
    mask = _compare_masks(spectrum.spectral_axis, mask, new_spectral_axis)
    
    # Masking values that were NaN
    new_flux[mask] = np.nan
    new_uncertainty[mask] = np.nan
    new_uncertainty = astropy.nddata.StdDevUncertainty(new_uncertainty)
    
    new_spectrum = sp.Spectrum1D(
        spectral_axis = new_spectral_axis,
        flux = new_flux * spectrum.flux.unit,
        uncertainty = new_uncertainty,
        meta = spectrum.meta.copy(),
        mask = mask,
        )
    return new_spectrum

@interpolate2commonframe.register(sp.SpectrumCollection)
def _interpolate2commonframe_2D(spectrum: sp.SpectrumCollection, 
                               new_spectral_axis: sp.spectra.spectral_axis.SpectralAxis
                               ) -> sp.SpectrumCollection:
    """
    Interpolate spectrum to new spectral axis.

    Parameters
    ----------
    spectrum : sp.SpectrumCollection
        Spectrum to interpolate to new spectral axis
    new_spectral_axis : sp.Spectrum1D.spectral_axis 
        Spectral axis to intepolate the spectrum to

    Returns
    -------
    new_spectrum : sp.SpectrumCollection
        New spectrum interpolated to given spectral_axis
    """
    # Don't run when the spectrum already has the desired spectral axis.
    if (spectrum.spectral_axis == new_spectral_axis).all(): 
        logger.debug('Skipping interpolation as the desired spectral axis is the same as of the spectrum.')
        return spectrum
    
    # Combine the mask arrays of flux and errors
    mask_flux = ~np.isfinite(spectrum.flux)
    mask_err = ~np.isfinite(spectrum.uncertainty.array)
    mask = np.ma.mask_or(mask_flux, mask_err)
    
    order_list = sp.SpectrumList()
    
    ind = 0
    for mask_order, order, new_spectral_axis_order in zip(
        mask,
        spectrum,
        new_spectral_axis
        ):
        
        flux_interpolate = sci.interpolate.CubicSpline(
            order.spectral_axis[~mask_order],
            order.flux[~mask_order],
            extrapolate= False
            )
        # Error is interpolated as square, then taken a square-root from it
        error_interpolate = sci.interpolate.CubicSpline(
            order.spectral_axis[~mask_order],
            order.uncertainty.array[~mask_order]**2,
            extrapolate= False
            )
        ind += 1
        new_flux_order = flux_interpolate(new_spectral_axis_order) # Interpolate on the old wave_grid

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_uncertainty_order = np.sqrt(error_interpolate(new_spectral_axis_order)) # Interpolate on the old wave_grid
        
        # Masking values that were NaN
        new_flux_order[mask_order] = np.nan
        new_uncertainty_order[mask_order] = np.nan
        new_uncertainty_order = astropy.nddata.StdDevUncertainty(new_uncertainty_order)
        
        order_list.append(
            sp.Spectrum1D(
                spectral_axis = new_spectral_axis_order,
                flux = new_flux_order * spectrum.flux.unit,
                uncertainty = new_uncertainty_order,
                meta = spectrum.meta.copy(),
                mask = mask_order,
                wcs = spectrum.wcs,
                )
            )
    new_spectrum = sp.SpectrumCollection.from_spectra(order_list)
    return new_spectrum

def _compare_masks(spectral_axis: sp.SpectralAxis,
                   mask: np.ndarray,
                   new_spectral_axis: sp.SpectralAxis) -> np.ndarray:
    """
    Utility function to create new mask for new spectral axis. This is necessary when interpolating on spectral axis of different length.

    Parameters
    ----------
    spectral_axis : sp.SpectralAxis
        Original spectral axis
    mask : np.ndarray
        Original mask for given spectral_axis
    new_spectral_axis : sp.SpectralAxis
        New spectral axis on which we interpolate.

    Returns
    -------
    new_mask : np.ndarray
        New mask as reshaped from input.
    """
    new_spectral_axis = sp.SpectralAxis(new_spectral_axis)
    new_mask = np.zeros_like(new_spectral_axis.value, dtype= bool)
    
    for masked_pixel in spectral_axis[mask]:
        ind = 0
        while True: # FIXME Not optimal
            if masked_pixel < new_spectral_axis.bin_edges[ind]:
                ind += 1
            else:
                new_mask[ind] = True
                break

    return new_mask
#%% binning_list
@progress_tracker
@skip_function
def binning_list(spectrum_list: sp.SpectrumList,
                 force_multiprocessing:bool = False,
                 force_skip:bool =False,
                 new_spectral_axis: None | sp.spectra.spectral_axis.SpectralAxis = None) -> sp.SpectrumList:
    """
    Bins the spectrum list to a common wavelength grid.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list for which to interpolate to common wavelength grid.
    force_multiprocessing : bool, optional
        Multiprocessing of the function, by default True
    force_skip : bool, optional
        If true, will skip the function completely, by default False
    new_spectral_axis : None | sp.spectra.spectral_axis.SpectralAxis
        If defined, will interpolate to this wavelength grid. Otherwise, it will interpolate to the wavelength grid of the first spectrum. By default None.

    Returns
    -------
    new_spectrum_list : sp.SpectrumList
        New spectrum list that is interpolated to a common wavelength grid

    Raises
    ------
    NotImplementedError
        The multiprocessing is currently not implemented
    """

    new_spectrum_list = sp.SpectrumList()
    if new_spectral_axis is None:
        new_spectral_axis = spectrum_list[0].spectral_axis

    if force_multiprocessing:
        logger.warning('Starting multiprocessing - interpolation to common grid')
        with Pool() as p:
            new_spectrum_list = p.starmap(interpolate2commonframe,
                        zip(spectrum_list,repeat(new_spectral_axis))
                        ) 
        logger.warning('Finished multiprocessing')
    else:
        for spectrum in spectrum_list:
            new_spectrum_list.append(
                interpolate2commonframe(spectrum,
                                        new_spectral_axis= new_spectral_axis)
            )
            logger.debug(f'Interpolated spectrum number {spectrum.meta["Spec_num"]} on common frame')
            
    return new_spectrum_list

def median_subtract(spectrum_list: sp.SpectrumList):
    
    if type(spectrum_list[0]) == sp.Spectrum1D:
        median_spectrum = sp.Spectrum1D(
            spectral_axis= spectrum_list[0].spectral_axis,
            flux = np.nanmedian(np.asarray([spectrum.flux for spectrum in spectrum_list]), axis=0) *u.dimensionless_unscaled
        )
        
    elif type(spectrum_list[0]) == sp.SpectrumCollection:
        raise NotImplementedError('To be added')
    
    new_spectrum_list = sp.SpectrumList()
    for spectrum in spectrum_list:
        new_spectrum_list.append(
            spectrum.subtract(median_spectrum, handle_meta='first_found').add(1*u.dimensionless_unscaled, handle_meta='first_found')
        )
    return new_spectrum_list

@time_function
def filter_median_absolute_deviation(spectrum_list: sp.SpectrumList,
                                     number_of_sigma: float = 5,
                                     window_size: int = 40,
                   ): 
    import polars as pl
    from copy import deepcopy
    
    flux = np.asarray([spectrum.flux.value for spectrum in spectrum_list])
    median_flux = np.nanmedian(flux, axis = 0)
    absolute_deviations = abs(flux - median_flux)
    flattened = pl.Series(absolute_deviations.T.flatten()).fill_nan(None)
    
    # As Series objects are only 1D objects, the flux has been reshaped properly
    output = flattened.rolling_median(window_size= window_size * len(spectrum_list),
                                      center=True,
                                      min_periods=1
                                      ).gather_every(
                                          len(spectrum_list)
                                          ) * number_of_sigma
    
    new_spectrum_list = sp.SpectrumList()
    spectrum_list = deepcopy(spectrum_list)
    for spectrum in spectrum_list:
        if type(spectrum) == sp.Spectrum1D:
            indices = spectrum.uncertainty.array > output.to_numpy()
            new_flux = spectrum.flux
            new_flux[indices] = np.nan
            new_uncertainty = spectrum.uncertainty.array
            new_uncertainty[indices] = np.nan
            
            new_spectrum_list.append(
                sp.Spectrum1D(
                    spectral_axis= spectrum.spectral_axis,
                    flux = new_flux,
                    uncertainty = astropy.nddata.StdDevUncertainty(new_uncertainty),
                    meta = spectrum.meta,
                    mask = np.isnan(new_flux)
                )
            )
    return new_spectrum_list

#%%
def normalize_spectrum(spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                            quantile: float = .85,
                            polyfit: None | int = None,
                            window_size: int = 7500) -> sp.Spectrum1D | sp.SpectrumCollection:
    # DOCUMENTME
    match polyfit:
        case None:
            normalized_spectrum = _normalize_spectrum_quantile(spectrum= spectrum,
                                                               quantile= quantile,
                                                               window_size= window_size)
        case polynomial_order if isinstance(polyfit, int):
            normalized_spectrum = _normalize_spectrum_polyfit(spectrum= spectrum,
                                                              polynomial_order= polyfit)
    return normalized_spectrum

def _normalize_spectrum_polyfit(spectrum,
                                polynomial_order: int) -> sp.Spectrum1D | sp.SpectrumCollection:
    # TODO
    return

def _normalize_spectrum_quantile(spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                                 quantile: float = .85,
                                 window_size: int = 7500) -> sp.Spectrum1D | sp.SpectrumCollection:
    # DOCUMENTME
    # FIXME
    if type(spectrum) == sp.SpectrumCollection:
        raise NotImplementedError
    old_flux = astropy.nddata.NDDataArray(data= spectrum.flux.value * u.dimensionless_unscaled,
                                          uncertainty = astropy.nddata.StdDevUncertainty(spectrum.uncertainty.array))
    
    # Dealing with endianness of arrays for pandas
    new_array = np.asarray(spectrum.flux.value)
    if new_array.dtype.byteorder != '=':
        new_array = new_array.byteswap().newbyteorder()
    
    normalization_function = pd.Series(new_array).rolling(window_size,
                             min_periods=1,
                             center=True).quantile(
                                 quantile
                                 )
                             
    normalization_function = astropy.nddata.NDDataArray(data= normalization_function)
    
    match type(spectrum):
        case sp.Spectrum1D:
            new_spectrum = sp.Spectrum1D(
                spectral_axis= spectrum.spectral_axis, 
                flux = old_flux.divide(normalization_function).data * u.dimensionless_unscaled,
                uncertainty= astropy.nddata.StdDevUncertainty(old_flux.divide(normalization_function).uncertainty.array.to_numpy()), # The numpy thing is weird, why does this work like this?
                mask= spectrum.mask.copy(),
                meta= spectrum.meta.copy(),
            )
        case sp.SpectrumCollection:
            raise NotImplementedError
        
    new_spectrum.meta['normalization'] = True
    return new_spectrum

#%% normalize_list
@progress_tracker
@save_and_load
def normalize_list(spectrum_list: sp.SpectrumList,
                   quantile: float= .85,
                   polyfit_order: int | None= None,
                   force_multiprocessing: bool= False,
                   force_load:bool= False,
                   force_skip:bool= False) -> sp.SpectrumList:
    """
    Normalize a spectrum list

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list to normalize.
    quantile : float, optional
        Quantile to which to normalize with rolling window, by default .85
    linfit : bool, optional
        If true, will do a linear fit instead, by default False
    force_multiprocessing : bool, optional
        If true, will run this function through multiprocessing, by default True
    force_load : bool, optional
        Whether to force loading of result (True), instead of calculation (False), by default False
    force_skip : bool, optional
        Whether to skip running the function completely (True), or run normally (False),, by default False

    Returns
    -------
    new_spectrum_list : sp.SpectrumList
        New spectrum list that has been normalized.
    """
    # For looping through list via multiprocesses
    
    if force_multiprocessing:
        # raise NotImplementedError
        logger.warning('Starting multiprocessing - normalization')
        with Pool() as p:
            # Throws bunch of warning on weakref, but seems to work. WTF?
            new_spectrum_list = p.starmap(normalize_spectrum,
                        zip(spectrum_list, repeat(quantile), repeat(polyfit_order))
                        )
        logger.warning('Finished multiprocessing')
        
    else:
        new_spectrum_list = sp.SpectrumList()
        for spectrum in spectrum_list:
            new_spectrum_list.append(
                normalize_spectrum(spectrum= spectrum,
                                   quantile= quantile,
                                   polyfit= polyfit_order)
            )

    return new_spectrum_list



#%% replace_flux_units_transmission
@save_and_load
def replace_flux_units_transmission(spec_list, unit, force_load = False, force_skip= False, pkl_name = ''):
    """
    Replace the flux units in the transmission spectra

    Parameters
    ----------
    spec_list : sp.SpectrumList
        Transmission spectra list.
    unit : TYPE
        Unit to replace the arbitrary dimensionless unit.

    Returns
    -------
    new_spec_list : sp.SpectrumList
        Transmission spectra list with updated units.

    """
    new_spec_list = sp.SpectrumList()
    
    for spectrum in spec_list:
        new_spectrum = sp.Spectrum1D(
            spectral_axis = spectrum.spectral_axis,
            flux = spectrum.flux.value * unit,
            uncertainty = astropy.nddata.StdDevUncertainty(spectrum.uncertainty.array * unit),
            mask = spectrum.mask,
            meta = spectrum.meta,
            )
        new_spec_list.append(new_spectrum)
    
    return new_spec_list
#%% cosmic_correction
@time_function
@progress_tracker
@save_and_load
def cosmic_correction_all(spec_list,force_load=False,force_skip=False):
    """
    Correction for cosmic for each night separately, using entire spectra list

    Parameters
    ----------
    spec_list : sp.SpectrumList
        Spectra to correct.

    Returns
    -------
    new_spec_list : sp.SpectrumList
        Corrected spectra.

    """
    num_nights = spec_list[-1].meta['Night_num']
    new_spec_list = sp.SpectrumList()
    for ni in range(num_nights):
        # Getting night data
        sublist = get_sublist(spec_list,'Night_num',ni+1)
        # Calculating master_night
        cosmic_corrected = smcalc._cosmic_correction_night(sublist)
        # Appending master_night
        new_spec_list.append(cosmic_corrected)
    new_spec_list = sum(new_spec_list,[])
    return new_spec_list


#%% calculate_master_list
@progress_tracker
@save_and_load
def calculate_master_list(spectrum_list: sp.SpectrumList,
                          key: None | str = None,
                          value: None | str = None,
                          sn_type: None | str = None,
                          method: str = 'average',
                          force_load: bool = False,
                          force_skip: bool = False,
                          pkl_name: str = ''
                          ) -> sp.SpectrumList:
    """
    Calculates list of masters for the full dataset. The type of master is defined by key, value, sn_type and method used. Ea

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list including the full dataset for which to calculate masters
    key : None | str, optional
        Key which is used to define type of master, by default None
    value : None | str, optional
        Value which is used for filtering the spectrum_list, by default None
    sn_type : None | str, optional
        Type of weighting, by default None. Check rats.spectra_manipulation_subroutines.calculation._gain_weights() for more details.
    method : str, optional
        Method which to use to generate the master, by default 'average'
    force_load : bool, optional
        Force loading the result of this function, instead of recalculating it, by default False
    force_skip : bool, optional
        Force skipping this funciton, by default False
    pkl_name : str, optional
        Name of the pickle_file where to save the output, by default ''

    Returns
    -------
    master_list : sp.SpectrumList
        List of master spectra made for each night [ind == 1...n] and nights combined [ind == 0].
    """
    
        
    # FIXME Should have a division on different instruments
    # TODO: The master_all spectrum could have multiple spectra inside the spectrum1D/ spectrumCollection to include different instruments.
    # CLEANME
    # Warning for non-normalized spectra
    if (spectrum_list[0].meta['normalization'] == False) & (key is not None):
        logger.warning('The spectra are not normalized.')
        logger.warning('    The master list will still be calculated.')
    
    # Getting the right sublist based on type of master
    if key is None:
        sublist = spectrum_list.copy()
    else:
        sublist = get_sublist(spectrum_list,key,value)
        spectrum_list = sublist
    
    assert (len(spectrum_list) != 0), 'The spectrum list is empty. Cannot calculate the master.' 
    
    spectrum_type = smcalc._get_spectrum_type(key= key,
                                             value= value)
    smcalc._check_night_ordering(spectrum_list)
    
    number_of_nights = sublist[-1].meta['Night_num']

    # Master of nights combined
    master_list = sp.SpectrumList()
    master_all = smcalc._calculate_master(sublist,
                            spectrum_type= spectrum_type,
                            night='nights-combined',
                            num_night = '"combined"',
                            rf = sublist[0].meta['RF'],
                            sn_type=sn_type,
                            method= method)
    master_list.append(master_all)
    
    for ni in range(number_of_nights):
        # Getting night data
        sublist = get_sublist(spectrum_list,'Night_num',ni+1)
        if len(sublist) == 0:
            logger.warning(f'No spectra were found for given key/value selection for night {ni}. This night will be skipped and the resulting master list will not have a spectrum for this night.')
            continue
        
        master_night = smcalc._calculate_master(
            spectrum_list= sublist,
            spectrum_type= spectrum_type,
            night= sublist[0].meta['Night'],
            num_night= str(ni+1),
            rf= sublist[0].meta['RF'],
            sn_type= sn_type,
            method= method
            )
        master_list.append(master_night)

    return master_list

#%% spec_list_master_correct
@progress_tracker
@save_and_load
def master_out_correction(spectrum_list: sp.SpectrumList,
                          master_list: sp.SpectrumList,
                          unit: u.Unit | None = None,
                          force_load: bool = False,
                          force_skip: bool = False,
                          pkl_name: str = ''
                          ) -> sp.SpectrumList:
    """
    Correct spectrum list for master out.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list to correct data for.
    master_list : sp.SpectrumList
        Master list to correct data with. Must be ordered such that 'Night_num' keyword is the night respective indice.
    unit : u.Unit | None, optional
        Unit to add to the corrected spectrum, by default None. If None, no unit is added. Normally the F_lam unit is correct, which corresponds to the differential unit gained by dividing normalized spectrum with its master out. Adding unit provides further functionality to the transmission spectrum, allowing different representation (such as atmospheric scale height).
    force_load : bool, optional
        Whether to force loading the output, by default False. If true, a pkl_name file is going to be looked for, and if found, the output is going to be loaded.
    force_skip : bool, optional
        Force skipping of the function, by default False. If true, the function is skipped completely. This means NO output is being given.
    pkl_name : str, optional
        Pickle name for the save data, by default ''. 

    Returns
    -------
    corrected_list : sp.SpectrumList
        Master out corrected spectrum list.
    """
    corrected_list = sp.SpectrumList()
    
    for item in spectrum_list:
        num_night = item.meta['Night_num']
        master_night = master_list[num_night]
        divided_spectrum = item.divide(master_night, handle_meta='first_found')
        
        corrected_list.append(divided_spectrum)
    if unit is not None:
        corrected_list = replace_flux_units_transmission(corrected_list, unit)
    
    return corrected_list
#%%
def master_out_with_RM_correction(spectrum_list: sp.SpectrumList,
                                  RM_model,
                                  unit: u.Unit | None = None,
                                  force_load: bool = False,
                                  force_skip: bool = False,
                                  pkl_name: str = '',
                                  ) -> sp.SpectrumList:
    ...
    # corrected_list = sp.SpectrumList()
    
    # for item in spectrum_list:
    #     master = sp.Spectrum1D(
    #         spectral_axis = RM_model.wl* 10 *u.AA,
    #         # flux = 
            
            
    #     )
        
    #     divided_spectrum = item.divide(master, handle_meta='first_found')
        
    
    
    return


#%% exciser_fill_with_nan
def exciser_fill_with_nan(spectrum,region):
    """
    Takes a spectrum and fills given region with NaNs
    Input:
        spectrum ; sp.Spectrum1D - spectrum to mask
        region ; sp.SpectralRegion - region to mask
    Output:
        new_spectrum ; sp.Spectrum1D - masked spectrum
    
    """
    spectral_axis = spectrum.spectral_axis
    excise_indices = None

    for subregion in region:
        # Find the indices of the spectral_axis array corresponding to the subregion
        region_mask = (spectral_axis >= region.lower) & (spectral_axis < region.upper)
        region_mask = (spectral_axis >= subregion.lower) & (spectral_axis < subregion.upper)
        temp_indices = np.nonzero(region_mask)[0]
        if excise_indices is None:
            excise_indices = temp_indices
        else:
            excise_indices = np.hstack((excise_indices, temp_indices))

    new_spectral_axis = spectrum.spectral_axis.copy()
    new_flux = spectrum.flux.copy()
    modified_flux = new_flux
    modified_flux[excise_indices] = np.nan
    if spectrum.mask is not None:

        new_mask = spectrum.mask
        new_mask[excise_indices] = True
    else:
        new_mask = None
    if spectrum.uncertainty is not None:

        new_uncertainty = spectrum.uncertainty
        # new_uncertainty[excise_indices] = np.nan
    else:
        new_uncertainty = None

    # Return a new object with the regions excised.
    return sp.Spectrum1D(flux=modified_flux,
                      spectral_axis=new_spectral_axis,
                      uncertainty=new_uncertainty,
                      mask=new_mask,
                      meta = spectrum.meta.copy(),
                      wcs=spectrum.wcs,
                      velocity_convention=spectrum.velocity_convention)

#%% spec_list_mask_region
def spec_list_mask_region(spec_list,sr):
    """
    Mask region with NaNs in a spectrum list
    Input:
        spec_list ; sp.SpectrumList to mask
        sr ; sp.SpectralRegion which values to mask
    Output:
        new_spec_list ; sp.SpectrumList masked spectrum list
    """
    new_spec_list = sp.SpectrumList()
    for ii,spec in enumerate(spec_list):
        item = exciser_fill_with_nan(spec,sr)
        new_spec_list.append(item)
    return new_spec_list

#%% extract_velocity_field
def extract_velocity_field(spectrum:sp.Spectrum1D,
                           shift_BERV:float,
                           shift_v_sys:float,
                           shift_v_star:float,
                           shift_v_planet:float,
                           shift_constant=0,
                           ):
    """
    Extracts velocity field for the shift
    Input:
        spectrum ; sp.Spectrum1D - spectrum from which to extract velocities
        shift_BERV ; float - 1/0/-1, otherwise the velocity is scaled
        shift_v_sys ; float - 1/0/-1, otherwise the velocity is scaled 
        shift_v_star ; float - 1/0/-1, otherwise the velocity is scaled 
        shift_v_planet ; float - 1/0/-1, otherwise the velocity is scaled
        shift_constant ; float * u.m/u.s or equivalent
    Output:
        velocity_field ; list - list of velocities to shift by
    """
    velocity_field = []
    if shift_BERV != 0:
        velocity_field.append(spectrum.meta['velocity_BERV'].data * spectrum.meta['velocity_BERV'].unit * shift_BERV)
    if shift_v_sys != 0:
        velocity_field.append(spectrum.meta['velocity_system'].data * spectrum.meta['velocity_system'].unit * shift_v_sys)
    if shift_v_star != 0:
        velocity_field.append(spectrum.meta['velocity_star'].data * spectrum.meta['velocity_star'].unit * shift_v_star)
    if shift_v_planet != 0:
        velocity_field.append(spectrum.meta['velocity_planet'].data * spectrum.meta['velocity_planet'].unit * shift_v_planet)
    if shift_constant != 0:
        velocity_field.append(shift_constant)
    return velocity_field
#%% shift_spectrum_multiprocessing
def shift_spectrum_multiprocessing(spectrum,
                           shift_BERV,
                           shift_v_sys,
                           shift_v_star,
                           shift_v_planet,
                           shift_constant):
    """
    Convenience function to pass to multiprocessing

    Parameters
    ----------
    spectrum : sp.Spectrum1D
        Spectrum to shift.
    shift_BERV : float
        Shift by BERV?
    shift_v_sys : float
        Shift by systemic velocity?
    shift_v_star : float
        Shift by stellar velocity?
    shift_v_planet : float
        Shift by planetary velocity?
    shift_constant : float
        Shift by arbitrary constant velocity?

    Returns
    -------
    new_spectrum : sp.Spectrum1D
        Shifted spectrum.

    """
    velocity_field =  extract_velocity_field(spectrum,
                               shift_BERV = shift_BERV,
                               shift_v_sys = shift_v_sys,
                               shift_v_star = shift_v_star,
                               shift_v_planet = shift_v_planet,
                               shift_constant = shift_constant,
                               )
    new_spectrum = _shift_spectrum(spectrum,velocity_field)
    return new_spectrum



#%% shift_list
@progress_tracker
@save_and_load
def shift_list(spectrum_list:sp.SpectrumList,
               shift_BERV:float,
               shift_v_sys:float, 
               shift_v_star:float, 
               shift_v_planet:float, 
               shift_constant:u.Quantity= 0, 
               shift_key: str | None = None,
               force_multiprocessing:bool = False,
               force_load = False,
               force_skip = False,
               pkl_name = '',
               ) -> sp.SpectrumList:
    """
    Shift spectrum list to different rest frame.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list which to shift.
    shift_BERV : float
        Whether to shift spectrum by BERV? If 1, will shift by BERV keyword inside each spectrum. If -1, will shift by -BERV. If 0, will not shift by BERV at all. Other floats are also possible, though the usage is not meaningful.
    shift_v_sys : float
        Whether to shift spectrum by systemic velocity. If 1, will shift by v_sys keyword inside each spectrum. If -1, will shift by -v_sys. If 0, will not shift by v_sys at all. Other floats are also possible, though the usage is not meaningful.
    shift_v_star : float
        Whether to shift by stellar velocity. If 1, will shift by v_star keyword inside each spectrum. If -1, will shift by -v_star. If 0, will not shift by v_star at all. Other floats are also possible, though the usage is not meaningful.
    shift_v_planet : float
        Whether to shift by planet velocity. If 1, will shift by v_planet keyword inside each spectrum. If -1, will shift by -v_planet. If 0, will not shift by v_planet at all. Other floats are also possible, though the usage is not meaningful.
    shift_constant : u.Quantity, optional
        Whether to shift by a constant shift, by default 0. Must be astropy quantity with velocity units.
    shift_key : str | None, optional
        Whether to shift by custom defined velocity, by default None. If not None, will look for the key in each spectrum.meta parameter. This parameter must be defined and a astropy.units Quantity in units of velocity.
    force_multiprocessing : bool, optional
        Whether to force multiprocessing, by default True.
    force_load : bool, optional
        _description_, by default False
    force_skip : bool, optional
        _description_, by default False
    pkl_name : str, optional
        _description_, by default ''

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """

    if shift_key is not None:
        raise NotImplementedError
    # Deepcopying spec_list
    spectrum_list = deepcopy(spectrum_list)
    # For cycle that shifts the spectral_axis by given list of velocities
    if force_multiprocessing:
        logger.warning('Starting multiprocessing - Shifting spectrum list')
        with Pool() as p:
            new_spectrum_list = p.starmap(shift_spectrum_multiprocessing,
                                          zip(spectrum_list,
                                              repeat(shift_BERV),
                                              repeat(shift_v_sys),
                                              repeat(shift_v_star),
                                              repeat(shift_v_planet),
                                              repeat(shift_constant),
                                            #   repeat(linfit)
                                              )
                        )
        new_spectrum_list = sp.SpectrumList(new_spectrum_list)
        logger.warning('Finished multiprocessing')
    else:
        new_spectrum_list = sp.SpectrumList()
        for spectrum in spectrum_list:
            new_spectrum_list.append(
                shift_spectrum_multiprocessing(spectrum,shift_BERV,
                                shift_v_sys,
                                shift_v_star,
                                shift_v_planet,
                                shift_constant)
            )
    return new_spectrum_list
#%% sort_spec_list
@skip_function
def sort_spectrum_list(spectrum_list:sp.SpectrumList,
                       force_skip=False) -> sp.SpectrumList:
    """
    Sorts spectrum list by BJD time located in the meta['BJD'] of each spectrum
    
    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list to sort.
    force_skip : bool, optional
        Whether to force skipping, by default False

    Returns
    -------
    sorted_spectrum_list : sp.SpectrumList
        Sorted spectrum list.
    """
    bjd = []
    for item in spectrum_list:
        bjd.append(item.meta['BJD'].value)
    ind = np.argsort(bjd)
    sorted_spectrum_list = sp.SpectrumList()
    for new_index, spectrum_index in enumerate(ind):
        sorted_spectrum_list.append(spectrum_list[spectrum_index])
    return sorted_spectrum_list
#%% extract_average_uncertainty_from_region
def extract_average_uncertainty_from_region(spec,line_list,diff_unc=1*u.AA):
    """
    Extracts average uncertainty from spectral region
    Input:
        spec ; sp.Spectrum1D - spectrum from which to extract uncertainty
        line_list ; list - list of lines in format [position1*u.AA,position2*u.AA]
        diff_unc ; float - value which defines the range around the line to calculate error from
    Output:
        unc_list ; list - list of errors for each line in list
    """
    unc_list = []
    for line in line_list:
        spec_region = sp.SpectralRegion(line-diff_unc,line+diff_unc)
        unc_list.append(np.nanmean(extract_region_in_spectrum(spec,spec_region).uncertainty.array))
    return unc_list
#%% velocity_domain_sgl
def velocity_domain_sgl(spectrum,line,constraint=[-100,100]*u.km/u.s):
    """
    Returns a spectrum in velocity domain around the line
    
    Input:
        spectrum ; sp.Spectrum1D - spectrum to represent in velocities
        line ; value *u.AA - line around which to calculate the velocities
    Output:
        velocity_spectrum ; sp.Spectrum1D - spectrum represented in velocity
    """
    velocity = (spectrum.spectral_axis - line)/ line * con.c # Transform to velocity range
    new_spectrum = sp.Spectrum1D( # Create new spectrum shifted
        spectral_axis = velocity,
        flux = spectrum.flux,
        uncertainty = astropy.nddata.StdDevUncertainty(spectrum.uncertainty.array),
        meta = spectrum.meta.copy(),
        mask = spectrum.mask.copy(),
        )
    spec_region = sp.SpectralRegion(constraint[0].to(u.m/u.s),constraint[1].to(u.m/u.s)) # Create subregion for the spectrum
    velocity_spectrum = extract_region_in_spectrum(new_spectrum,spec_region)
    return velocity_spectrum
#%% velocity_fold
def velocity_fold(spec_list,line_list,constraint=[-100,100]*u.km/u.s):
    """
    Folds the spectrum from wavelength range to velocity range and sums it over list of lines (eg. for cases of doublets)
    Input:
        spec_list ; sp.SpectrumList - list of spectra to fold
        line_list ; list ; list of lines to fold by
            line = value * u.AA
        constraint ; list ; list of lower and upper limit of velocity range
            value = velocity * u.km/u.s
    Output:
        spec_list_fold ; sp.SpectrumList - list of spectra folded around the lines and summed
    """
    for spectrum in spec_list:# For cycle through all spectra
        spec_list_velocity = sp.SpectrumList()
        for line in line_list: # For cycle through all lines           
            spec_list_velocity.append(velocity_domain_sgl(spectrum,line,constraint=constraint)) # Shift to velocity domain

            # print(velocity_domain_sgl(spectrum,line,constraint=constraint).flux)
            
        spec_list_velocity = binning_list(spec_list_velocity)
        flux = []
        uncertainty = []

        for item in spec_list_velocity:
            flux.append(item.flux)
            uncertainty.append(item.uncertainty.array**2/len(line_list))
            
        flux = np.array(flux)
        uncertainty = np.array(uncertainty)
        flux = np.nanmean(flux,axis = 0)
        uncertainty = np.nansum(uncertainty,axis=0)

        
        new_spectrum = sp.Spectrum1D(
            spectral_axis = item.spectral_axis,
            flux = deepcopy(flux) * u.ct,
            uncertainty = astropy.nddata.StdDevUncertainty(np.sqrt(uncertainty)),
            meta = item.meta.copy(),
            mask = item.mask.copy(),
            )
    return spec_list_velocity, new_spectrum
#%% RM_simulation
@disable_func
def RM_simulation(spec_list,master_RF_star_oot,sys_para):
    """
    Simulates the RM effect on the TS using the out-of-transit spectra and stellar local velocities. It is not very good at high stellar local velocities
    
    Input:
        spec_list ; sp.SpectrumList - list of spectra with the 'vel_stel_loc' and 'delta' meta information
        master_RF_star_oot ; sp.Spectrum1D - master out of the out-of-transit dataset
        
    Output:
        RM_simulated ; sp.SpectrumList - list of spectra showing the effect of RM in individual spectra
        In_transit ; sp.Spectrum1D - 1D spectrum with the overall effect of the RM effect on TS
            This spectrum is in units of wavelength dependent transit depth.
    """
    RM_simulated = sp.SpectrumList()
    for item in spec_list:
        if item.meta['Transit_full']:
            # Calculate master out spectrum shifted by local stellar velocity
            shifted = _shift_spectrum(master_RF_star_oot,
                                velocities= [item.meta['vel_stel_loc']]
                                )
            
            # Divide unshifted by shifted
            tmp_spec = prepare_const_spec(item,item.meta['delta'].value*u.dimensionless_unscaled)
            scaled_master = master_RF_star_oot.multiply(tmp_spec)
            new_shifted = scaled_master.divide(shifted)
            
            # Subtract 1 (Fout/Flocal -1)
            tmp_spec = prepare_const_spec(item,item.meta['delta'].value*u.dimensionless_unscaled)
            new_shifted_1= new_shifted.subtract(tmp_spec)
            
            # Shift to RFP
            shifted_to_RFP = _shift_spectrum(new_shifted_1,
                                velocities= [-item.meta['vel_st'],item.meta['vel_pl']]
                                )
            shifted_to_RFP.meta = item.meta.copy()
            
            RM_simulated.append(shifted_to_RFP)
            
    # Calculate master in transit
    In_transit = smcalc._calculate_master(RM_simulated,sn_type = 'quadratic_combined')
    return RM_simulated,In_transit
