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
from copy import deepcopy
import astropy.constants as con
import pickle
import math
import specutils.fitting as fitting
import scipy as sci
import pandas as pd
import multiprocessing
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
import sys
import time
import traceback
from astropy.wcs import WCS
from rats.utilities import time_function, save_and_load, progress_tracker, disable_func, skip_function, default_logger_format
from typing import Any
import logging

logger = logging.getLogger(__name__)
logger = default_logger_format(logger)
#%% Default pickler
set_loky_pickler('dill')

def mask_non_photon_noise_dominated_pixels_in_list(spectrum_list: sp.SpectrumList,
                                                   threshold: float= 0.5) -> sp.SpectrumList:
    """
    Give a spectra list an updated mask to mask the non-photon noise dominated pixels.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectra list for which to find the mask.
    threshold : float, optional
        Threshold by which to mask, by default 0.5. This masks pixels that have 50% structure of non-photon noise. The factor is a percentage of how much "domination" should the non-photon noise sources have. For example, to get a 10% mask (pixels with non-photon noise on the order of magnitude as the photon noise), threshold = 0.1.
        
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
def custom_transmission_units(sys_para: para.SystemParametersComposite
                              ) -> tuple[u.Equivalency, u.Unit, u.Unit, u.Unit, u.Unit, u.Unit]:
    """
    Defines transmission spectrum specific units for given planet. Conversion equations are available at https://www.overleaf.com/read/gkdtkqvwffzn

    Parameters
    ----------
    sys_para : SystemParameters
        System parameters of given system..

    Returns
    -------
    tuple[Equivalency, Unit, Unit, Unit, Unit, Unit]
        equivalency : Equivalency
            Equivalency between the unit system (see below definitions).
        F_lam : Unit
            Standard way of representing transmission spectrum as (1- atmospheric absorption).
        R : Unit
            Atmospheric absorption only. No signal should be centered at 0, absorption signal should go up.
        R_plam : Unit
            Absorption in units of planetary radius. No atmosphere should be centered at Rp value, atmospheric signal is aiming up
        delta_lam : Unit
            Transit depth like units. No atmosphere should be centered at transit depth (delta) value (represented as number, not percentage). Atmospheric signal is aiming up.
        H_num : Unit
            Number of atmospheric scale heights

    """
    # Constants for given system
    Rp, Rs = sys_para.planet.radius, sys_para.star.radius
    H = (sys_para.planet.H) # In km
    rat = (Rs/Rp).decompose().value # There is a rat in my code, oh no!
    Rp_km = Rp.to(u.km).value # Planet radius in km
    
    # Definition of units
    F_lam = u.def_unit(['','T','Transmitance','Transmitted flux'])
    R = u.def_unit(['','R','Excess atmospheric absorption', 'A'])
    R_plam =  u.def_unit(['Rp','Planetary radius'])
    delta_lam =  u.def_unit(['','Wavelength dependent transit depth'])
    
    # TODO:
        # Implement scale heights units
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
        wcs = add_spam_wcs(1)
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
def determine_rest_frame(spectrum: sp.Spectrum1D | sp.SpectrumCollection):
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
def print_info(spectrum_list):
    # DOCUMENTME
    # CLEANME
    for item in spectrum_list:
        print('Night:',item.meta['Night'],
              '#Night',item.meta['Night_num'],
              'Num_spec_night',item.meta['Night_spec_num'],
              'Num_spec',item.meta['spec_num'])

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
#%% prepare_const_spec
def prepare_const_spec(spec,num,unit=u.dimensionless_unscaled):
    """
    Prepares a constant spectrum with spectrl_axis of spec of the value num*unit

    Parameters
    ----------
    spec : sp.Spectrum1D
        Spectrum from which to take spectral_axis.
    num : float
        Value of the flux.
    unit : Unit, optional
        Unit of the flux spectrum. The default is u.dimensionless_unscaled.

    Returns
    -------
    con_spec : sp.Spectrum1D
        Output constant spectrum.

    """
    # CLEANME
    # TODO Check this is necessary anyway
    con_spec = sp.Spectrum1D(spectral_axis = spec.spectral_axis,
                             flux =[num]*len(spec.spectral_axis)*unit,
                             )
    return con_spec


#%% binning_spectrum
def binning_spectrum(spectrum,bin_factor = 10):
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
    # CLEANME
    # TESTME check this hasn't been changed
    num_bins = round(spectrum.spectral_axis.shape[0] / bin_factor)
    x = sci.stats.binned_statistic(spectrum.spectral_axis, spectrum.spectral_axis, statistic='mean', bins=num_bins)
    y = sci.stats.binned_statistic(spectrum.spectral_axis, spectrum.flux, statistic='mean', bins=num_bins)
    y_err = sci.stats.binned_statistic(spectrum.spectral_axis, spectrum.uncertainty.array, statistic=smcalc._error_bin, bins=num_bins)
    return x.statistic,y.statistic,y_err.statistic

#%% shift_spectrum
def shift_spectrum(spectrum, 
                   velocities:list
                   ):
    """
    # TODO
        Clean up the masking region handling
    Shifts wavegrid of spectrum by velocities list. Returns new shifted spectrum that has same sp.Spectrum1D.spectral_axis, but flux interpolated from the shifted spectrum. Uncertainty are interpolated the same way, but as np.sqrt(Interpolate(uncertainty**2))
    
    Input:
        spectrum ; sp.Spectrum1D - spectrum to shift
        velocities ; list - list of velocities to shift by
    Output:
        new_spectrum ; sp.Spectrum1D - shifted spectrum, interpolated to same wavelength_grid
            spectral_axis - same as spectrum
            flux - interpolated from the shifted spectrum to same spectral_axis
            uncertainty - interpolated as np.sqrt(Interpolate(uncertainty**2))
            meta - same meta as spectrum
            mask - new mask to ensure nans are included from extrapolation
    """
    # CLEANME
    # DOCUMENTME
    
    term = 1 # Initiation of the term (1 since we are dividing/multiplying the term)
    for velocity in velocities: # To factor all velocities (replace with relativistic one?)
        term = term / (1+velocity.to(u.m/u.s)/con.c) # Divide as the velocities are in opposite sign
    new_x_axis = spectrum.spectral_axis * term # Apply Doppler shift
    
    # Mask handling
    mask_flux = ~np.isfinite(spectrum.flux) # Ensure nans are not included
    mask_err = ~np.isfinite(spectrum.uncertainty.array) # Sometimes y_err is NaN while flux isnt? Possible through some divide or np.sqrt(negative)
    mask = mask_flux + mask_err # Gives zero values in each good pixel (values are False and False)
    mask = ~mask # Indices of good pixels (both flux and error)
    
    # CLEANME
    change_value = np.where(mask[:-1] != mask[1:])[0]
    mask_region_list = []
    for ind,value in enumerate(change_value):
        if ind == len(change_value)-1:
            break
        next_value = change_value[ind+1]
        if mask[value] and ~mask[value+1] and ~mask[next_value] and mask[next_value+1]:
            mask_region_list.append(sp.SpectralRegion(
                np.nanmean([new_x_axis[value].value,new_x_axis[value+1].value])*new_x_axis.unit
                ,
                np.nanmean([new_x_axis[next_value].value,new_x_axis[next_value+1].value])*new_x_axis.unit
                ))
        pass

    # Interpolation function for flux - cubic spline with no extrapolation
    flux_int = sci.interpolate.CubicSpline(new_x_axis[mask],
                                           spectrum.flux[mask],
                                           extrapolate= False)
    # Interpolation function for uncertainty - cubic spline with no extrapolation
    
    # Calculated with square of uncertainty, than final uncertainty is np.sqrt()
    err_int = sci.interpolate.CubicSpline(new_x_axis[mask],
                                           spectrum.uncertainty.array[mask]**2,
                                           extrapolate= False)
    # Applying interpolation functions to the old wave_grid
    # mask = ~mask # Indices of good pixels (both flux and error)
    new_flux = flux_int(spectrum.spectral_axis) # Interpolate on the old wave_grid
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_uncertainty = np.sqrt(err_int(spectrum.spectral_axis)) # Interpolate on the old wave_grid
    new_uncertainty = astropy.nddata.StdDevUncertainty(new_uncertainty) # Interpolate on the old wave_grid
    
    # Create new spectrum
    new_spectrum = sp.Spectrum1D(
        spectral_axis = spectrum.spectral_axis,
        flux = new_flux * spectrum.flux.unit,
        uncertainty = new_uncertainty,
        meta = spectrum.meta.copy(),
        mask = np.isnan(new_flux),
        wcs = spectrum.wcs,
        )
    if len(mask_region_list) !=0:
        for region in mask_region_list:
            new_spectrum = exciser_fill_with_nan(new_spectrum,region)
    
    return new_spectrum
#%% interpolate2commonframe
def interpolate2commonframe(spectrum,new_spectral_axis):
    """
    # TODO
        Clean up the masking region handling
    Interpolate spectrum to new spectral axis

    Parameters
    ----------
    spectrum : sp.Spectrum1D
        Spectrum to interpolate to new spectral axis
    new_spectral_axis : sp.Spectrum1D.spectral_axis 
        Spectral axis to intepolate the spectrum to

    Returns
    -------
    new_spectrum : sp.Spectrum1D
        New spectrum interpolated to given spectral_axis
        
    TODO:
        Check how does this function behaves with masks
        Don't interpolate for same wavelength grid

    """
    # CLEANME
    # DOCUMENTME
    # Mask handling
    mask_flux = ~np.isfinite(spectrum.flux) # Ensure nans are not included
    mask_err = ~np.isfinite(spectrum.uncertainty.array) # Sometimes y_err is NaN while flux isnt? Possible through some divide or np.sqrt(negative)
    mask = mask_flux + mask_err # Gives zero values in each good pixel (values are False and False)
    mask = ~mask # Indices of good pixels (both flux and error)
        
    change_value = np.where(mask[:-1] != mask[1:])[0]
    mask_region_list = []
    for ind,value in enumerate(change_value):
        if ind == len(change_value)-1:
            break
        next_value = change_value[ind+1]
        if mask[value] and ~mask[value+1] and ~mask[next_value] and mask[next_value+1]:
            mask_region_list.append(sp.SpectralRegion(
                np.nanmean([new_spectral_axis[value].value,new_spectral_axis[value+1].value])*new_spectral_axis.unit
                ,
                np.nanmean([new_spectral_axis[next_value].value,new_spectral_axis[next_value+1].value])*new_spectral_axis.unit
                ))
        pass

    # Interpolation function for flux - cubic spline with no extrapolation
    flux_int = sci.interpolate.CubicSpline(spectrum.spectral_axis[mask],
                                           spectrum.flux[mask],
                                           extrapolate= False)
    # Interpolation function for uncertainty - cubic spline with no extrapolation
    
    # Calculated with square of uncertainty, than final uncertainty is np.sqrt()
    err_int = sci.interpolate.CubicSpline(spectrum.spectral_axis[mask],
                                           spectrum.uncertainty.array[mask]**2,
                                           extrapolate= False)
    new_flux = flux_int(new_spectral_axis) # Interpolate on the old wave_grid
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_uncertainty = np.sqrt(err_int(new_spectral_axis)) # Interpolate on the old wave_grid
    new_uncertainty = astropy.nddata.StdDevUncertainty(new_uncertainty) # Interpolate on the old wave_grid
    new_spectrum = sp.Spectrum1D(
        spectral_axis = new_spectral_axis,
        flux = new_flux * spectrum.flux.unit,
        uncertainty = new_uncertainty,
        meta = spectrum.meta.copy(),
        mask = np.isnan(new_flux),
        wcs = spectrum.wcs,
        )
    
    if len(mask_region_list) !=0:
        for region in mask_region_list:
            new_spectrum = exciser_fill_with_nan(new_spectrum,region)
    
    
    return new_spectrum


#%% binning_list
@progress_tracker
@skip_function
def binning_list(spec_list: sp.SpectrumList,
                 force_skip:bool =False,) -> sp.SpectrumList:
    """
    Reinterpolates the spectrum list to same wavelength grid.

    Parameters
    ----------
    spec_list : sp.SpectrumList
        Spectrum list to interpolate to same wavelength grid.
    force_skip : bool, optional
        Whether to skip this function. Useful for loading output later in the pipeline. The default is False.

    Returns
    -------
    new_spec_list : sp.SpectrumList
        Spectrum list with common wavelength frame.

    """

    new_spec_list = sp.SpectrumList()
    new_spectral_axis = spec_list[0].spectral_axis
        
    # Run for loop via multiprocessing
    num_cores = multiprocessing.cpu_count()
    new_spec_list = sp.SpectrumList(Parallel(n_jobs=num_cores)(delayed(interpolate2commonframe)(i,new_spectral_axis) for i in spec_list))
    return new_spec_list

#%% normalize_spectrum
def normalize_spectrum(spectrum,quantile=.85,linfit=False):
    """
    Normalize spectrum depending on the size and linfit values.
    Normalization function is either rolling quantile window (with size of 7500 pixels), or linear fit
    
    Parameters
    ----------
    spectrum : sp.Spectrum1D
        Spectrum to normalize.
    quantile : Float [0;1], optional
        Quantile by which to normalize the spectrum. The default is .85.
    linfit : bool, optional
        Whether to fit by linear fit. Works only for spectra of length less than 10000. The default is False.

    Returns
    -------
    normalized_spectrum : sp.Spectrum1D
        Normalized spectrum based on parameters.

    """
    if (len(spectrum.flux) <10000) and linfit==True: 
        p = np.polyfit(spectrum.spectral_axis.value[~np.isnan(spectrum.flux)],
                         spectrum.flux.value[~np.isnan(spectrum.flux)],
                         1,
                         rcond=None,
                         full=False,
                         w=None,
                         cov=False
                         )
        tmp_spec = sp.Spectrum1D(
            spectral_axis = spectrum.spectral_axis,
            flux = (np.polyval(p,spectrum.spectral_axis.value))*pd.Series(spectrum.flux.value / np.polyval(p,spectrum.spectral_axis.value)).quantile(quantile)*spectrum.flux.unit,
            )
    elif (len(spectrum.flux) <10000) and linfit==False:
        tmp_spec = sp.Spectrum1D(
            spectral_axis = spectrum.spectral_axis,
            flux =np.full_like(spectrum.spectral_axis.value, pd.Series(spectrum.flux.value).fillna(999999).quantile(quantile))*spectrum.flux.unit,
            )
    else:
        tmp_spec = sp.Spectrum1D(spectral_axis = spectrum.spectral_axis,
                                 flux = np.array( pd.Series(spectrum.flux.value).rolling(7500
                                   ,min_periods=1,center=True).quantile(quantile))*spectrum.flux.unit,
                                  )
    
    normalization = spectrum.divide(tmp_spec,
                                    handle_mask = 'first_found',
                                    handle_meta = 'first_found',
                                    )
    normalization.spectral_axis.value.put(np.arange(len(spectrum.spectral_axis)),spectrum.spectral_axis)
    
    # Rewrite the spectral axis, as operations are done only on the flux array and WCS is dumb.
    normalized_spectrum = sp.Spectrum1D(spectral_axis = spectrum.spectral_axis,
                             flux = normalization.flux*u.dimensionless_unscaled,
                             uncertainty = normalization.uncertainty,
                             mask = spectrum.mask.copy(),
                             meta = spectrum.meta.copy(),
                              )
    normalized_spectrum.meta['normalization'] = True
    return normalized_spectrum

#%% normalize_list
@progress_tracker
@save_and_load
def normalize_list(spec_list,quantile=.85,linfit=False,force_load = False,force_skip=False):
    """
    Normalize a spectrum list by each spectrum's mean
    Input:
        spec_list ; sp.SpectrumList - list to normalize
    Output:
        new_spec_list ; sp.SpectrumList - resulting normalized list
    """
    
    # For looping through list via multiprocesses
    num_cores = multiprocessing.cpu_count()
    new_spec_list = sp.SpectrumList(Parallel(n_jobs=num_cores)(delayed(normalize_spectrum)(i,quantile,linfit) for i in spec_list))
    return new_spec_list



#%% replace_flux_units_transmission
@save_and_load
def replace_flux_units_transmission(spec_list, unit,force_load = False, force_skip= False, pkl_name = ''):
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
        cosmic_corrected = smcalc.cosmic_correction_night(sublist)
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
                          sn_type: None | str ='quadratic',
                          force_load: bool = False,
                          force_skip: bool = False,
                          pkl_name: str = ''
                          ) -> sp.SpectrumList:
    # DOCUMENTME
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
    
    spectrum_type = smcalc.get_spectrum_type(key= key,
                                             value= value)
    smcalc._check_night_ordering(spectrum_list)
    number_of_nights = sublist[-1].meta['Night_num']

    
    # FIXME Should have a division on different instruments
    
    # Master of nights combined
    master_list = sp.SpectrumList()
    master_all = smcalc.calculate_master_test(sublist,
                            spectrum_type,
                            night='nights-combined',
                            num_night = '"combined"',
                            rf = sublist[0].meta['RF'],
                            sn_type=sn_type)
    master_list.append(master_all)
    
    for ni in range(number_of_nights):
        # Getting night data
        sublist = get_sublist(spectrum_list,'Night_num',ni+1)
        if len(sublist) == 0:
            logger.warning(f'No spectra were found for given key/value selection for night {ni}. This night will be skipped and the resulting master list will not have a spectrum for this night.')
            continue
        
        master_night = smcalc.calculate_master_test(
            spectrum_list= sublist,
            spectrum_type= spectrum_type,
            night= sublist[0].meta['Night'],
            num_night= str(ni+1),
            rf= sublist[0].meta['RF'],
            sn_type= sn_type
            )
        master_list.append(master_night)

    return master_list

#%% rm_correction
def rm_correction(spec,master,star_para):
    """
    Corrects spectrum for RM_effect
    Equation used (latex syntax):
        F_{corr} = 1 + (\frac{R_p^2}{R_s^2}) (1 - \frac{M - F_i \delta_i}{M_{i,v_i}\Delta_i})
    
    Input:
        spec ; sp.Spectrum1D , RF_star
        master ; sp.Spectrum1D, out-of-transit spectrum
        star_para ; Star_para class with parameters
    Output:
        corrected_spectrum ; sp.Spectrum1D, corrected spectrum for RM
    """
    # Defining all necessary variables
    vel_RM = spec.meta['vel_stel_loc']
    # vel_RM = -spec.meta['vel_stel_loc']
    lc_flux = spec.meta['light_curve_flux']
    delta = spec.meta['delta']
    scaling_factor = (star_para.planet.radius **2 / star_para.star.radius**2).decompose()
    meta_new = spec.meta.copy()
    mask_new = spec.mask.copy()
    spec_axis = spec.spectral_axis
    # Creating both shifted and unshifted master
    master_non_shifted = deepcopy(master)
    master_shifted = shift_spectrum(master, 
                       velocities = [vel_RM]
                       )
    # Binning shifted spectrum to same wave_grid
    spec_list = sp.SpectrumList([spec,master_non_shifted,master_shifted])
    new_spec_list = binning_list(spec_list)
    spec,master_non_shifted,master_shifted = new_spec_list[0],new_spec_list[1],new_spec_list[2]
    """
    Equation for RM_correction (works with latex)
    
    F_{corr} = 1 + (\frac{R_p^2}{R_s^2}) (1 - \frac{M - F_i \delta_i}{M_{i,v_i}\Delta_i})
    
    
    """
    tmp_spec = prepare_const_spec(spec,lc_flux)
    numerator = master_non_shifted.subtract(spec*tmp_spec)
    
    # err_numerator = np.sqrt( master_non_shifted.uncertainty.array **2 + (spec.uncertainty.array*lc_flux)**2)
    # print('Numerator:',np.nansum(numerator.uncertainty.array - err_numerator))
    
    tmp_spec = prepare_const_spec(master_shifted,delta)
    denominator = master_shifted.multiply(tmp_spec)
    # err_denominator = master_shifted.uncertainty.array * delta
    # print('Denominator',np.nansum(denominator.uncertainty.array - err_denominator))
    
    
    fraction = numerator.divide(denominator)
    # err_fraction=  np.sqrt( (numerator.uncertainty.array / denominator.flux.value) **2 +\
    #                         (numerator.flux.value / denominator.flux.value**2 *\
    #                           denominator.uncertainty.array
    #                           )**2)
    # print('Fraction:',np.nansum(fraction.uncertainty.array - err_fraction))
    
    tmp_spec = prepare_const_spec(fraction,-1 * u.dimensionless_unscaled)
    reverse = fraction.multiply(tmp_spec)
    # err_reverse = fraction.uncertainty.array 
    # print(np.nansum(reverse.uncertainty.array - err_reverse))
    
    tmp_spec = prepare_const_spec(reverse,1 * u.dimensionless_unscaled)
    reverse_plus_1 = reverse.add(tmp_spec)
    # err_reverse_plus_1 = reverse_plus_1.uncertainty.array 
    # print(np.nansum(reverse_plus_1.uncertainty.array - err_reverse_plus_1))
    
    tmp_spec = prepare_const_spec(reverse,scaling_factor)
    scaled = reverse_plus_1.multiply(tmp_spec)
    # err_scaled = reverse_plus_1.uncertainty.array * scaling_factor
    # print(np.nansum(scaled.uncertainty.array - err_scaled))
    
    
    tmp_spec = prepare_const_spec(scaled,1 * u.dimensionless_unscaled)
    corrected_spectrum = scaled.add(tmp_spec)
    # err_corrected_spectrum = corrected_spectrum.uncertainty.array
    # print(np.nansum(corrected_spectrum.uncertainty.array - err_corrected_spectrum))
    
    
    # Assigning mask and meta values to spectrum
    corrected_spectrum.mask = mask_new
    corrected_spectrum.meta = meta_new
    corrected_spectrum.meta['RM_corrected'] = True
    
    rm_corrected_spectrum = sp.Spectrum1D(
        spectral_axis = spec_axis,
        flux = corrected_spectrum.flux,
        uncertainty = corrected_spectrum.uncertainty,
        meta = corrected_spectrum.meta.copy(),
        mask = mask_new,
        wcs = add_spam_wcs(1)
        
        )
    
    return rm_corrected_spectrum

#%% spec_list_master_correct
@progress_tracker
@save_and_load
def spec_list_master_correct(spec_list,master,force_load=False, force_skip=False,pkl_name=''):
    """
    Corrects spec_list by master (spec/master)
    Input:
        spec_list ; sp.SpectrumList to correct
        master ; sp.SpectrumList with masters divided by night
    Output:
        corrected_list ; sp.SpectrumList corrected by master
    """
    corrected_list = sp.SpectrumList()
    for item in spec_list:
        num_night = item.meta['Night_num']
        master_night = master[num_night]
        divided_spectrum = item.divide(master_night)
        
        corrected_spectrum =sp.Spectrum1D(
            spectral_axis = item.spectral_axis,
            flux = divided_spectrum.flux,
            uncertainty = divided_spectrum.uncertainty,
            mask = item.mask.copy(),
            meta = item.meta.copy(),
            wcs = item.wcs,
            )

        corrected_list.append(corrected_spectrum)
    return corrected_list

#%% rm_list_correct
def rm_list_correct(spec_list,master,star_para):
    """
    Corrects spec_list for RM effect
    Input:
        spec_list ; sp.SpectrumList to correct for RM
        master ; sp.SpectrumList containing master divided by night
        star_para ; Star_para class containing stellar parameters (P_R,S_R necessary)
    Output:
        rm_list_corrected ; sp.SpectrumList corrected by RM
    """

    sublist = get_sublist(spec_list,'RM_velocity',True)
    rm_list_corrected = sp.SpectrumList()
    entire_list = spec_list.copy()
    master_RF_star_oot = calculate_master_list(spec_list,key = 'Transit',value =False)
    entire_list = spec_list_master_correct(spec_list,master_RF_star_oot)
    for item in sublist:
        num_night = item.meta['Night_num']
        master_night = master[num_night]
        corrected_spectrum = rm_correction(item,master_night,star_para)
        rm_list_corrected.append(corrected_spectrum)
        for ind,ent_item in enumerate(entire_list):
            if ent_item.meta['spec_num'] == corrected_spectrum.meta['spec_num']:
                entire_list[ind] = corrected_spectrum

    return rm_list_corrected,entire_list

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

#%% flux2height
@disable_func
def flux2height_todo(spec_list,star_para):
    """
    Changes transmission spectrum to [R_p]
    Input: 
        spec_list ; sp.SpectrumList - list of spectra
        star_para ; lb.Star_para - stellar parameters
        
    Output:
        new_spec_list ; sp.SpectrumList - list of spectra scaled in [R_p]
    """
    new_spec_list = sp.SpectrumList()
    
    sp_ratio = (star_para.star.radius / star_para.planet.radius).decompose()
    ps_ratio = (star_para.planet.radius / star_para.star.radius).decompose()
    
    for item in spec_list:
        flux = item.flux.value
        uncertainty = item.uncertainty.array
        
        new_flux = sp_ratio * np.sqrt(1 - flux + (ps_ratio)**2 * flux)
        sqrt_inside = 1- flux + (ps_ratio)**2 * flux
        
        new_uncertainty = (((uncertainty)**2 + (uncertainty* (ps_ratio)**2)**2) * (1/2) / sqrt_inside)**(1/2) * sp_ratio * new_flux
        new_spec_list.append(
            sp.Spectrum1D(flux = new_flux,
                             spectral_axis = item.spectral_axis,
                             uncertainty = astropy.nddata.StdDevUncertainty(new_uncertainty),
                             mask = item.mask.copy(),
                             meta = item.meta.copy())
                                 )
    
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
        velocity_field.append(spectrum.meta['BERV'] * shift_BERV)
    if shift_v_sys != 0:
        velocity_field.append(spectrum.meta['vel_sys'] * shift_v_sys)
    if shift_v_star != 0:
        velocity_field.append(spectrum.meta['vel_st'] * shift_v_star)
    if shift_v_planet != 0:
        velocity_field.append(spectrum.meta['vel_pl'] * shift_v_planet)
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
    new_spectrum = shift_spectrum(spectrum,velocity_field)
    return new_spectrum
    

#%% shift_list
@progress_tracker
@save_and_load
def shift_list(spec_list_to_shift:sp.SpectrumList,
                   shift_BERV:float, # Sensible values = 1/0/-1, otherwise BERV is scaled 
                   shift_v_sys:float, # Sensible values = 1/0/-1, necessary for spectrum to include 'vel_sys' parameter
                   shift_v_star:float, # Sensible values = 1/0/-1, otherwise scaled
                   shift_v_planet:float, # Sensible values = 1/0/-1, otherwise scaled
                   shift_constant=0, # Value in units u.m/u.s (or equivalent)
                   force_load = False,
                   force_skip = False,
                   pkl_name = '',
                   ):
    """
    Function to shift list of spectra by velocity list
    Input:
        spec_list ; sp.SpectrumList - to shift
        shift_* ; float - if we want to shift by given keyword, with + or - sign
            shift_BERV - Shifting by BERV
            shift_v_sys - Shifting by systemic velocity
            shift_v_star - Shifting by stellar velocity
            shift_v_planet - Shifting by planetary velocity
        shift_constant ; float * u.m/u.s - Additional constant shift

    Output:
        new_spec_list ; sp.SpectrumList - already shifted list of spectra
            Interpolated to the same wavelength grid[]
    """
    # Deepcopying spec_list
    spec_list = deepcopy(spec_list_to_shift)
    # For cycle that shifts the spectral_axis by given list of velocities
    num_cores = multiprocessing.cpu_count()
    new_spec_list = sp.SpectrumList(Parallel(n_jobs=num_cores)(delayed(shift_spectrum_multiprocessing)(item,
                               shift_BERV,
                               shift_v_sys,
                               shift_v_star,
                               shift_v_planet,
                               shift_constant) for item in spec_list))
    return new_spec_list
#%% sort_spec_list
@skip_function
def sort_spec_list(spec_list,force_skip=False):
    """
    Sorts spectrum list based on BJD time observation in case the order got mingled
    
    Input:
        spec_list ; sp.SpectrumList - spectrum list to order
    Output:
        new_spec_list ; sp.SpectrumList - spectrum list ordered by BJD
    """
    bjd = []
    for item in spec_list:
        bjd.append(item.meta['BJD'].value)
    ind = np.argsort(bjd)
    new_spec_list = sp.SpectrumList()
    for number,ii in enumerate(ind):
        new_spec_list.append(spec_list[ii])
        new_spec_list[number].meta['spec_num'] = number
    return new_spec_list
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
        wcs = add_spam_wcs(1)
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
            wcs = add_spam_wcs(1)
            )
    return spec_list_velocity, new_spectrum
#%% RM_simulation
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
            shifted = shift_spectrum(master_RF_star_oot,
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
            shifted_to_RFP = shift_spectrum(new_shifted_1,
                                velocities= [-item.meta['vel_st'],item.meta['vel_pl']]
                                )
            shifted_to_RFP.meta = item.meta.copy()
            
            RM_simulated.append(shifted_to_RFP)
            
    # Calculate master in transit
    In_transit = smcalc.calculate_master_test(RM_simulated,sn_type = 'quadratic_combined')
    return RM_simulated,In_transit

#%% non_scaled2RpRssquare
def non_scaled2RpRssquare(spectrum,sys_para):
    """
    Rescaling the unscaled transmission spectrum (1-delta) to RpRs**2
    Input:
        spectrum ; sp.Spectrum1D - spectrum to rescale
        sys_para ; para.system_parameters_class - system parameters
    Output:
        rescaled_spectrum ; sp.Spectrum1D - spectrum rescaled to RpRs**2
    """
    reversed_spectrum = spectrum.multiply(-1*u.dimensionless_unscaled,handle_meta = 'first_found',handle_mask = 'first_found')
    reversed_add_1 = reversed_spectrum.add(1*u.dimensionless_unscaled,handle_meta = 'first_found',handle_mask = 'first_found')
    multiply_rprssquare = reversed_add_1.multiply((sys_para.planet.radius**2 / sys_para.star.radius**2).decompose()*u.dimensionless_unscaled
        ,handle_meta = 'first_found',handle_mask = 'first_found')
    rescaled_spectrum = multiply_rprssquare.divide((sys_para.transit.delta*0.01)*u.dimensionless_unscaled
        ,handle_meta = 'first_found',handle_mask = 'first_found')
    return rescaled_spectrum

#%% norm2RlamRp
def norm2RlamRp(flux,sys_para):
    """
    Rescaling normalized flux to R_lambda / R_p
    Latex equation:
        \frac{R_\lambda}{R_p} = \frac{R_s}{R_p}\sqrt{1-F_\lambda + \frac{R_p^2}{R_s^2}F_\lambda}
    """
    return (sys_para.star.radius/sys_para.planet.radius).decompose() * np.sqrt(1- flux + (sys_para.planet.radius**2/sys_para.star.radius**2).decompose()*flux)

#%% norm2RlamRssquare
def norm2RlamRssquare(flux,sys_para):
    """
    Rescaling normalized flux to R_lambda / R_star squared
    """
    return 1- flux + (sys_para.planet.radius**2/sys_para.star.radius**2).decompose()*flux
#%% square_root
def square_root(flux,uncertainty):
    """
    Calculates square root and uncertainty of flux
    """
    new_flux = np.sqrt(flux)
    new_uncertainty = (1/2 * uncertainty.array/flux) *new_flux
    
    return new_flux,new_uncertainty

#%%
def spec_norm2RlamRp(spectrum,sys_para):
    """
    Rescaling the spectrum to R_lambda / R_p_wlc from normalized spectrum
    Latex equation:
        \frac{R_\lambda}{R_p} = \sqrt{\frac{R_s}{R_p}-\frac{R_s}{R_p}F_\lambda + F_\lambda}
    
    Input:
        spectrum ; sp.Spectrum1D - spectrum to rescale
        sys_para ; para.system_parameters_class - system parameters
        
    Output:
        rescaled_spectrum ; sp.Spectrum1D - spectrum rescaled to R_lambda / R_p_wlc
    """
    
    term_in_sqrt_flux = spectrum.multiply( # Terms with F_\lambda
        (1-(sys_para.planet.radius**2/sys_para.star.radius**2).decompose())
        )
    
    term_in_sqrt = term_in_sqrt_flux.add((sys_para.planet.radius**2/sys_para.star.radius**2).decompose())
    new_flux,new_uncertainty = square_root(term_in_sqrt.flux,term_in_sqrt.uncertainty)
    
    new_spectrum = sp.Spectrum1D(
        spectral_axis= spectrum.spectral_axis,
        flux = new_flux,
        uncertainty = astropy.nddata.StdDevUncertainty(new_uncertainty),
        meta = spectrum.meta.copy(),
        mask = np.isnan(new_flux)
        )
    return new_spectrum
#%%
def RpRs2Rp(x,sys_para):
    return np.sign(x)*np.sqrt(np.sign(x)*x)*(sys_para.star.radius / sys_para.planet.radius).decompose()

def Rp2RpRs(x,sys_para):
    return (np.sign(x)*x/(sys_para.star.radius / sys_para.planet.radius).decompose())**2

#%% spec_norm2RlamRssquare
def spec_norm2RlamRssquare(spectrum,sys_para):
    """
    Rescaling the spectrum to R_lambda / R_p_wlc from normalized spectrum
    Latex equation:
        \frac{R_\lambda^2}{R_s^2} = 1-F_\lambda + F_\lambda \frac{R_p^2}{R_s^2}
    
    Input:
        spectrum ; sp.Spectrum1D - spectrum to rescale
        sys_para ; para.system_parameters_class - system parameters
        
    Output:
        rescaled_spectrum ; sp.Spectrum1D - spectrum rescaled to R_lambda / R_p_wlc
    """
    
    tmp_spec = prepare_const_spec(spectrum, -1*u.dimensionless_unscaled)
    first_term = spectrum.multiply( # Terms with F_\lambda
        tmp_spec
        )
    
    tmp_spec = prepare_const_spec(spectrum,(sys_para.planet.radius**2/sys_para.star.radius**2).decompose())
    second_term = spectrum.multiply(
        tmp_spec
        )
    
    tmp_spec = prepare_const_spec(first_term, 1*u.dimensionless_unscaled)
    new_spectrum = first_term.add(second_term).add(tmp_spec)
    new_spectrum = replace_spectral_axis(new_spectrum, spectrum.spectral_axis)
    return new_spectrum

#%% lost_planetary_signal
def lost_planetary_signal(sr_region,spec_list):
    """
    Calculate fraction of lost (NaN) pixels in spectral region 
    """
    # TODO - Rewrite
    num_pixels = 0
    num_pixels_lost = 0

    for item in spec_list:
        region_spec = extract_region_in_spectrum(item, sr_region)
        try:
            num_pixels += len(region_spec.flux)
            num_pixels_lost += sum(np.isnan(region_spec.flux))
        except:
            for subspec in region_spec:
                num_pixels += len(subspec.flux)
                num_pixels_lost += sum(np.isnan(subspec.flux))
    print('Lost signal:',num_pixels_lost/num_pixels)
    
    return num_pixels_lost/num_pixels
