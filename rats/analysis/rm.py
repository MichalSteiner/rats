#%% Importing libraries
import astropy
from rats.utilities import default_logger_format, save_and_load, time_function
import specutils as sp
import numpy as np
import logging
import astropy.units as u
import pymc as pm

#%% Setup logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

#%% Fitting a Gaussian profile to the 
def _fit_gaussian_profile(CCF: sp.Spectrum1D) -> astropy.modeling.core.CompoundModel:
    """
    Fit a Gaussian profile to the CCF output.

    Parameters
    ----------
    CCF : sp.Spectrum1D
        CCF to fit a Gaussian profile to.

    Returns
    -------
    Gaussian_fit : astropy.modeling.core.CompoundModel
        Gaussian fit as fitted by astropy package.
    """
    
    Gaussian_model = astropy.modeling.models.Gaussian1D(
        amplitude=-np.nanmedian(CCF.flux),
        mean=CCF.spectral_axis.mean(),
        stddev=5 *u.km/u.s) + astropy.modeling.models.Const1D(
            amplitude=np.nanmedian(CCF.flux)
            )
        
    gaussian_fitter = astropy.modeling.fitting.LevMarLSQFitter(calc_uncertainties=True)
    
    finite_indices = np.isfinite(CCF.flux)
    Gaussian_fit = gaussian_fitter(Gaussian_model,
                         CCF.spectral_axis[finite_indices],
                         CCF.flux[finite_indices],
                         weights= 1.0/CCF.uncertainty.array[finite_indices]
                         )

    if gaussian_fitter.fit_info['param_cov'] is not None:
        fit_uncertainty = np.sqrt(np.diag(gaussian_fitter.fit_info['param_cov']))
    else:
        fit_uncertainty = [np.nan, np.nan, np.nan, np.nan]
    return Gaussian_fit, fit_uncertainty

#%% Normalization of the CCF by continuum by fitting a Gaussian + constant profile
def normalize_by_continuum(CCF: sp.Spectrum1D) -> sp.Spectrum1D:
    """
    Normalize a CCF by its continuum as fitted by astropy.

    Parameters
    ----------
    CCF : sp.Spectrum1D
        CCF to normalize.

    Returns
    -------
    CCF_normalized : sp.Spectrum1D
        Normalized CCF by its continuum
    """
    Gaussian_fit, _ = _fit_gaussian_profile(CCF)
    
    CCF_normalized = CCF.divide(Gaussian_fit.amplitude_1.value * Gaussian_fit.amplitude_1.unit, handle_meta='first_found')
    CCF_normalized.meta['normalization'] = True
    return CCF_normalized

#%% Normalize the full CCF list
@time_function
def normalize_CCF_list(CCF_list: sp.SpectrumList,
                       scale_by_transit_depth: bool = True) -> sp.SpectrumList:
    """
    Normalize CCF list by its continuum.

    Parameters
    ----------
    CCF_list : sp.SpectrumList
        CCF list to normalize by continuum.
    scale_by_transit_depth : bool
        Whether to scale by transit depth, by default True. If true, the out-of-transit CCF are scaled to 1, the in transit spectra are scaled to 1 - delta(t), calculated based on the BJD time stamp of each exposure.

    Returns
    -------
    CCF_list_normalized : sp.SpectrumList
        Normalized list of CCF functions.
    """
    if scale_by_transit_depth == False:
        raise NotImplementedError('By default, the normalization is always scaling')
    
    CCF_list_normalized = sp.SpectrumList(
        [
            normalize_by_continuum(
                CCF
                ).multiply(
                    CCF.meta['light_curve_flux']*u.dimensionless_unscaled,
                    handle_meta='first_found') for CCF in CCF_list
        ]
    )
    return CCF_list_normalized

def _remove_systemic_velocity(data: sp.SpectrumList,
                              system_parameters) -> sp.SpectrumList:
    
    new_spectrum_list = sp.SpectrumList()
    
    for spectrum in data:
        new_spectrum_list.append(
            sp.Spectrum1D(
                spectral_axis= spectrum.spectral_axis -system_parameters.System.systemic_velocity.data*system_parameters.System.systemic_velocity.unit,
                flux= spectrum.flux,
                uncertainty= spectrum.uncertainty,
                meta= spectrum.meta,
            )
        )
    return new_spectrum_list
#%% Subtraction of master out
def subtract_master_out(CCF_list: sp.SpectrumList,
                        master_out_list: sp.SpectrumList) -> [sp.SpectrumList, sp.SpectrumList]:
    """
    Subtract master out from the CCF list.
    
    This in principle creates a list of residual CCF's. Furthermore, the list is also divided by the transit depth to create a intrinsic CCF's list.

    Parameters
    ----------
    CCF_list : sp.SpectrumList
        List of normalized CCF's
    master_out_list : sp.SpectrumList
        Master out list of the CCF.

    Returns
    -------
    CCF_residual : sp.SpectrumList,
        List of residual CCF's. These are used in the Reloaded method of RM.
    CCF_intrinsic : sp.SpectrumList
        List of intrinsic CCF's. These are used in the Revolutions method of RM.
    """
    
    CCF_residual = sp.SpectrumList(
        [
            CCF.subtract(
                master_out_list[CCF.meta['Night_num']],
                handle_meta='first_found'
                ).multiply(
                    -1*u.dimensionless_unscaled,
                    handle_meta='first_found'
                    ) for CCF in CCF_list
        ]
    )
    
    CCF_intrinsic = sp.SpectrumList()
    for CCF in CCF_list:
        if CCF.meta['delta'] != 0:
            CCF_intrinsic.append(
                CCF.subtract(
                    master_out_list[CCF.meta['Night_num']],
                    handle_meta='first_found'
                    ).multiply(
                        -1*u.dimensionless_unscaled,
                        handle_meta='first_found'
                        ).divide(
                            CCF.meta['delta']*u.dimensionless_unscaled,
                            handle_meta='first_found'
                            )
            )
        else:
            CCF_intrinsic.append(
                CCF.subtract(
                    master_out_list[CCF.meta['Night_num']],
                    handle_meta='first_found'
                    ).multiply(
                        -1*u.dimensionless_unscaled,
                        handle_meta='first_found'
                        ).add(
                            1*u.dimensionless_unscaled,
                            handle_meta='first_found'
                            )
            )
    return CCF_residual, CCF_intrinsic

#%% Model local intrinsic CCF's
def _MCMC_model_local_CCF(CCF_local: sp.Spectrum1D,
                         draws: int = 15000,
                         tune: int = 15000,
                         chains: int = 15,
                         target_accept: float = 0.99,
                         **kwargs_pymc,
                         ):
    
    assert CCF_local.meta['Transit_partial'], 'Spectrum is not transiting'
    
    ind = np.logical_and(np.isfinite(CCF_local.flux),
                         np.isfinite(CCF_local.uncertainty.array))
    
    lower_bound = CCF_local.spectral_axis[0].value
    upper_bound = CCF_local.spectral_axis[-1].value
    spread_CCF = (upper_bound - lower_bound)/2
    
    local_CCF_model = pm.Model()
    with local_CCF_model:
        contrast = pm.Uniform('contrast',
                              lower=-2,
                              upper=2
                              )
        
        fwhm = pm.Uniform('fwhm',
                          lower= 0,
                          upper= spread_CCF
                          )
        
        rvcenter = pm.Uniform('rvcenter',
                              lower= lower_bound,
                              upper= upper_bound
                              )
        
        # Expecting a Gaussian profile
        expected = (-contrast *
                    np.exp(-((CCF_local.spectral_axis[ind].value-rvcenter)**2)/(2*((fwhm/2.35482)**2))) + 1) # fwhm to sigma is approx fwhm = 2.35482*sigma
        
        # Observed, expecting Gaussian, sigma = uncertainty, observed = flux
        observed = pm.Normal("observed",
                             mu=expected,
                             sigma= CCF_local.uncertainty[ind].array,
                             observed=CCF_local.flux[ind].value
                             )
        
        idata = pm.sample(
            draws= draws,
            tune= tune,
            chains= chains,
            target_accept=target_accept,
            **kwargs_pymc
        )
    return idata
#%% Model CCF's distribution in list
@save_and_load
def MCMC_model_local_CCF_list(
    CCF_list: sp.SpectrumList,
    force_load: bool = True,
    force_skip: bool = False,
    pkl_name: str = 'distribution_local_CCF.pkl') -> list:
    """
    Calculates the posterior distribution of intrinsic CCF by MCMC routine and fitting a Gaussian profile.

    Parameters
    ----------
    CCF_list : sp.SpectrumList
        CCF list to which to calculate the posteriors.
    force_load : bool, optional
        Whether to force loading the result of the function, by default True. This will not rerun the MCMC and instead only load the data, if available
    force_skip : bool, optional
        Whether to force skipping the function, by default False.
    pkl_name : str, optional
        Location of the saved output, by default 'distribution_local_CCF.pkl'. It will be saved in the './saved_data/ directory

    Returns
    -------
    data_chain : list
        List of idata as calculated by the PyMC. For more info about idata structure check PyMC and arviz python documentation.
    """
    
    
    data_chain = []
    for CCF_local in CCF_list:
        if not(CCF_local.meta['Transit_partial']):
            continue
        data_chain.append(_MCMC_model_local_CCF(CCF_local))
    
    return data_chain


def wrapper_RM_Revolutions(data_raw_A: sp.SpectrumList):
    
    data_normalized = normalize_CCF_list(data_raw_A)
    # Shift list to stellar rest frame
    import rats.spectra_manipulation as sm
    
    data_SRF = sm.shift_list(data_normalized,
                            shift_BERV=0,
                            shift_v_sys = 1,
                            shift_v_star = 1,
                            shift_v_planet = 0,
                            force_load = False,
                            force_skip = False,
                            force_multiprocessing= False,
                            pkl_name = 'data_SRF_RM_CCF.pkl'
                            )
    
    master_SRF_out = sm.calculate_master_list(data_SRF,
                                          key = 'Transit_partial',
                                          value =False,
                                          force_load = force_load,
                                          force_skip = force_skip,
                                          pkl_name = 'master_out_SRF_RM_CCF.pkl'
                                          )
    CCF_residual, CCF_intrinsic = subtract_master_out(data_SRF ,master_SRF_out)
    
    data_chain = MCMC_model_local_CCF_list(CCF_intrinsic)
    ...