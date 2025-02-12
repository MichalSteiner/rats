"""
This module provides functions for analyzing the Rossiter-McLaughlin effect.

It includes functions for fitting Gaussian profiles, normalizing CCFs, calculating systemic velocities, subtracting systemic velocities, 
subtracting master out spectra, and modeling local intrinsic CCFs using MCMC routines.

Functions:
- _fit_gaussian_profile(CCF: sp.Spectrum1D) -> astropy.modeling.core.CompoundModel:

- normalize_by_continuum(CCF: sp.Spectrum1D) -> sp.Spectrum1D:

- normalize_CCF_list(CCF_list: sp.SpectrumList, scale_by_transit_depth: bool = True) -> sp.SpectrumList:

- calculate_systemic_velocity(master_SRF_out: sp.SpectrumList, data_SRF: sp.SpectrumList | None = None) -> None:

- remove_systemic_velocity(spectrum_list: sp.SpectrumList) -> sp.SpectrumList:

- subtract_master_out(CCF_list: sp.SpectrumList, master_out_list: sp.SpectrumList) -> tuple[sp.SpectrumList, sp.SpectrumList]:

- calculate_residual_CCF(CCF_list: sp.SpectrumList, master_out_list: sp.SpectrumList) -> sp.SpectrumList:

- _MCMC_model_local_CCF(CCF_local: sp.Spectrum1D, draws: int = 15000, tune: int = 15000, chains: int = 15, target_accept: float = 0.99, **kwargs_pymc):
    Model local intrinsic CCFs using MCMC routine.

- MCMC_model_local_CCF_list(CCF_list: sp.SpectrumList, draws: int = 15000, tune: int = 15000, chains: int = 15, force_load: bool = True, force_skip: bool = False, pkl_name: str = 'distribution_local_CCF.pkl', **kwargs_pymc) -> list:

- _extract_values_in_transit(CCF_list):
    Extracts wavelength, flux, uncertainty, and phase values for in-transit CCFs.

- add_contrast_model(model, CCF_intrinsic_list, order: int | None):
    Adds contrast model to the given PyMC model.

- add_fwhm_model(model, CCF_intrinsic_list, order: int | None):
    Adds FWHM model to the given PyMC model.

- add_observed_data(model, CCF_intrinsic_list):
    Adds observed data to the given PyMC model.

- generate_polynomials_per_night(model, coefficients, polynomial, key, order, night, phase):
    Generates polynomials for each night based on the given coefficients and order.

- add_asymmetry_factor():
    Adds asymmetry factor to the model.

- MCMC_model_Revolutions(system_parameters, CCF_intrinsic_list: sp.SpectrumList, contrast_order: int | None = 0, fwhm_order: int | None = 0, contrast_phase: str = 'Phase', fwhm_phase: str = 'Phase', draws: int = 15000, tune: int = 15000, force_skip: bool = False, force_load: bool = True, pkl_name: str = ''):
    Models the Revolutions method of RM using MCMC.

- set_valid_and_invalid_indices(CCF_intrinsic_filtered, indices):
    Sets valid and invalid indices for the CCF intrinsic list.

    
"""

# %% Importing libraries
# Built-in libraries
import logging

# External packages
import arviz
import astropy
import astropy.units as u
import numpy as np
import pymc as pm
import specutils as sp

# Local rats package
from rats.utilities import default_logger_format, save_and_load, time_function
import rats.spectra_manipulation as sm

# %% Setup logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

# %% Fitting a Gaussian profile to the


def _fit_gaussian_profile(CCF: sp.Spectrum1D) -> astropy.modeling.core.CompoundModel:  # type: ignore
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

    Gaussian_model = astropy.modeling.models.Gaussian1D(  # type: ignore
        amplitude=-np.nanmedian(CCF.flux),
        mean=CCF.spectral_axis.mean(),
        stddev=5 * u.km/u.s) + astropy.modeling.models.Const1D(  # type: ignore
            amplitude=np.nanmedian(CCF.flux)
    )
    if abs(Gaussian_model.mean_0.value) < 1E-4:
        Gaussian_model.mean_0.value = 1E-4

    gaussian_fitter = astropy.modeling.fitting.LevMarLSQFitter(  # type: ignore
        calc_uncertainties=True)

    finite_indices = np.isfinite(CCF.flux)
    Gaussian_fit = gaussian_fitter(Gaussian_model,
                                   CCF.spectral_axis[finite_indices],
                                   CCF.flux[finite_indices],
                                   weights=1.0 /
                                   CCF.uncertainty.array[finite_indices]
                                   )

    if gaussian_fitter.fit_info['param_cov'] is not None:
        fit_uncertainty = np.sqrt(
            np.diag(gaussian_fitter.fit_info['param_cov']))
    else:
        fit_uncertainty = [np.nan] * len(Gaussian_model.parameters)
    return Gaussian_fit, fit_uncertainty

# %% Normalization of the CCF by continuum by fitting a Gaussian + constant profile


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

    if Gaussian_fit.amplitude_1.value != 0:
        CCF_normalized = CCF.divide(
            Gaussian_fit.amplitude_1.value * Gaussian_fit.amplitude_1.unit, handle_meta='first_found')
    else:
        raise ValueError(
            "Gaussian fit amplitude is zero, cannot normalize CCF.")
    CCF_normalized.meta['normalization'] = True
    return CCF_normalized

# %% Normalize the full CCF list


def normalize_CCF_list(
    CCF_list: sp.SpectrumList,
    scale_by_transit_depth: bool = True
) -> sp.SpectrumList:
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
    """
    if scale_by_transit_depth == False:
        CCF_list_normalized = sp.SpectrumList(
            [normalize_by_continuum(CCF) for CCF in CCF_list])
    else:
        if not all('light_curve_flux' in CCF.meta for CCF in CCF_list):
            raise KeyError(
                "Not all CCFs have 'light_curve_flux' in their metadata")
        CCF_list_normalized = sp.SpectrumList([
            normalize_by_continuum(CCF).multiply(
                CCF.meta['light_curve_flux'] * u.dimensionless_unscaled, handle_meta='first_found'
            ) for CCF in CCF_list
        ])
    return CCF_list_normalized
# %% Systemic velocity calculation


def calculate_systemic_velocity(master_SRF_out: sp.SpectrumList,
                                data_SRF: sp.SpectrumList | None = None):
    """
    Calculates systemic velocity from the master out spectrum.

    Parameters
    ----------
    master_SRF_out : sp.SpectrumList
        Master out spectra in rest frame of the star
    data_SRF : sp.SpectrumList | None
        Spectrum list to input the systemic velocity in. If None, this part of the function is ignored. 
    """

    logger.print('Calculating systemic velocity:')  # type: ignore
    logger.print('='*50)  # type: ignore
    for ind, night in enumerate(master_SRF_out):
        results, uncertainty = _fit_gaussian_profile(night)

        logger.print(
            f'Systemic velocity for night {night.meta["night"]}: {results.mean_0.value} {results.mean_0.unit} ± {uncertainty[1]}')

        if data_SRF is None:
            continue
        for item in data_SRF:
            if item.meta['Night'] == night.meta['night']:
                item.meta['velocity_system'] = astropy.nddata.NDDataArray(
                    results.mean_0.value, unit=results.mean_0.unit)  # type: ignore
# %% Subtract systemic velocity


def remove_systemic_velocity(spectrum_list: sp.SpectrumList) -> sp.SpectrumList:
    """
    Removes systemic velocity from the data.

    This needs to be a separate function from shift spectrum, as CCF's are often defined on too narrow region. 
    In particular, the spectral axis changes in this function, unlike the shift_spectrum function.
    The algorithm is just literal subtraction of the "velocity_system" keyword from the meta of each spectrum in data.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Data from which to remove systemic velocity. Each spectrum in the list should have a 'velocity_system' entry in its metadata, which is expected to be of type `astropy.nddata.NDDataArray`.

    Returns
    -------
    new_spectrum_list : sp.SpectrumList
        Shifted spectrum list without the systemic velocity

    Raises
    ------
    KeyError
        If 'velocity_system' is not in the metadata of a spectrum.
    """

    new_spectrum_list = sp.SpectrumList()

    for spectrum in spectrum_list:
        if 'velocity_system' not in spectrum.meta:
            raise KeyError("Spectrum metadata missing 'velocity_system' key")
        if not isinstance(spectrum.meta['velocity_system'], astropy.nddata.NDDataArray):
            raise TypeError(
                "'velocity_system' is not an instance of astropy.nddata.NDDataArray")

        new_spectrum_list.append(
            sp.Spectrum1D(
                spectral_axis=spectrum.spectral_axis -
                (spectrum.meta['velocity_system'].data *
                 spectrum.meta['velocity_system'].unit).to(spectrum.spectral_axis.unit),
                flux=spectrum.flux,
                uncertainty=spectrum.uncertainty,
                meta=spectrum.meta,
            )
        )
    return sp.SpectrumList(new_spectrum_list)
# %% Residual CCFs


def calculate_residual_CCF(CCF_list: sp.SpectrumList, master_out_list: sp.SpectrumList) -> sp.SpectrumList:
    """
    Calculate the residual CCF list by subtracting the master out CCF from each CCF in the list.

    This function takes a list of normalized CCFs and a master out list, and calculates the residual CCFs by subtracting the corresponding master out CCF from each CCF in the list. The residual CCFs are then multiplied by -1 to invert them.

    Parameters
    ----------
    CCF_list : sp.SpectrumList
        List of normalized CCF's
    master_out_list : sp.SpectrumList
        Master out list of the CCF.

    Returns
    -------
    CCF_residual : sp.SpectrumList
        List of residual CCF's.
    """
    night_index_map = {
        str(test.meta['num_night']): idx for idx, test in enumerate(master_out_list)}
    return sp.SpectrumList(
        [
            CCF.subtract(
                master_out_list[night_index_map[str(CCF.meta['Night_num'])]],
                handle_meta='first_found'
            ).multiply(
                -1*u.dimensionless_unscaled,
                handle_meta='first_found'
            ) for CCF in CCF_list
        ]
    )
# %% Intrinsic CCFs


def calculate_intrinsic_CCF(CCF_list: sp.SpectrumList, master_out_list: sp.SpectrumList) -> sp.SpectrumList:
    """
    Calculate the intrinsic CCF list.

    Parameters
    ----------
    CCF_list : sp.SpectrumList
        List of normalized CCF's
    master_out_list : sp.SpectrumList
        Master out list of the CCF.

    Returns
    -------
    CCF_intrinsic : sp.SpectrumList
        List of intrinsic CCF's.
    """
    CCF_intrinsic = sp.SpectrumList()
    for CCF in CCF_list:
        if CCF.meta['delta'].value != 0:
            CCF_intrinsic.append(
                CCF.subtract(
                    master_out_list[int(np.argwhere(np.asarray(
                        [test.meta['num_night'] for test in master_out_list]) == str(CCF.meta['Night_num'])).flatten())],
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
                    master_out_list[int(np.argwhere(np.asarray(
                        [test.meta['num_night'] for test in master_out_list]) == str(CCF.meta['Night_num'])).flatten())],
                    handle_meta='first_found'
                ).multiply(
                    -1*u.dimensionless_unscaled,
                    handle_meta='first_found'
                ).add(
                    1*u.dimensionless_unscaled,
                    handle_meta='first_found'
                )
            )
    return CCF_intrinsic

# %% Subtraction of master out


def subtract_master_out(CCF_list: sp.SpectrumList,
                        master_out_list: sp.SpectrumList) -> tuple[sp.SpectrumList, sp.SpectrumList]:
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
    CCF_residual = calculate_residual_CCF(CCF_list, master_out_list)
    CCF_intrinsic = calculate_intrinsic_CCF(CCF_list, master_out_list)
    return CCF_residual, CCF_intrinsic


# %% Model local intrinsic CCF's


def _MCMC_model_local_CCF(CCF_local: sp.Spectrum1D, draws: int = 15000, tune: int = 15000, chains: int = 15, target_accept: float = 0.99, **kwargs_pymc: dict) -> arviz.InferenceData:
    """
    Model local intrinsic CCFs using MCMC routine.

    This function fits a Gaussian profile to the provided CCF using a Markov Chain Monte Carlo (MCMC) routine. 
    The function expects the CCF to be in transit and have valid metadata.

    Parameters
    ----------
    CCF_local : sp.Spectrum1D
        The local CCF to model. Must have 'Transit_partial' and 'Spec_num' in its metadata.
    draws : int, optional
        The number of samples to draw from the posterior distribution, by default 15000.
    tune : int, optional
        The number of tuning (burn-in) steps to take before sampling, by default 15000.
    chains : int, optional
        The number of chains to run in parallel, by default 15.
    target_accept : float, optional
        The target acceptance rate for the sampler, by default 0.99.
    **kwargs_pymc : dict, optional
        Additional keyword arguments to pass to the PyMC sampling function.

    Returns
    -------
    idata : arviz.InferenceData
        The inference data object containing the posterior samples and metadata.

    Raises
    ------
    ValueError
        If the provided CCF does not have metadata.
    AssertionError
        If the provided CCF is not in transit.
    """
    if CCF_local.meta is None:
        raise ValueError('Spectrum has no meta data')

    assert CCF_local.meta['Transit_partial'], 'Spectrum is not transiting'

    ind = np.logical_and(np.isfinite(CCF_local.flux),
                         np.isfinite(CCF_local.uncertainty.array))

    lower_bound = CCF_local.spectral_axis[0].value
    upper_bound = CCF_local.spectral_axis[-1].value
    spread_CCF = (upper_bound - lower_bound)

    local_CCF_model = pm.Model()
    with local_CCF_model:
        contrast = pm.Uniform('contrast',
                              lower=-2,
                              upper=2
                              )

        fwhm = pm.Uniform('fwhm',
                          lower=0,
                          upper=spread_CCF
                          )

        rvcenter = pm.Uniform('rvcenter',
                              lower=lower_bound,
                              upper=upper_bound
                              )

        # Expecting a Gaussian profile
        # fwhm to sigma is approx fwhm = 2.35482*sigma
        expected = (-contrast *
                    pm.math.exp(-((CCF_local.spectral_axis[ind].value-rvcenter)**2)/(2*((fwhm/2.35482)**2))) + 1)  # type: ignore

        # Observed, expecting Gaussian, sigma = uncertainty, observed = flux
        observed = pm.Normal("observed",
                             mu=expected,
                             sigma=CCF_local.uncertainty[ind].array,
                             observed=CCF_local.flux.value[ind]
                             )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            **kwargs_pymc
        )

        idata.attrs['Spec_num'] = CCF_local.meta['Spec_num']
    return idata
# %% Model CCF's distribution in list


@save_and_load
def MCMC_model_local_CCF_list(CCF_list: sp.SpectrumList, draws: int = 15000, tune: int = 15000, chains: int = 15, force_load: bool = True, force_skip: bool = False, pkl_name: str = 'distribution_local_CCF.pkl', **kwargs_pymc: dict) -> list:
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
    draws : int, optional
        The number of samples to draw from the posterior distribution, by default 15000.
    tune : int, optional
        The number of tuning (burn-in) steps to take before sampling, by default 15000.
    chains : int, optional
        The number of chains to run in parallel, by default 15.
    pkl_name : str, optional
        Location of the saved output, by default 'distribution_local_CCF.pkl'. It will be saved in the './saved_data/ directory
        **kwargs_pymc : dict, optional
        Additional keyword arguments to pass to the PyMC sampling function.

    Returns
    -------
    data_chain : list
        List of idata as calculated by the PyMC. For more info about idata structure check PyMC and arviz python documentation.
    """

    data_chain = []
    for CCF_local in CCF_list:
        if not (CCF_local.meta['Transit_partial']):
            continue
        data_chain.append(_MCMC_model_local_CCF(
            CCF_local,
            draws=draws,
            tune=tune,
            chains=chains,
            **kwargs_pymc,
        )
        )

    return data_chain
# %% Extract spectrum values


def _extract_values_in_transit(CCF_list: sp.SpectrumList) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts wavelength, flux, uncertainty, and phase values for in-transit CCFs.

    This function filters the provided list of CCFs to include only those that are marked as 'Transit_partial' in their metadata. 
    It then extracts the spectral axis values (wavelength), flux values, uncertainty values, and phase values for these in-transit CCFs.

    Parameters
    ----------
    CCF_list : sp.SpectrumList
        List of CCFs from which to extract values. Each CCF is expected to have 'Transit_partial' and 'Phase' keys in its metadata.

    Returns
    -------
    wavelength : np.ndarray
        Array of wavelength values for in-transit CCFs.
    flux : np.ndarray
        Array of flux values for in-transit CCFs.
    uncertainty : np.ndarray
        Array of uncertainty values for in-transit CCFs.
    phase : np.ndarray
        Array of phase values for in-transit CCFs.
    """

    wavelength = np.asarray(
        [CCF.spectral_axis.value for CCF in CCF_list if CCF.meta['Transit_partial']])
    flux = np.asarray(
        [CCF.flux.value for CCF in CCF_list if CCF.meta['Transit_partial']])
    uncertainty = np.asarray(
        [CCF.uncertainty.array for CCF in CCF_list if CCF.meta['Transit_partial']])
    phase = np.asarray([[CCF.meta['Phase'].data]*len(CCF.spectral_axis)
                       for CCF in CCF_list if CCF.meta['Transit_partial']])

    return wavelength, flux, uncertainty, phase
# %% Setup contrast model


def _add_contrast_model(model: pm.Model, CCF_intrinsic_list: sp.SpectrumList, order: int | None) -> dict | None:
    """
    Adds a contrast model to the given PyMC model.

    This function creates a uniform prior distribution for the contrast parameter and adds it to the provided PyMC model. 
    If an order is specified, it generates polynomial coefficients for each night and adds them to the model.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to which the contrast parameter will be added.
    CCF_intrinsic_list : sp.SpectrumList
        List of intrinsic CCF spectra.
    order : int | None
        The order of the polynomial to fit the contrast parameter. If None, a single uniform prior is added.

    Returns
    -------
    contrast_coefficients : dict | None
        A dictionary of contrast polynomial coefficients if an order is specified, otherwise None.
    """

    if order is None:
        with model:
            pm.Uniform(f'contrast',
                       lower=0,
                       upper=1)
            return None
    else:
        contrast_coefficients = {}
        for night in np.unique([CCF.meta['Night'] for CCF in CCF_intrinsic_list]):
            with model:
                for i in range(order + 1):
                    bound = 1 / np.math.factorial(i)
                    if i == 0:
                        coef = pm.Uniform(f'contrast_night{night}_{i}',
                                          lower=0,
                                          upper=bound)
                    else:
                        coef = pm.Uniform(f'contrast_night{night}_{i}',
                                          lower=-bound,
                                          upper=bound)
                    contrast_coefficients[f'contrast_night{night}_{i}'] = coef
        return contrast_coefficients
# %% Add FWHM model


def _add_fwhm_model(model: pm.Model, CCF_intrinsic_list: sp.SpectrumList, order: int | None) -> dict | None:
    """
    Adds a Full Width at Half Maximum (FWHM) model to the given PyMC model.

    This function creates a uniform prior distribution for the FWHM parameter and adds it to the provided PyMC model. 
    If an order is specified, it generates polynomial coefficients for each night and adds them to the model.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to which the FWHM parameter will be added.
    CCF_intrinsic_list : sp.SpectrumList
        List of intrinsic CCF spectra.
    order : int | None
        The order of the polynomial to fit the FWHM parameter. If None, a single uniform prior is added.

    Returns
    -------
    fwhm_coefficients : dict | None
        A dictionary of FWHM polynomial coefficients if an order is specified, otherwise None.
    """

    lower_bound = CCF_intrinsic_list[0].spectral_axis[0].value
    upper_bound = CCF_intrinsic_list[0].spectral_axis[-1].value
    spread_CCF = (upper_bound - lower_bound)

    if order is None:
        with model:
            pm.Uniform(f'fwhm',
                       lower=0,
                       upper=spread_CCF)
            return None
    else:
        fwhm_coefficients = {}
        for night in np.unique([CCF.meta['Night'] for CCF in CCF_intrinsic_list]):
            with model:
                for i in range(order + 1):
                    bound = spread_CCF / np.math.factorial(i)
                    if i == 0:
                        coef = pm.Uniform(f'fwhm_night{night}_{i}',
                                          lower=0,
                                          upper=bound)
                    else:
                        coef = pm.Uniform(f'fwhm_night{night}_{i}',
                                          lower=-bound,
                                          upper=bound)
                    fwhm_coefficients[f'fwhm_night{night}_{i}'] = coef
        return fwhm_coefficients


def generate_polynomials_per_night(model: pm.Model,
                                   coefficients: dict,
                                   polynomial: dict,
                                   key: str,
                                   order: int | None,
                                   night: int,
                                   phase: np.ndarray) -> dict:
    """
    Generates polynomials for each night based on the given coefficients and order.

    This function updates the `polynomial` dictionary with polynomial expressions for a given night, 
    using the provided coefficients and order. If the order is None, it assigns the model's key directly 
    to the polynomial dictionary. Otherwise, it constructs the polynomial expression by summing the 
    coefficients multiplied by the phase raised to the power of the current order.

    Parameters
    ----------
    model : pm.Model
        The PyMC model containing the parameters.
    coefficients : dict
        A dictionary of coefficients for the polynomial terms.
    polynomial : dict
        A dictionary to store the generated polynomial expressions.
    key : str
        The key to identify the type of polynomial (e.g., 'contrast', 'fwhm').
    order : int | None
        The order of the polynomial. If None, the model's key is used directly.
    night : int
        The night identifier for which the polynomial is generated.
    phase : np.ndarray
        The phase values used to generate the polynomial terms.

    Returns
    -------
    dict
        The updated polynomial dictionary with the generated polynomial expressions for the given night.
    """
    if order is None:
        polynomial[f'{key}_night{night}_polynomial'] = model[f'{key}']
    else:
        polynomial[f'{key}_night{night}_polynomial'] = 0
        for c_order in range(order+1):
            polynomial[f'{key}_night{night}_polynomial'] += coefficients[f'{key}_night{night}_{c_order}'] * \
                phase**(c_order)

    return polynomial


# %% Revolutions MCMC
@save_and_load
def MCMC_model_Revolutions(system_parameters, CCF_intrinsic_list: sp.SpectrumList, contrast_order: int | None = 0, fwhm_order: int | None = 0, contrast_phase: str = 'Phase', fwhm_phase: str = 'Phase', draws: int = 15000, tune: int = 15000, force_skip: bool = False, force_load: bool = True, pkl_name: str = ''):
    """
    Models the Revolutions method of the Rossiter-McLaughlin effect using MCMC.

    This function constructs a PyMC model to fit the intrinsic CCFs using a Gaussian profile. It accounts for the contrast and FWHM variations over different nights and phases. The model parameters include the obliquity and the projected rotational velocity (veqsini) of the star.

    Parameters
    ----------
    system_parameters : object
        An object containing the system parameters, including the semi-major axis and inclination of the planet, and the radius of the star.
    CCF_intrinsic_list : sp.SpectrumList
        List of intrinsic CCF spectra.
    contrast_order : int | None, optional
        The order of the polynomial to fit the contrast parameter, by default 0.
    fwhm_order : int | None, optional
        The order of the polynomial to fit the FWHM parameter, by default 0.
    contrast_phase : str, optional
        The phase key for contrast, by default 'Phase'.
    fwhm_phase : str, optional
        The phase key for FWHM, by default 'Phase'.
    draws : int, optional
        The number of samples to draw from the posterior distribution, by default 15000.
    tune : int, optional
        The number of tuning (burn-in) steps to take before sampling, by default 15000.
    force_skip : bool, optional
        Whether to force skipping the function, by default False.
    force_load : bool, optional
        Whether to force loading the result of the function, by default True.
    pkl_name : str, optional
        Location of the saved output, by default ''.

    Returns
    -------
    idata : arviz.InferenceData
        The inference data object containing the posterior samples and metadata.
    basic_model : pm.Model
        The constructed PyMC model.

    Raises
    ------
    NotImplementedError
        If custom phase handling for contrast and FWHM is not implemented.
    """
    if contrast_phase != 'Phase' or fwhm_phase != 'Phase':
        raise NotImplementedError(
            'Custom phase handling for contrast and FWHM is not implemented yet. Please ensure contrast_phase and fwhm_phase are set to "Phase".')

    basic_model = pm.Model()

    aRs = (system_parameters.Planet.semimajor_axis.divide(
        system_parameters.Star.radius)).convert_unit_to(u.dimensionless_unscaled).data

    lower_bound = CCF_intrinsic_list[0].spectral_axis[0].value
    upper_bound = CCF_intrinsic_list[0].spectral_axis[-1].value
    spread_CCF = (upper_bound - lower_bound)

    night_polynomial = {}

    with basic_model:
        obliquity = pm.Uniform('obliquity',
                               lower=-180,
                               upper=180)

        veqsini = pm.Uniform('veqsini',
                             lower=0,
                             upper=spread_CCF)
    contrast_coefficients = _add_contrast_model(
        basic_model, CCF_intrinsic_list, contrast_order)
    fwhm_coefficients = _add_fwhm_model(
        basic_model, CCF_intrinsic_list, fwhm_order)

    for night in np.unique([item.meta['Night'] for item in CCF_intrinsic_list]):
        sublist = sm.get_sublist(CCF_intrinsic_list, 'Night', night)
        sublist = sm.get_sublist(sublist, 'Revolutions', True)
        if len(sublist) == 0:
            continue

        wavelength, flux, uncertainty, phase = _extract_values_in_transit(
            sublist)

        indices = np.logical_and(
            np.isfinite(flux),
            np.isfinite(uncertainty)
        )

        wavelength = wavelength[indices]
        flux = flux[indices]
        uncertainty = uncertainty[indices]
        phase = phase[indices]

        lower_bound = sublist[0].spectral_axis[0].value
        upper_bound = sublist[0].spectral_axis[-1].value
        spread_CCF = (upper_bound - lower_bound)

        with basic_model:

            x_p = aRs * pm.math.sin(2*np.pi * phase)  # type: ignore
            y_p = aRs * pm.math.cos(2*np.pi * phase) * pm.math.cos(
                system_parameters.Planet.inclination.data / 180*np.pi)  # type: ignore
            x_perpendicular = x_p * \
                pm.math.cos(obliquity/180*np.pi) - y_p * \
                pm.math.sin(obliquity/180*np.pi)  # type: ignore
            y_perpendicular = x_p * \
                pm.math.sin(obliquity/180*np.pi) - y_p * \
                pm.math.cos(obliquity/180*np.pi)  # type: ignore

            local_stellar_velocity = x_perpendicular * veqsini

            night_polynomial = generate_polynomials_per_night(
                basic_model, contrast_coefficients, night_polynomial, 'contrast', contrast_order, night, phase)
            night_polynomial = generate_polynomials_per_night(
                basic_model, fwhm_coefficients, night_polynomial, 'fwhm', fwhm_order, night, phase)

            # Expecting a Gaussian profile
            expected = (
                -night_polynomial[f'contrast_night{night}_polynomial'] * pm.math.exp(-(  # type: ignore
                    (wavelength - local_stellar_velocity)**2
                ) /
                    (2 *
                     ((night_polynomial[f'fwhm_night{night}_polynomial']/2.35482)**2))
                ) + 1
            )  # fwhm to sigma is approx fwhm = 2.35482*sigma

            # Observed, expecting Gaussian, sigma = uncertainty, observed = flux
            observed = pm.Normal(f"observed_night{night}",
                                 mu=expected,
                                 sigma=uncertainty,
                                 observed=flux
                                 )
    with basic_model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=30,
            target_accept=0.9,
            idata_kwargs={
                'log_likelihood': True
            }
        )

    return idata, basic_model


def set_valid_and_invalid_indices(CCF_intrinsic_filtered: sp.SpectrumList, indices: list[int]) -> sp.SpectrumList:
    """
    Set valid and invalid indices for CCF intrinsic filtered data.

    This function iterates through the provided CCF intrinsic filtered data and sets the 'Revolutions' meta key to True if the 'Transit_partial' meta key is True. It then sets the 'Revolutions' meta key to False for the CCFs whose 'Spec_num' is in the provided indices list.

    Parameters
    ----------
    CCF_intrinsic_filtered : sp.SpectrumList
        A list of CCF intrinsic filtered data, where each element is expected to have a 'meta' dictionary with 'Transit_partial' and 'Spec_num' keys.
    indices : list[int]
        A list of indices specifying which CCFs should have their 'Revolutions' meta key set to False.
    Returns
    -------
    sp.SpectrumList
        The modified list of CCF intrinsic filtered data with updated 'Revolutions' meta keys.
    """

    for CCF in CCF_intrinsic_filtered:
        if 'Transit_partial' not in CCF.meta:
            raise KeyError("Spectrum metadata missing 'Transit_partial' key")
        if CCF.meta['Transit_partial']:
            CCF.meta['Revolutions'] = True
        else:
            CCF.meta['Revolutions'] = False

    array_indices = [i for i, ccf in enumerate(
        CCF_intrinsic_filtered) if ccf.meta['Spec_num'] in indices]
    for index in array_indices:
        CCF_intrinsic_filtered[index].meta['Revolutions'] = False

    return CCF_intrinsic_filtered
