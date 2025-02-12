"""
This module provides functions for modeling planetary atmospheres using the petitRADTRANS package.

Functions:
- _airtovac(wlnm):
    Converts air wavelengths to vacuum wavelengths.

- _vactoair(wlnm):
    Converts vacuum wavelengths to air wavelengths.

- _get_planet(planet_name: str) -> planet.Planet:
    Gets planet parameters from the planet name.

- _get_edges_wavelength(spectral_axis: sp.SpectralAxis) -> list:
    Gets edges of the wavelengths to use for the model from the spectral axis.

- _define_atmosphere_model(pressures: np.ndarray, line_species: list, wavelength_boundaries: list | sp.SpectralAxis = [0.3, 0.8], rayleigh_species: list | None = None, gas_continuum_contributors: list | None = None, **kwargs) -> prt.Radtrans:
    Defines an atmosphere Radtrans model.

- _T_P_profile_guillot(SystemParameters: rats.parameters.SystemParametersComposite, atmosphere: prt.Radtrans) -> np.ndarray:
    Creates a T-P profile using the Guillot approximation.

- _get_mass_fraction(line_species: list, temperatures: np.ndarray, value: float = 1E-7) -> dict:
    Gets a mass fraction for related species.

- _calculate_transmission_model(atmosphere: prt.Radtrans, spectral_axis: sp.SpectralAxis, temperatures: np.ndarray, mass_fractions: dict, mean_molar_masses: np.ndarray, SystemParameters: rats.parameters.SystemParametersComposite, reference_pressure: float = 10E-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Calculates the transmission model based on the input parameters.

- _remove_continuum(spectral_axis: sp.SpectralAxis, flux: u.Quantity) -> u.Quantity:
    Removes the continuum from the template.

- create_template(spectral_axis: sp.SpectralAxis, SystemParameters: rats.parameters.SystemParametersComposite, pressures: np.ndarray = np.logspace(-10,2,130), line_species: list = [], reference_pressure: float = 10E-2, abundance: float = 1E-7) -> sp.Spectrum1D:
    Generates a model template for given species using the Guillot TP profile.

- create_templates(spectral_axis: sp.SpectralAxis, SystemParameters: rats.parameters.SystemParametersComposite, pressures: np.ndarray = np.logspace(-10,2,130), line_species: list = [], reference_pressure: float = 10E-2, abundance: float = 1E-7, force_recalculate: bool = False) -> sp.SpectrumList:
    Generates a list of templates for given species list.

- create_all_templates(spectral_axis: sp.SpectralAxis, SystemParameters: rats.parameters.SystemParametersComposite, pressures: np.ndarray = np.logspace(-10,2,130), reference_pressure: float = 10E-2, abundance: float = 1E-7, force_recalculate: bool = False) -> sp.SpectrumList:
    Generates a list of templates for all petitRADTRANS species individually.
"""

import petitRADTRANS.radtrans as prt
import petitRADTRANS.planet as planet
from petitRADTRANS.physics import temperature_profile_function_guillot_global
from rats.lists.sources import LINELIST_pRT
import numpy as np
import specutils as sp
import rats
import scipy
import dill as pickle
import os
import rats.spectra_manipulation as sm
import rats.parameters
import astropy.units as u
from astropy.nddata import StdDevUncertainty
from rats.utilities import time_function, save_and_load, default_logger_format, progress_tracker
import logging

logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

'''Shifting air to vac and vac to air wavelength'''
def _airtovac(wlnm):
    wlA=wlnm*10.0
    s = 1e4 / wlA
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return(wlA*n/10.0)

def _vactoair(wlnm):
    wlA = wlnm*10.0
    s = 1e4/wlA
    f = 1.0 + 5.792105e-2/(238.0185e0 - s**2) + 1.67917e-3/( 57.362e0 - s**2)
    return(wlA/f/10.0)


def _get_planet(planet_name: str) -> planet.Planet:
    """
    Get planet parameters from planet name.

    Parameters
    ----------
    planet_name : str
        Planet name as defined by NASA exoplanet archive.
        
    Returns
    -------
    prt.planet.Planet
        Planet parameters as extracted by pRT.
    """
    planet_obj = planet.Planet.get(
        planet_name
        )
    return planet_obj

def _get_edges_wavelength(spectral_axis: sp.SpectralAxis) -> list:
    """
    Get edges of the wavelengths to use for the model from spectral axis.

    Parameters
    ----------
    spectral_axis : sp.SpectralAxis
        Spectral axis for which to create the template.

    Returns
    -------
    list
        Edges of the spectral axis as can be added to the Radtrans object. 
    """
    return [spectral_axis[0].to(u.um).value, spectral_axis[-1].to(u.um).value] #type: ignore

@time_function
def _define_atmosphere_model(pressures: np.ndarray,
                             line_species: list,
                             wavelength_boundaries: list | sp.SpectralAxis = [0.3, 0.8],
                             rayleigh_species: list | None = None,
                             gas_continuum_contributors: list | None = None,
                             **kwargs,
                             ) -> prt.Radtrans:
    """
    Define an atmosphere Radtrans model

    Parameters
    ----------
    pressures : np.ndarray
        Pressures for which to create the model.
    line_species : list
        Line species for which to calculate the model.
    wavelength_boundaries : list | sp.SpectralAxis, optional
        Wavelengths boundaries, by default [0.3, 0.8]. If spectral axis is passed instead, it will be reformated for Radtrans input.
    rayleigh_species : list | None, optional
        Which species to consider for rayleigh species, by default None
    gas_continuum_contributors : list | None, optional
        Gas continuum contributors, by default None.

    Returns
    -------
    prt.Radtrans
        Radtrans model for the atmosphere
    """
    
    if isinstance(wavelength_boundaries, sp.SpectralAxis):
        wavelength_boundaries = _get_edges_wavelength(wavelength_boundaries)
    
    radtrans_params = {
        'pressures': pressures,
        'line_species': line_species,
        'wavelength_boundaries': wavelength_boundaries,
        'line_opacity_mode': 'lbl'
    }
    
    if rayleigh_species is not None:
        radtrans_params['rayleigh_species'] = rayleigh_species
    
    if gas_continuum_contributors is not None:
        radtrans_params['gas_continuum_contributors'] = gas_continuum_contributors
    
    radtrans_params.update(kwargs)
    
    atmosphere = prt.Radtrans(**radtrans_params)
    
    return atmosphere

@time_function
def _T_P_profile_guillot(SystemParameters: rats.parameters.SystemParametersComposite,
                         atmosphere: prt.Radtrans
                         ) -> np.ndarray:
    """
    Create a T-P profile using guillot approximation.

    Parameters
    ----------
    SystemParameters : para.SystemParametersComposite
        System parameters of the explored planet.
    atmosphere : Radtrans
        Atmosphere model.

    Returns
    -------
    temperature : np.ndarray
        Temperature array for given pressure array.
    """
    gravity = SystemParameters.Planet.gravity_acceleration.convert_unit_to(u.cm/u.s/u.s).data
    pressures_bar = atmosphere.pressures *1e-6 # cgs to bar
    # FIXME Check what are these values and see if they should be played with instead of being hardcoded
    kappa_IR = 0.01
    gamma = 0.4
    T_int = 300.
    
    if SystemParameters.Planet.equilibrium_temperature is not None:
        T_equ = SystemParameters.Planet.equilibrium_temperature.data
    else:
        raise ValueError("SystemParameters.Planet.equilibrium_temperature is None")

    temperatures = temperature_profile_function_guillot_global(
        pressures= pressures_bar,
        infrared_mean_opacity= kappa_IR,
        gamma= gamma,
        gravities= gravity,
        intrinsic_temperature= T_int,
        equilibrium_temperature= T_equ
        )
    return temperatures

def _get_mass_fraction(line_species: list,
                       temperatures: np.ndarray,
                       value: float = 1E-7) -> dict:
    """
    Get a mass fraction for related species.

    Parameters
    ----------
    line_species : list
        Species for which to add the mass fraction
    temperatures : np.ndarray
        Temperature array which is going to be used as shape for the mass fractions.
    value : float, optional
        Value to which to input for the lines, by default 1E-7. This is first guess, more robust method will be added.

    Returns
    -------
    dict
        Mass fractions for main (hydrogen and helium) and requested line species.
    """
    mass_fractions = {
        'H2': 0.74 * np.ones_like(temperatures),
        'He': 0.24 * np.ones_like(temperatures),
    }
    
    for line in line_species:
        mass_fractions[line] = value  * np.ones_like(temperatures)
    
    return mass_fractions

@time_function
def _calculate_transmission_model(atmosphere: prt.Radtrans,
                                  spectral_axis: sp.SpectralAxis,
                                  temperatures: np.ndarray,
                                  mass_fractions: dict,
                                  mean_molar_masses: np.ndarray,
                                  SystemParameters: rats.parameters.SystemParametersComposite,
                                  reference_pressure: float = 10E-2,
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates transmission model based on the input parameters.

    Parameters
    ----------
    spectral_axis : sp.SpectralAxis
        Spectral axis for which to calculate and interpolate to the template.
    temperatures : np.ndarray
        Temperatures array to provide T-P profile
    mass_fractions : dict
        Mass fraction dictionary containing all the requested elements
    mean_molar_masses : np.ndarray
        Mean molecular masses
    SystemParameters : rats.parameters.SystemParametersComposite
        System parameters for given system as loaded by rats.parameters package
    reference_pressure : float, optional
        Reference pressure to use, by default 10E-2

    Returns
    -------
    wavelengths, transit_radii, interpolated_transit_radii [np.ndarray, np.ndarray, np.ndarray]
        Wavelengths and transit radii obtained by the model, and interpolated transit radii on the requested spectral axis
    """
    reference_gravity = SystemParameters.Planet.gravity_acceleration.convert_unit_to(u.cm/u.s/u.s).data
    planet_radius = SystemParameters.Planet.radius.convert_unit_to(u.cm).data #type: ignore
    
    wavelengths, transit_radii, _ = atmosphere.calculate_transit_radii(temperatures=temperatures,
                                                                   mass_fractions=mass_fractions,
                                                                   mean_molar_masses=mean_molar_masses,
                                                                   reference_gravity=reference_gravity, #type: ignore
                                                                   planet_radius=planet_radius, #type: ignore
                                                                   reference_pressure=reference_pressure)
    wavelengths_air = ((_vactoair((wavelengths*u.cm).to(u.nm).value))*u.nm).to(u.cm).value #type: ignore
    flux_interpolation = scipy.interpolate.CubicSpline(wavelengths_air, transit_radii, extrapolate=False)
    interpolated_transit_depth = flux_interpolation(spectral_axis.to(u.cm).value)
    
    return wavelengths_air, transit_radii, interpolated_transit_depth



def _remove_continuum(spectral_axis: sp.SpectralAxis,
                      flux: u.Quantity
                      ) -> u.Quantity:
    """
    Removes continuum from the template. Necessary for high-resolution as we lose information about continuum through normalization.

    Parameters
    ----------
    spectral_axis : sp.SpectralAxis
        Spectral axis of the template
    flux : np.ndarray
        Flux of the template

    Returns
    -------
    flux : np.ndarray
        Template flux without the continuum.
    """
    
    ind = np.isfinite(flux)
    flux_fitting = flux[ind]
    spectral_axis_fitting = spectral_axis[ind]
    
    from astropy.modeling import models, fitting
    fit = fitting.LinearLSQFitter()
    line_init = models.Polynomial1D(6)
    
    fitted_line = fit(line_init, spectral_axis_fitting, flux_fitting)
    
    # Get a mask where the line and fitted model align within 1 %
    mask = np.where(abs((flux- fitted_line(spectral_axis))/ flux) < 0.05)
    fitted_line = fit(line_init, spectral_axis[mask], flux[mask])
    
    flux = flux - fitted_line(spectral_axis) + 1 *flux.unit #type: ignore
    mask = np.where(flux < 1.05*flux.unit) #type: ignore
    flux[mask] = 1 *flux.unit #type: ignore
    
    return flux

def create_template(spectral_axis: sp.SpectralAxis,
                    SystemParameters: rats.parameters.SystemParametersComposite,
                    pressures: np.ndarray = np.logspace(-10,2,130),
                    line_species: list = [],
                    reference_pressure: float = 10E-2,
                    abundance: float = 1E-7,
                    ) -> sp.Spectrum1D:
    """
    Generates a model template for given species, using Guillot TP profile.

    Parameters
    ----------
    spectral_axis : sp.SpectralAxis
        Spectral axis of a spectrum.
    SystemParameters : rats.parameters.SystemParametersComposite
        System parameters used to scale the template to planet radii
    pressures : np.ndarray, optional
        Pressures array to assume for TP profile, by default np.logspace(-10,2,130)
    line_species : list, optional
        Line species list to use, by default []. Can be any number of species. If empty, only rayleigh scattering and gas continuum contribution is provided.
    reference_pressure : float, optional
        Reference pressure for planet radii, by default 10E-2. This strengthen/weaken lines, as we probe deeper/shallower layers of the atmosphere.

    Returns
    -------
    sp.Spectrum1D
        Template spectrum of given species.
    """
    
    atmosphere = _define_atmosphere_model(
        pressures=pressures,
        line_species= line_species,
        wavelength_boundaries= spectral_axis,
        rayleigh_species=['H2', 'He'],
        gas_continuum_contributors = ['H2-H2', 'H2-He']
    )
    
    
    if type(spectral_axis) == sp.SpectralAxis:
        wavelength_boundaries = _get_edges_wavelength(spectral_axis)
    
    # atmosphere = prt.Radtrans(
    #     pressures=pressures,
    #     line_species=line_species,
    #     rayleigh_species= ['H2', 'He'],
    #     gas_continuum_contributors= ['H2-H2', 'H2-He', 'H-'],
    #     line_opacity_mode = 'lbl',
    #     wavelength_boundaries= wavelength_boundaries,
    #     cloud_species = ['H2O-NatAbund(s)_crystalline_194__DHS.R39_0.1-250mu'],
    # )
    
    
    temperatures = _T_P_profile_guillot(
        SystemParameters,
        atmosphere= atmosphere
    )
    
    mass_fractions = _get_mass_fraction(
        line_species= line_species,
        temperatures= temperatures,
        value = abundance
    )

    mean_molar_masses = 2.33 * np.ones_like(temperatures)

    wavelengths, transit_radii, interpolated_transit_depth = _calculate_transmission_model(
        atmosphere= atmosphere,
        spectral_axis= spectral_axis,
        temperatures= temperatures,
        mass_fractions= mass_fractions,
        mean_molar_masses= mean_molar_masses,
        SystemParameters= SystemParameters,
        reference_pressure= reference_pressure
    )
    
    transit_radii = (transit_radii*u.cm).to(u.R_jup) #type: ignore
    interpolated_transit_depth = (interpolated_transit_depth*u.cm).to(u.R_jup) #type: ignore
    
    interpolated_transit_depth = _remove_continuum(spectral_axis= spectral_axis,
                                                   flux= interpolated_transit_depth)
    
    logger.print(f'Generated template with petitRADtrans for line list {line_species}') #type: ignore
    
    equivalency_transmission, F_lam, R, R_plam, delta_lam, H_num = sm.custom_transmission_units(SystemParameters)
    
    template = sp.Spectrum1D(
        spectral_axis= spectral_axis,
        flux= interpolated_transit_depth.value * R_plam, 
        uncertainty= StdDevUncertainty(np.zeros_like(interpolated_transit_depth.value)),
        mask = np.isnan(interpolated_transit_depth.value),
        meta= {
            'species': line_species,
            'P0': reference_pressure,
            'T-P profile': 'Guillot TP profile'
        }
    )
    
    return template
    

def create_templates(spectral_axis: sp.SpectralAxis,
                    SystemParameters: rats.parameters.SystemParametersComposite,
                    pressures: np.ndarray = np.logspace(-10,2,130),
                    line_species: list = [],
                    reference_pressure: float = 10E-2,
                    abundance: float = 1E-7,
                    force_recalculate: bool = False) -> sp.SpectrumList:
    """
    Generates a list of templates for given species list.

    Parameters
    ----------
    spectral_axis : sp.SpectralAxis
        Spectral axis of a spectrum.
    SystemParameters : rats.parameters.SystemParametersComposite
        System parameters used to scale the template to planet radii
    pressures : np.ndarray, optional
        Pressures array to assume for TP profile, by default np.logspace(-10,2,130)
    line_species : list, optional
        Line species list to use, by default []. Can be any number of species. If empty, only rayleigh scattering and gas continuum contribution is provided.
    reference_pressure : float, optional
        Reference pressure for planet radii, by default 10E-2. This strengthen/weaken lines, as we probe deeper/shallower layers of the atmosphere.
    force_recalculate : bool, optional
        Whether to force recalculation of templates, by default False. If true, templates are recalculated even if they were previously calculated.

    Returns
    -------
    sp.SpectrumList
        List of templates as provided by create_template function.
    """
    templates = sp.SpectrumList()
    for line in line_species:
        save_directory = f'./saved_data/templates/petitradtrans/{line}/'
        os.makedirs(save_directory,
                    mode = 0o777,
                    exist_ok = True
                    )
        # Try load the template
        filename = save_directory+f'pressure_{str(reference_pressure)}_abund_{str(abundance)}.pkl'
        if not(force_recalculate):
            try:
                with open(filename, 'rb') as input_file:
                    template = pickle.load(input_file)
                    logger.info('Opened template in %s'%save_directory+f'{str(reference_pressure)}')
                    logger.warning('No checks on the validity of the file done, please make sure correct file was loaded')
                    templates.append(template)
                    continue
            except:
                pass
        
        template = create_template(spectral_axis= spectral_axis,
                                   SystemParameters= SystemParameters,
                                   pressures= pressures,
                                   line_species = [line],
                                   reference_pressure= reference_pressure,
                                   abundance= abundance
                                   )
        
        # Save the template for later use
        with open(filename, 'wb') as output_file:
            logger.info(f'Saved template in {filename}')
            pickle.dump(template, output_file)
        
        templates.append(template)
    return templates


def create_all_templates(spectral_axis: sp.SpectralAxis,
                    SystemParameters: rats.parameters.SystemParametersComposite,
                    pressures: np.ndarray = np.logspace(-10,2,130),
                    reference_pressure: float = 10E-2,
                    abundance: float = 1E-7,
                    force_recalculate: bool = False) -> sp.SpectrumList:
    """
    Generates a list of templates for all petitRADTRANS species individually.

    Parameters
    ----------
    spectral_axis : sp.SpectralAxis
        Spectral axis of a spectrum.
    SystemParameters : rats.parameters.SystemParametersComposite
        System parameters used to scale the template to planet radii
    pressures : np.ndarray, optional
        Pressures array to assume for TP profile, by default np.logspace(-10,2,130)
    reference_pressure : float, optional
        Reference pressure for planet radii, by default 10E-2. This strengthen/weaken lines, as we probe deeper/shallower layers of the atmosphere.
    force_recalculate : bool, optional
        Whether to force recalculation of templates, by default False. If true, templates are recalculated even if they were previously calculated.

    Returns
    -------
    sp.SpectrumList
        List of templates as provided by create_template function.
    """
    templates = create_templates(spectral_axis= spectral_axis,
                                   SystemParameters= SystemParameters,
                                   pressures= pressures,
                                   line_species = LINELIST_pRT,
                                   reference_pressure= reference_pressure,
                                   abundance= abundance,
                                   force_recalculate=force_recalculate)
    return templates

# List of undocumented functions for further review:
# - _airtovac
# - _vactoair
# - _get_planet
# - _get_edges_wavelength
# - _define_atmosphere_model
# - _T_P_profile_guillot
# - _get_mass_fraction
# - _calculate_transmission_model
# - _remove_continuum
# - create_template
# - create_templates
# - create_all_templates



