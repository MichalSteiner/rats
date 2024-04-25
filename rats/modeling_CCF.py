# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:26:53 2020

@author: Michal Steiner

Usage: 
    Please install petitradtrans based on the provided instructions.
    
"""
#%% Importing libraries
from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc
from rats.modeling_CCF_lists import *
import os
import specutils as sp
import astropy
import scipy as sci
import pandas as pd
import numpy as np
import astropy.units as u
from petitRADTRANS.physics import guillot_global
from rats.utilities import default_logger_format
import logging
import rats.parameters as para
import matplotlib.pyplot as plt
import matplotlib as mpl
from rats.utilities import time_function, save_and_load, progress_tracker, skip_function, disable_func, default_logger_format

#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

#%% Setup the opacity list location
OPACITY_LIST_LOCATION = '/media/chamaeleontis/Observatory_main/Code/petitradtrans/input_data_std/input_data/opacities/lines'
logger.warning('Current location of opacity tables ')
logger.warning(f'    {OPACITY_LIST_LOCATION}')

#%% Find available species
SPECIES_LIST = []
for item in os.listdir(OPACITY_LIST_LOCATION + '/line_by_line/'):
    if os.path.isdir(OPACITY_LIST_LOCATION + '/line_by_line/' + item):
        logger.debug('Added species in list:')
        SPECIES_LIST.append(item)
        logger.debug(f'    {item}')
# Remove invalid species from list
# TODO Check why they don't work.
SPECIES_LIST.remove('H2_main_iso')
# SPECIES_LIST.remove('Na_allard_new')
SPECIES_LIST.remove('VO_ExoMol_Specific_Transitions')
SPECIES_LIST.remove('VO_ExoMol_McKemmish')
SPECIES_LIST.remove('K_allard_cold')

# Different error then the rest
SPECIES_LIST.remove('Y')
#TODO Add the PH3 species.

#%% Solar metallicity dictionary
ABUNDANCE_SOLAR_METALLICITY = ABUNDANCE_SOLAR_METALLICITY_LODDERS2021
#%% Checks the abundance sum
def _check_abundance_sum():
    """
    Checks whether the solar abundance is close to 1. Used for testing.
    """
    sum_abundance = 0
    for key in ABUNDANCE_SOLAR_METALLICITY.keys():
        sum_abundance +=  ABUNDANCE_SOLAR_METALLICITY[key]
    logger.warning(f'The total fraction of solar abundance is {sum_abundance}')
    logger.warning('Please check it is close to 1')

#%% Modify solar abundance by metallicity
def _modify_abundance(metallicity: float) -> dict:
    """
    Modify mass fraction abundances given a metallicity.

    Parameters
    ----------
    metallicity : float
        Factor of how many times more the metals are present.

    Returns
    -------
    modified_mass_fraction : dict
        Modified mass fraction.
    """
    modified_mass_fraction = ABUNDANCE_SOLAR_METALLICITY.copy()
    
    original_metal_fraction = (1-modified_mass_fraction['H'] - modified_mass_fraction['He'])
    new_metal_fraction = original_metal_fraction * metallicity
    new_nonmetal_fraction = 1 - new_metal_fraction
    
    h_modify_fraction = modified_mass_fraction['H'] / (modified_mass_fraction['H'] + modified_mass_fraction['He'])
    he_modify_fraction = modified_mass_fraction['He'] / (modified_mass_fraction['H'] + modified_mass_fraction['He'])
    
    for key in modified_mass_fraction.keys():
        if key == 'H':
            modified_mass_fraction[key] = new_nonmetal_fraction * h_modify_fraction
        elif key == 'He':
            modified_mass_fraction[key] = new_nonmetal_fraction * he_modify_fraction
        else:
            modified_mass_fraction[key] = modified_mass_fraction[key] * metallicity
    
    return modified_mass_fraction

#%% Print scaled abundance (for testing)
def _print_scaled_abundance(metallicity: float):
    """
    Prints the content of the metallicity. Mostly useful for testing.

    Parameters
    ----------
    metallicity : float
        Metallicity by which to scale.
    """
    modified_mass_fraction = _modify_abundance(metallicity)
    sum_value = 0
    for key in modified_mass_fraction.keys():
        logger.print(f'Original abundance: {ABUNDANCE_SOLAR_METALLICITY[key]} | Modified abundance: {modified_mass_fraction[key]}')
        sum_value += modified_mass_fraction[key]
    logger.print('The sum of mass fractions is {sum_value}')
    
#%% Test species validity
def _test_species_validity(species: list):
    """
    Test the validity of species by running the Radtrans model.

    Parameters
    ----------
    species : list
        List of available species
    """
    
    logger.info('Test of species validity in petitRADTRANS')
    for spec in species:
        logger.info(f'    Testing species: {spec}')
        try:
            atmosphere = Radtrans(line_species = [
                spec
                ],
                wlen_bords_micron = [0.6, 0.62],
                mode = 'lbl'
                )
        except:
            location = OPACITY_LIST_LOCATION + '/line_by_line/' + spec
            logger.critical(f'    Loading of species {spec} located in:')
            logger.critical(f'        {location}')
            logger.critical('    failed')
    logger.info('Testing species validity finished.')
    return

#%% Prepare mass fraction for petitRADTrans use
def _prepare_mass_fraction_for_petitRADTrans(mass_fraction: dict,
                                             temperature: np.array,
                                             flag_normalization: bool = True) -> dict:
    """
    Prepare a mass fraction dictionary for petitRADTrans model.

    Parameters
    ----------
    mass_fraction : dict
        Mass fraction given a metallicity. 
    temperature : np.array
        Temperature array, used for giving the proper shape of arrays in the new_mass_fraction dictionary.
    flag_normalization : bool
        Whether to normalize the mass fraction to 1, by default True.

    Returns
    -------
    new_mass_fraction : dict
        Mass fraction dictionary prepared for petitRADTrans model.
    """
    # CLEANME 
    
    new_mass_fraction = {}
    sum_values = 0
    for key in SPECIES_LIST:
        value = _get_mass_fraction_value_for_key(key, mass_fraction)
        sum_values += value
        new_mass_fraction[key] = value * np.ones_like(temperature)
    
    new_mass_fraction['H2'] = mass_fraction['H']
    new_mass_fraction['He'] = mass_fraction['He']
    
    sum_values += mass_fraction['H'] + mass_fraction['He']
    
    # FIXME I think this should be moved outside the for-looping, if possible
    if flag_normalization:
        # FIXME Check if this step makes sense
        logger.debug(f'The sum of mass_fraction is: {sum_values}')
        logger.debug(f'Normalizing to 1.')
        logger.debug(f'    To disable this step, use flag_normalization = False')
    
    for key in SPECIES_LIST: # Normalize mass fraction dictionary content to 1|
        if not(flag_normalization): # Breaks the loops if normalization is not requested
            break
        new_mass_fraction[key] = new_mass_fraction[key] / sum_values
        
    logger.debug(f'Mass fraction dictionary prepared for petitRADTrans use.')
    logger.debug(f'    You can adapt the values manually by mass_fraction[key] = value * np.ones_like(temperature)')
    logger.debug(f'    or scale by mass_fraction[key] = mass_fraction[key] / scale_factor')
    
    return new_mass_fraction
#%% Get a mass fraction value for a key
def _get_mass_fraction_value_for_key(key: str,
                                     mass_fraction: dict) -> float:
    """
    Gets a value given a key for the stellar mass fraction.

    Parameters
    ----------
    key : str
        Key of the petitRADTrans mass fraction dictionary.
    mass_fraction : dict
        Stellar mass fraction to assume.

    Returns
    -------
    value : float
        Value of the mass fraction for given key
    """
    # FIXME Separate each species manually, especially for molecules. 
    # FIXME Distinguish isotopes from main_iso species.
    for species, value in mass_fraction.items():
        if key.startswith(species + '_'):
            return value
        if key == species:
            return value
    
    if key in mass_fraction.keys():
        value = mass_fraction[key]
        logger.debug(f'Found key in mass_fractions keys: {key} with value: {mass_fraction[key]}')
    else:
        logger.debug(f'No value found for: {key}, using: {1E-9}')
        value = 1E-9
    
    return value

#%% Print mass abundance
def _print_mass_abundance(mass_fraction: dict):
    """
    Prints the content of the mass fraction dictionary.

    Parameters
    ----------
    mass_fraction : float
        Mass fraction which to print.
    """
    for key in mass_fraction.keys():
        logger.print(f'Abundance of species {key}: {mass_fraction[key]}')
#%% Shifting between air and vacuum wavelengths
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

#%% List species available
def list_species_list():
    """
    Print all available species for high-resolution templates using petitRADTRANS.
    """    
    logger.print('Species available for HR modeling using petitRADTRANS')
    for item in SPECIES_LIST:
        logger.print(f'{item}')
        
#%% Setup T-P profiles
def _T_P_profile_guillot(SystemParameters: para.SystemParametersComposite,
                         atmosphere: Radtrans) -> np.ndarray:
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
    
    # DOCUMENTME
    # CLEANME 
    # FIXME Improve the pressure assumption.
    pressure = np.logspace(-10, 2, 130)
    atmosphere.setup_opa_structure(pressure)
    gravity = SystemParameters.Planet.gravity_acceleration.convert_unit_to(u.cm/u.s/u.s).data
    
    # FIXME Check what are these values and see if they should be played with instead of being hardcoded
    kappa_IR = 0.01
    gamma = 0.4
    T_int = 300.
    
    T_equ = SystemParameters.Planet.equilibrium_temperature.data

    temperature = guillot_global(P= pressure,
                                 kappa_IR= kappa_IR,
                                 gamma= gamma,
                                 grav= gravity,
                                 T_int= T_int,
                                 T_equ= T_equ)
    return temperature


#%% Interpolate to spectrum wavelength grid
def _interpolate_on_spectrum_grid(spectral_axis: sp.spectra.spectral_axis.SpectralAxis,
                                  template_wavegrid: np.ndarray,
                                  template_fluxgrid: np.ndarray
                                  ) -> np.ndarray:
    """
    Interpolate the template model on spectral axis.

    Parameters
    ----------
    spectral_axis : sp.spectra.spectral_axis.SpectralAxis
        Spectral axis on which to interpolate
    template_wavegrid : np.ndarray
        Wavelength of the template.
    template_fluxgrid : np.ndarray
        Flux of the template.

    Returns
    -------
    new_flux : np.ndarray
        Interpolated flux. The template model is then wavelength = spectral_axis, flux = new_flux.
    """
    
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(template_wavegrid, template_fluxgrid)
    new_flux = cs(spectral_axis.to(u.AA).value)
    return new_flux

#%% Ploat single template
def _plot_template(template: sp.Spectrum1D,
                   fig: None | mpl.figure.Figure = None,
                   ax: None | plt.Axes = None):
    # DOCUMENTME
    # TESTME
    if fig is None:
        fig, ax = plt.subplots(1)

    ax.plot(template.spectral_axis.value,
            template.flux.value,
            label = f'{template.meta["species"]}',
            alpha= 0.6,
            )
    
    ax.set_xlabel('Wavelength $\AA$')
    ax.set_ylabel('Planet radius $[R_{jup}]$')
    ax.legend()
    
    return fig, ax

#%% Plot all templates together
def plot_all_templates(template_list: sp.SpectrumList):
    # DOCUMENTME
    # TESTME
    for template in template_list:
        fig, ax = plt.subplots(1)
        fig, ax = _plot_template(template= template,
                                 fig= fig,
                                 ax= ax)
        
        fig.savefig(os.getcwd() + f'/figures/CCF_{template.meta["species"]}')
        plt.close(fig= fig)
    return
#%% Print detectability
def print_detectability(template_list: sp.SpectrumList):
    """
    Print out the detectability of all species.

    Parameters
    ----------
    template_list : sp.SpectrumList
        Template list with all the species.
    """
    for template in template_list:
        logger.print(f'Species: {template.meta["species"]} is detectable as {template.meta["CCF_detectability"]}')
    logger.warning('    This is a custom approximation of detectability, which is just a simple sum of the CCF model - continuum.')
    
    return

def _subtract_continuum(model: sp.Spectrum1D):

    continuum = sp.Spectrum1D(
            spectral_axis = model.spectral_axis,
            flux = np.array(
                pd.Series(model.flux.value).rolling(
                    7500,
                    min_periods=1,
                    center=True
                    ).quantile(0.25)
                )*model.flux.unit)
    
    model = model.subtract(continuum)
    
    return model

#%% Create a single template with petitRADtrans
def _create_single_template(SystemParameters: para.SystemParametersComposite,
                            spectral_axis: sp.spectra.spectral_axis.SpectralAxis,
                            species: str | list,
                            MMW_value : float = 2.33,
                            P0: float = 1,
                            ):
    # DOCUMENTME
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(spectral_axis), 'The spectral axis is not sorted, which is assumed for this function. Please sort the wavelength grid prior to creating templates.'
    
    spectral_axis_old = spectral_axis
    spectral_axis = airtovac(spectral_axis.value) * spectral_axis.unit
    
    atmosphere = Radtrans(line_species = list(species),
        wlen_bords_micron = [spectral_axis[0].to(u.um).value, # Spectrum wavelength-range
                             spectral_axis[-1].to(u.um).value], # Spectrum wavelength-range
        mode = 'lbl',
        )
    
    # System parameters
    R_pl = SystemParameters.Planet.radius.convert_unit_to(u.cm).data
    gravity = SystemParameters.Planet.gravity_acceleration.convert_unit_to(u.cm/u.s/u.s).data
    temperature = _T_P_profile_guillot(SystemParameters= SystemParameters,
                                       atmosphere= atmosphere)
    
    MMW = MMW_value * np.ones_like(temperature)
    mass_fraction = _prepare_mass_fraction_for_petitRADTrans(ABUNDANCE_SOLAR_METALLICITY,
                                                             temperature,
                                                             )

    atmosphere.calc_transm(temperature,
                           mass_fraction,
                           gravity,
                           MMW,
                           R_pl= R_pl,
                           P0_bar=P0
                           )
    
    wavelength_grid =  nc.c/atmosphere.freq/1e-4 * 10000 * u.AA
    wavelength_grid = vactoair(wavelength_grid.value) * wavelength_grid.unit
    flux = atmosphere.transm_rad/nc.r_jup_mean
    
    template = sp.Spectrum1D(
        spectral_axis = wavelength_grid,
        flux = flux * u.R_jup,
        uncertainty= astropy.nddata.StdDevUncertainty(np.zeros_like(flux)),
        mask = np.zeros_like(flux),
        meta = {
            'model': 'petitRADtrans',
            'species': species,
            }
        )

    template = _subtract_continuum(template)
    
    flux_interpolate = sci.interpolate.CubicSpline(
        template.spectral_axis.value,
        template.flux.value,
        extrapolate= False
        )
    
    template = sp.Spectrum1D(
        spectral_axis = spectral_axis_old,
        flux = flux_interpolate(spectral_axis_old) * u.R_jup,
        uncertainty= astropy.nddata.StdDevUncertainty(np.zeros_like(spectral_axis_old)),
        mask = np.zeros_like(spectral_axis_old),
        meta = {
            'model': 'petitRADtrans',
            'species': species,
            }
        )

    return template

#%% Create all available templates
def create_all_available_templates(SystemParameters: para.SystemParametersComposite,
                                   spectral_axis: sp.spectra.spectral_axis.SpectralAxis,
                                   species_list: None | list = None,
                                   MMW_value: float = 2.33,
                                   ) -> sp.SpectrumList:
    
    if species_list is None:
        species_list = SPECIES_LIST
    
    template_list = sp.SpectrumList()
    number_of_species_in_list = len(species_list)
    
    for ind, species in enumerate(species_list):
        logger.info(f'Calculation of species: {species}')
        logger.info(f'    Progress in species list: {ind+1} out of {number_of_species_in_list}')
        template = _create_single_template(SystemParameters= SystemParameters,
                                           spectral_axis= spectral_axis,
                                           species= species,
                                           MMW_value= MMW_value
                                           )
        
        if template.meta['CCF_detectability'] != 0:
            logger.print(f'Added template for species {species} in template list.')
            template_list.append(template)
    
    return template_list

def _create_template_grid(
    SystemParameters: para.SystemParametersComposite,
    spectral_axis: sp.spectra.spectral_axis.SpectralAxis,
    species_list: None | list = None,
    MMW_value: float = 2.33,
    ):
    
    abundance = np.logspace(-11, -1, 10)
    
    
    
    return

#%%
if __name__ == '__main__':
    os.chdir('/media/chamaeleontis/Observatory_main/Analysis_dataset/rats_test')
    logger.info('Starting test of rats.modeling_CCF:')
    logger.info('Loading parameters of TOI-132 b system')
    
    # Testing for TOI-132b system
    TOI132b = para.SystemParametersComposite(
        filename= os.getcwd() + '/saved_data/system_parameters.pkl'
        )
    TOI132b.load_NASA_CompositeTable_values(planet_name = 'TOI-132 b')
    TOI132b.print_main_values()
    
    logger.info('Initialization of petitRADTRANS')
    _check_abundance_sum()
    


    # _test_species_validity(SPECIES_LIST)
    # list_species_list()

    spectral_axis = np.linspace(3000, 8000, 400000) *u.AA
    template_list = create_all_available_templates(SystemParameters=TOI132b,
                                                   spectral_axis=spectral_axis,
                                                   species_list= None,
                                                   MMW_value= 2.33,
                                                   force_empty= False,
                                                   force_load= True,
                                                   force_skip= False,
                                                   pkl_name= '/templates_good_petitradtrans.pkl'
                                                   )
    
    print_detectability(template_list)
    plot_all_templates(template_list= template_list)
    
    logger.info('Test of rats.modeling_CCF successful')
    

