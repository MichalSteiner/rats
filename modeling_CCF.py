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
import os
import specutils as sp
import numpy as np
import astropy.units as u
from petitRADTRANS.physics import guillot_global
from rats.utilities import default_logger_format
import logging
import rats.parameters as para
from rats.utilities import time_function, save_and_load, progress_tracker, skip_function, disable_func, default_logger_format, todo_function

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
    logger.info('Added species in list:')
    if os.path.isdir(OPACITY_LIST_LOCATION + '/line_by_line/' + item):
        SPECIES_LIST.append(item)
        logger.info(f'    {item}')

def list_species_list():
    logger.print('Species available for HR modeling using petitRADTRANS')
    for item in SPECIES_LIST:
        logger.print(f'{item}')

#%%
def create_template(SystemParameters: para.SystemParametersComposite,
                    species: list | str,
                    spectral_axis: sp.spectra.spectral_axis.SpectralAxis
                    ):
    
    
    
    
    return

#%% 
def _prepare_atmosphere_model(species: list,
                              rayleigh_species):
    atmosphere = Radtrans(line_species = species,
                          )
    
    
    return

#%%
def _test_species_validity(species: list):
    logger.info('Test of species validity in petitRADTRANS')
    for spec in species:
        logger.info(f'    Testing species: {spec}')
        try:
            atmosphere = Radtrans(line_species = [
                spec
                ],
                wlen_bords_micron = [0.3, 0.8],
                mode = 'lbl'
                )
        except:
            location = OPACITY_LIST_LOCATION + '/line_by_line/' + spec
            logger.critical(f'    Loading of species {spec} located in:')
            logger.critical(f'        {location}')
            logger.critical('    failed')
    logger.info('Testing species validity finished.')
    return

#%%
if __name__ == '__main__':
    logger.info('Starting test of rats.modeling_CCF:')
    logger.info('Loading parameters of TOI-132 b system')
    TOI132 = para.SystemParametersComposite()
    TOI132.load_NASA_CompositeTable_values(planet_name = 'TOI-132 b')
    TOI132.print_main_values()
    
    logger.info('Initialization of petitRADTRANS')
    
    # _test_species_validity(SPECIES_LIST)
    list_species_list()
    

    
    atmosphere = Radtrans(line_species = [
        'H2O_main_iso',
        'CO_all_iso',
        'CH4_main_iso',
        'CO2_main_iso',
        'Na_allard',
        'K'
        ],
        rayleigh_species = ['H2', 'He'],
        continuum_opacities = ['H2-H2', 'H2-He'],
        wlen_bords_micron = [0.3, 0.8],
        mode = 'lbl'
        )
    atmosphere_continuum = Radtrans(line_species = [
        'H2O_main_iso',
        'CO_all_iso',
        'CH4_main_iso',
        'CO2_main_iso',
        # 'Na_allard',
        'K'
        ],
        rayleigh_species = ['H2', 'He'],
        continuum_opacities = ['H2-H2', 'H2-He'],
        wlen_bords_micron = [0.3, 0.8],
        mode = 'lbl'
        )

    pressures = np.logspace(-10, 2, 130)
    atmosphere.setup_opa_structure(pressures)



    R_pl = TOI132.Planet.radius.convert_unit_to(u.cm).data
    P0 = 0.01
    TOI132.Planet._calculate_gravity_acceleration()
    gravity = TOI132.Planet.gravity_acceleration.to(u.cm/u.s/u.s).value
    
    kappa_IR = 0.01
    gamma = 0.4
    T_int = 200.
    T_equ = 1500.

    temperature = guillot_global(P= pressures,
                                 kappa_IR= kappa_IR,
                                 gamma= gamma,
                                 grav= gravity,
                                 T_int= T_int,
                                 T_equ= T_equ)
    
    mass_fractions = {}
    mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
    mass_fractions['He'] = 0.24 * np.ones_like(temperature)
    mass_fractions['H2O_main_iso'] = 0.001 * np.ones_like(temperature)
    mass_fractions['CO_all_iso'] = 0.01 * np.ones_like(temperature)
    mass_fractions['CO2_main_iso'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['CH4_main_iso'] = 0.000001 * np.ones_like(temperature)
    mass_fractions['Na_allard'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['K'] = 0.000001 * np.ones_like(temperature)

    MMW = 2.33 * np.ones_like(temperature)
    atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0)
    atmosphere_continuum.calc_transm(temperature, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0)

    stop
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)

    ax.plot(nc.c/atmosphere.freq/1e-4 * 10000, atmosphere.transm_rad/nc.r_jup_mean - atmosphere_continuum.transm_rad/nc.r_jup_mean)
    # ax.plot(nc.c/atmosphere_continuum.freq/1e-4 * 10000, atmosphere_continuum.transm_rad/nc.r_jup_mean)

    ax.set_xlabel('Wavelength (microns)')
    ax.set_ylabel(r'Transit radius ($\rm R_{Jup}$)')
    fig.show()

    