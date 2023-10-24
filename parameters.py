# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 08:34:27 2021

@author: Michal Steiner


"""
#%% Importing libraries
import pyvo as vo
from astropy.nddata import NDData, StdDevUncertainty, NDDataArray
import numpy as np
import os
import re
from html import unescape
# import astropy.units as u
from rats.utilities import default_logger_format
from dataclasses import dataclass
import pandas as pd
import logging
#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

#%% Load Composite table for given system through TAP query
def _load_NASA_CompositeTable(system_name: str | None = None,
                              planet_name: str | None = None) -> pd.DataFrame:
    """
    Load the NASA Exoplanet Composite Table using TAP query (https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html).

    Parameters
    ----------
    system_name : str | None, optional
        System name which to query, by default None. If None and planet_name = None, this will provide full composite table. Otherwise, only rows of the system will be passed.
    planet_name : str | None, optional
        Planet name which to query, by default None. If None, will use the system_name keyword instead. If given planet_name, only a single row will be passed.

    Returns
    -------
    pd.DataFrame
        Composite Table as loaded from the NASA archive. If planet_name is defined, will provide single row of parameters for given planet. If system_name is defined while planet_name = None, the number of rows equal number of planets in the system. Otherwise the number of rows equal the number of detected planets
    """
    service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP/") # Initialization of TAP service
    
    
    if system_name is None and planet_name is None:
        logger.warning('Requesting NASA Exoplanet Composite table from the TAP service. This can take a long time (~10 min).')
        CompositeTable = pd.DataFrame(service.search("SELECT * FROM pscomppars"))
    elif not(planet_name is None):
        logger.info('Loading NASA Exoplanet Composite table for planet: '+ planet_name)
        planet_name = "'%s'"%planet_name # To have it in correct format for the search
        CompositeTable = pd.DataFrame(service.search("SELECT * FROM pscomppars WHERE pl_name = %s"% planet_name))
    else:
        logger.info('Loading NASA Exoplanet Composite table for system: '+ system_name)
        system_name = "'%s'"%system_name # To have it in correct format for the search
        CompositeTable = pd.DataFrame(service.search("SELECT * FROM pscomppars WHERE hostname = %s"% system_name))
    logger.info("Loading finished.")
    return CompositeTable
#%% Load NASA Full Table through TAP query
def _load_NASA_FullTable(planet_name: str | None = None) -> pd.DataFrame:
    """
    Load the NASA Exoplanet Table (full table) using TAP query (https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html).

    Parameters
    ----------
    planet_name : str | None, optional
        Planet name, by default None. If None, will load full table. Otherwise, only rows of the given planet will be passed.

    Returns
    -------
    pd.DataFrame
        Full NASA Exoplanet Table with the requested rows.
    """
    service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP/") # Initialization of TAP service
    
    if planet_name is None:
        logger.warning('Requesting full NASA Exoplanet table from the TAP service. This will take a long time (~1hour).')
        logger.info('    This is not expected usage for this function. Please consider whether full NASA Exoplanet Table is actually needed.')
        logger.info('Loading full NASA Exoplanet table for all systems.')
        FullTable = pd.DataFrame(service.search("SELECT * FROM ps"))
    else:
        logger.info('Loading NASA Exoplanet Full Table for planet: '+ planet_name)
        planet_name = "'%s'"%planet_name # To have it in correct format for the search
        FullTable = pd.DataFrame(service.search("SELECT * FROM pscomppars WHERE pl_name = %s"% planet_name))
    logger.info("Loading finished.")
    return FullTable
#%% Convenience function to extract NDDataArray from Composite table
def _load_array_from_CompositeTable(CompositeTableRow: pd.DataFrame,
                                    keyword: str,
                                    parameter_name: str) -> NDDataArray:
    """
    Load a NDDataArray from NASA Table (either Full or Composite) given a keyword.

    Parameters
    ----------
    CompositeTableRow : pd.DataFrame
        NASA Exoplanet Table (Composite) from which to load the keyword.
    keyword : str
        Keyword to load.
    parameter_name : str
        Name of the parameter of the keyword.

    Returns
    -------
    NDDataArray
        Array including the uncertainty (if defined for a key), and reference in meta dictionary (if defined). NDDataArray supports error propagation.
    """
    if keyword + 'err1' in CompositeTableRow.keys():
        logger.debug('Loading key:' + keyword)
        logger.debug('Value:' +  str(CompositeTableRow[keyword]))
        logger.debug('Lower error:' + str(CompositeTableRow[keyword + 'err1']))
        logger.debug('Upper error:' + str(CompositeTableRow[keyword + 'err2']))
        logger.debug('Reference:' + str(CompositeTableRow[keyword+'_reflink']))
        parameter = NDDataArray(
            data= CompositeTableRow[keyword],
            uncertainty= StdDevUncertainty(
                np.max([CompositeTableRow[keyword + 'err1'],
                        CompositeTableRow[keyword + 'err2']]
                        )
                ),
            meta= {'reference': CompositeTableRow[keyword+'_reflink'],
                   'parameter': parameter_name}
            )
    elif keyword + '_reflink' in CompositeTableRow.keys():
        logger.debug('Loading key:' + keyword)
        logger.debug('Value:' + str(CompositeTableRow[keyword]))
        logger.debug('Reference:' + str(CompositeTableRow[keyword+'_reflink']))
        parameter = NDDataArray(
            data= CompositeTableRow[keyword],
            meta= {'reference': CompositeTableRow[keyword+'_reflink'],
                   'parameter': parameter_name}
            )
    else:
        logger.debug('Loading key:' + keyword)
        logger.debug('Value:' + str(CompositeTableRow[keyword]))
        parameter = NDDataArray(
            data= CompositeTableRow[keyword],
            meta= {'parameter': parameter_name}
            )
    return parameter
#%% Convenience function to extract NDDataArray from Full table
def _load_array_from_FullTable(FullTableRow: pd.DataFrame,
                               keyword: str,
                               parameter_name: str,
                               ) -> NDDataArray:
    """
    Convenience function to load NDDataArray from Full Table loaded from NASA Exoplanet archive.

    Parameters
    ----------
    FullTableRow : pd.DataFrame
        NASA Exoplanet Table (Full) from which to load the keyword
    keyword : str
        Keyword to load.
    parameter_name : str
        Name of the parameter of the keyword.

    Returns
    -------
    NDDataArray
        Array including the uncertainty (if defined for a key), and reference in meta dictionary (if defined). NDDataArray supports error propagation.
    """
    
    assert (keyword[:2] + '_refname') in FullTableRow.keys(), 'Keyword %s does not have a reference.'%(keyword)
        
    if (keyword[:2] + '_refname') in FullTableRow.keys():
        logger.debug('Loading key:' + keyword)
        logger.debug('Value:' + str(FullTableRow[keyword]))
        logger.debug('Lower error:' + str(FullTableRow[keyword + 'err1']))
        logger.debug('Upper error:' + str(FullTableRow[keyword + 'err2']))
        logger.debug('Reference:' + str(FullTableRow[keyword+'_reflink']))
        parameter = NDDataArray(
            data= FullTableRow[keyword],
            uncertainty= StdDevUncertainty(
                np.max([FullTableRow[keyword + 'err1'],
                        FullTableRow[keyword + 'err2']]
                        )
                ),
            meta= {'reference': FullTableRow[keyword[:2]+'_reflink'],
                   'parameter': parameter}
            )
    else:
        logger.debug('Loading key:' + keyword)
        logger.debug('Value:' + str(FullTableRow[keyword]))
        logger.debug('Lower error:' + str(FullTableRow[keyword + 'err1']))
        logger.debug('Upper error:' + str(FullTableRow[keyword + 'err2']))
        logger.debug('Reference:' + str(FullTableRow[keyword+'_reflink']))
        parameter = NDDataArray(
            data= FullTableRow[keyword],
            uncertainty= StdDevUncertainty(
                np.max([FullTableRow[keyword + 'err1'],
                        FullTableRow[keyword + 'err2']]
                        )
                ),
            meta= {'reference': FullTableRow[keyword[:2]+'_reflink'],
                   'parameter': parameter_name}
            )
    return parameter
#%% Convenience function for printing parameters
def _print_NDDataArray(Array: NDDataArray):
    # Removes the html link and unescape HTML encoded characters.
    pattern = re.compile('<.*?>')
    reference = re.sub(pattern, '', Array.meta["reference"])
    reference = unescape(reference)
    # Print through logger.print level
    if Array.uncertainty is None:
        logger.print(f'{Array.meta["parameter"]}: {Array.data} | {reference}')
    else:
        logger.print(f'{Array.meta["parameter"]}: {Array.data} ± {Array.uncertainty.array} | {reference}')
    return

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class _Magnitudes():
    #TODO
    pass
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class _Catalogues():
    #TODO
    pass
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class _PlanetDiscovery():
    #TODO
    method: str = ''
    year: int = ''
    reference: str = ''
    publication_date: str = ''
    locale: str = ''
    facility: str = ''
    instrument: str = ''
    
    def _load_values_from_NASATable(self):
        # TODO
        return
    
    pass


class _DetectionFlags():
    #TODO
    pass
    

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class _StellarParameters():
    name: str | None = None
    stellar_type: NDDataArray | None = None
    temperature: NDDataArray | None = None
    radius: NDDataArray | None = None
    mass: NDDataArray | None = None
    luminosity: NDDataArray | None = None
    metallicity: NDDataArray | None = None
    logg: NDDataArray | None = None
    age: NDDataArray | None = None
    density: NDDataArray | None = None
    vsini: NDDataArray | None = None
    rotation_period: NDDataArray | None = None
    magnitudes: _Magnitudes | None = None
    catalogues: _Catalogues | None = None
    
    def _load_values_from_composite_table(self,
                                          CompositeTableRow: pd.DataFrame):
        """
        Load values from NASA Exoplanet Composite Table into class attributes.

        Parameters
        ----------
        CompositeTable : pd.DataFrame
            NASA Exoplanet Composite Table as loaded through the TAP service.
        """
        self.name = CompositeTableRow['hostname']
        self.stellar_type = _load_array_from_CompositeTable(CompositeTableRow, 'st_spectype', 'Spectral Type')
        self.temperature = _load_array_from_CompositeTable(CompositeTableRow, 'st_teff', 'Stellar Temperature')
        self.radius = _load_array_from_CompositeTable(CompositeTableRow, 'st_rad', 'Stellar radius')
        self.mass = _load_array_from_CompositeTable(CompositeTableRow, 'st_mass', 'Stellar mass')
        self.luminosity = _load_array_from_CompositeTable(CompositeTableRow, 'st_lum', 'Stellar luminosity')
        self.metallicity = _load_array_from_CompositeTable(CompositeTableRow, 'st_met', 'Stellar metalicity')
        self.logg = _load_array_from_CompositeTable(CompositeTableRow, 'st_logg', 'Stellar logg')
        self.age = _load_array_from_CompositeTable(CompositeTableRow, 'st_age', 'Stellar age')
        self.density = _load_array_from_CompositeTable(CompositeTableRow, 'st_dens', 'Stellar density')
        self.vsini = _load_array_from_CompositeTable(CompositeTableRow, 'st_vsin', 'Stellar vsini')
        self.rotation_period = _load_array_from_CompositeTable(CompositeTableRow, 'st_rotp', 'Stellar rotation period')
    
    def print_values(self):
        logger.print(f'Host star: {self.name}')
        
        STANDARD_STELLAR_LIST = [self.temperature,
                                 self.radius,
                                 self.mass,
                                 self.stellar_type,
                                 self.metallicity,
                                 self.logg,
                                 self.age,
                                 self.vsini,
                                 self.rotation_period]
        for array in STANDARD_STELLAR_LIST:
            _print_NDDataArray(array)
        
        return
    
    def create_latex_table(self):
        # TODO
        return
    
    pass
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class _PlanetParameters():
    name: str | None = None
    letter: str | None = None
    
    radius: NDDataArray | None = None
    mass: NDDataArray | None = None
    density: NDDataArray | None = None
    semimajor_axis: NDDataArray | None = None
    period: NDDataArray | None = None
    impact_parameter: NDDataArray | None = None
    inclination: NDDataArray | None = None
    eccentricity: NDDataArray | None = None
    argument_of_periastron: NDDataArray | None = None
    a_rs_ratio: NDDataArray | None = None
    rs_a_ratio: NDDataArray | None = None
    keplerian_semiamplitude: NDDataArray | None = None
    insolation_flux: NDDataArray | None = None
    equilibrium_temperature: NDDataArray | None = None
    rprs: NDDataArray | None = None
    epoch_periastron: NDDataArray | None = None
    obliquity: NDDataArray | None = None
    trueobliquity: NDDataArray | None = None
    # TODO
    discovery: _PlanetDiscovery | None = None
    detection_flags: _DetectionFlags | None = None
    
    def _load_values_from_composite_table(self,
                                          CompositeTableRow: pd.DataFrame):
        """
        Load planet parameters from NASA Exoplanet Composite Table as loaded through the TAP service.

        Parameters
        ----------
        CompositeTableRow : pd.DataFrame
            NASA Exoplanet Composite Table holding the planetary values.
        """
        self.name = CompositeTableRow['pl_name']
        self.letter = CompositeTableRow['pl_letter']
        self.radius = _load_array_from_CompositeTable(CompositeTableRow, 'pl_radj', 'Planetary radius')
        self.mass = _load_array_from_CompositeTable(CompositeTableRow, 'pl_bmassj', 'Planetary mass')
        self.density = _load_array_from_CompositeTable(CompositeTableRow, 'pl_dens', 'Planetary density')
        self.semimajor_axis = _load_array_from_CompositeTable(CompositeTableRow, 'pl_orbsmax', 'Planet semimajor-axis')
        self.period = _load_array_from_CompositeTable(CompositeTableRow, 'pl_orbper', 'Planetary period')
        self.impact_parameter = _load_array_from_CompositeTable(CompositeTableRow, 'pl_imppar', 'Planetary impact parameter')
        self.inclination = _load_array_from_CompositeTable(CompositeTableRow, 'pl_orbincl', 'Planet orbital inclination')
        self.eccentricity = _load_array_from_CompositeTable(CompositeTableRow, 'pl_orbeccen', 'Planetary eccentricity')
        self.argument_of_periastron = _load_array_from_CompositeTable(CompositeTableRow, 'pl_orblper', 'Planetary argument of periastron')
        self.a_rs_ratio = _load_array_from_CompositeTable(CompositeTableRow, 'pl_ratdor', 'Planet semimajor-axis to stellar radius ratio')
        self.rs_a_ratio = NDDataArray(1).divide(self.a_rs_ratio, handle_meta = 'first_found')
        self.rs_a_ratio.meta['parameter'] = 'Ratio of stellar radius to planet semimajor axis'
        self.keplerian_semiamplitude = _load_array_from_CompositeTable(CompositeTableRow, 'pl_rvamp', 'Keplerian semiamplitude')
        self.insolation_flux = _load_array_from_CompositeTable(CompositeTableRow, 'pl_insol', 'Insolation flux')
        self.equilibrium_temperature = _load_array_from_CompositeTable(CompositeTableRow, 'pl_eqt', 'Equilibrium temperature')
        self.rprs = _load_array_from_CompositeTable(CompositeTableRow, 'pl_ratror', 'Ratio of planet to star radius')
        self.epoch_periastron = _load_array_from_CompositeTable(CompositeTableRow, 'pl_orbtper', 'Epoch of periastron')
        self.obliquity = _load_array_from_CompositeTable(CompositeTableRow, 'pl_projobliq', 'Projected obliquity')
        self.trueobliquity = _load_array_from_CompositeTable(CompositeTableRow, 'pl_trueobliq', 'True obliquity')

    
    def print_values(self):
        logger.print(f'Planet name: {self.name}')
        
        STANDARD_PLANET_LIST = [self.radius,
                                self.mass,
                                self.density,
                                self.insolation_flux,
                                self.equilibrium_temperature,
                                self.eccentricity,
                                self.period,
                                self.obliquity,
                                self.trueobliquity]
        for array in STANDARD_PLANET_LIST:
            _print_NDDataArray(array)
        
        return
    def create_latex_table(self):
        # TODO
        return
    
    pass

class _SystemParameters():
    systemic_velocity: NDData | None = None
    distance: NDData | None = None
    number_of_stars: int | None = None
    number_of_planets: int | None = None
    number_of_moons: int | None = None
    
    total_proper_motion: NDData | None = None
    proper_motion_right_ascension: NDData | None = None
    proper_motion_declination: NDData | None = None
    parallax: NDData | None = None
    
    right_ascension: None = None
    declination: None = None
    galactic_latitude: None = None
    galactic_longitude: None = None
    
    ecliptic_latitude: None = None
    ecliptic_longitude: None = None
    
    def print_values(self):
        return
    def create_latex_table(self):
        return
    
    
    
    pass

class _EphemerisParameters():
    transit_center: NDData | None = None
    period: NDData | None = None
    transit_length: NDData | None = None
    transit_depth: NDData | None = None
    occultation_depth: NDData | None = None
    ttv_flag: bool = False
    
    def _try_load_single_value_fromFullTable(self,
                                             FullTable: pd.DataFrame,
                                             CompositeTable: pd.DataFrame,
                                             keyword: str,
                                             parameter_name: str):
        try:
            parameter = _load_array_from_FullTable(FullTableRow= FullTable,
                                                   keyword= keyword,
                                                   parameter_name= parameter_name
                                                   )
        except:
            parameter = _load_array_from_CompositeTable(CompositeTableRow= CompositeTable,
                                                        keyword= keyword,
                                                        parameter_name=parameter_name)
        return parameter
    
    def _load_values_from_FullTable(self,
                                    FullTablePlanet: pd.DataFrame,
                                    CompositeTableRow: pd.DataFrame):
        
        self.transit_center = self._try_load_single_value_fromFullTable(
            FullTable = FullTablePlanet,
            CompositeTable= CompositeTableRow,
            keyword = 'pl_tranmid',
            parameter_name= 'Transit center'
            )
        self.period = self._try_load_single_value_fromFullTable(
            FullTable = FullTablePlanet,
            CompositeTable= CompositeTableRow,
            keyword = 'pl_orbper',
            parameter_name= 'Planetary period'
            )
        self.transit_length = self._try_load_single_value_fromFullTable(
            FullTable = FullTablePlanet,
            CompositeTable= CompositeTableRow,
            keyword = 'pl_trandur',
            parameter_name= 'Transit length'
            )
        self.transit_depth = self._try_load_single_value_fromFullTable(
            FullTable = FullTablePlanet,
            CompositeTable= CompositeTableRow,
            keyword = 'pl_trandep',
            parameter_name= 'Transit depth'
            )
        self.occultation_depth = self._try_load_single_value_fromFullTable(
            FullTable = FullTablePlanet,
            CompositeTable= CompositeTableRow,
            keyword = 'pl_occdep',
            parameter_name= 'Occultation depth'
            )
        self.ttv_flag = self._try_load_single_value_fromFullTable(
            FullTable = FullTablePlanet,
            CompositeTable= CompositeTableRow,
            keyword = 'ttv_flag',
            parameter_name= 'TTV flag'
            )
        
    def print_values(self):
        logger.print(f'Ephemeris:')
        
        STANDARD_EPHEMERIS_LIST = [self.transit_center,
                                   self.period,
                                   self.transit_length,
                                   self.transit_depth]
        for array in STANDARD_EPHEMERIS_LIST:
            _print_NDDataArray(array)
        return
    def create_latex_table(self):
        # TODO
        return
    pass

class SystemParameters():
    
    
    pass

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class SystemParametersComposite():
    
    Star: _StellarParameters = _StellarParameters()
    Planet: _PlanetParameters = _PlanetParameters()
    Ephemeris: _EphemerisParameters = _EphemerisParameters()
    
    def _save(self):
        # TODO
        return
    def _load(self):
        return
    
    def load_NASA_CompositeTable_values(self,
                                        planet_name: str):
        CompositeTableRow = _load_NASA_CompositeTable(system_name = None,
                                                      planet_name = planet_name).iloc[0]
        FullTablePlanet = _load_NASA_FullTable(planet_name = planet_name)
        assert CompositeTableRow['pl_name'] == planet_name, 'Composite table does not include the planet name. This should not happen in any scenario.'
        
        self.Star._load_values_from_composite_table(CompositeTableRow= CompositeTableRow)
        self.Planet._load_values_from_composite_table(CompositeTableRow= CompositeTableRow)
        self.Ephemeris._load_values_from_FullTable(FullTablePlanet= FullTablePlanet,
                                                   CompositeTableRow= CompositeTableRow)
        return
    
    def print_main_values(self):
        # TODO
        logger.print('='*25)
        self.Star.print_values()
        logger.print('='*25)
        self.Planet.print_values()
        logger.print('='*25)
        self.Ephemeris.print_values()
        logger.print('='*25)
        return
    
    
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class SystemParametersFull():
    # TODO
    
    
    def load_NASA_FullTable_values(self):
        # TODO
        return
    pass

class EphemerisPlanet():
    
    def __init__(self,
                 ):
        
        
        
        return
    pass

class CompositeTable():
    
    def _save(self):
        
        return
    def _load():
        return
    
    
    def __init__():
        self.__path =  __file__
        
        # if os.path.exists():
        
        self.CompositeTable = _load_NASA_CompositeTable()

class FullTable():
    
    def _save():
        return
    def _load():
        return
    
    def __init__(self):
        self.FullTable = _load_NASA_FullTable()
        
    pass

if __name__ == '__main__':
    
    logger.info('Testing setup for rats.parameters module')
    os.chdir('/media/chamaeleontis/Observatory_main/Analysis_dataset/rats_test')
    system_name = 'TOI-132'
    planet_name = 'TOI-132 b'
 
    logger.info('Loading system parameters for system: ' + system_name)
    TOI132 = SystemParametersComposite()
    TOI132.load_NASA_CompositeTable_values(planet_name = 'TOI-132 b')
    TOI132.print_main_values()
    
    # CompositeTableAll = _load_NASA_CompositeTable(system_name = None)
    # FullTableTOI132 = _load_NASA_FullTable(planet_name = planet_name)
    # FullTableAll = _Load_NASA_Full_Table(planet_name = None)
    logger.info('Test succesful. All Tables were succesfully loaded:')