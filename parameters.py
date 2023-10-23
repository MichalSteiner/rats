# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 08:34:27 2021

@author: Michal Steiner


"""
#%% Importing libraries
import pyvo as vo
from astropy.nddata import NDData, StdDevUncertainty, NDDataArray
# import numpy as np
# import os
# import astropy.units as u
from rats.utilities import default_logger_format
from dataclasses import dataclass
import pandas as pd
#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

#%% Load Composite table for given system



def _load_NASA_Composite_Table(system_name: str | None = None) -> pd.DataFrame:
    """
    Load the NASA Exoplanet Composite Table using TAP query (https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html).

    Parameters
    ----------
    system_name : str | None, optional
        System name which to query, by default None If None, will provide full composite table. Otherwise, only rows of the system will be passed.

    Returns
    -------
    pd.DataFrame
        Composite Table as loaded from the NASA archive. If system_name is defined, the number of rows equal number of planets in the system. Otherwise the number of rows equal the number of detected planets
    """
    service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP/") # Initialization of TAP service
    
    if system_name is None:
        logger.warning('Requesting NASA Exoplanet Composite table from the TAP service. This can take a long time (~10 min).')
        CompositeTable = pd.DataFrame(service.search("SELECT * FROM pscomppars"))
    else:
        logger.info('Loading NASA Exoplanet Composite table for system: '+ system_name)
        system_name = "'%s'"%system_name # To have it in correct format for the search
        CompositeTable = pd.DataFrame(service.search("SELECT * FROM pscomppars WHERE hostname = %s"% system_name))
    return CompositeTable

def _Load_NASA_Full_Table(planet_name: str | None = None) -> pd.DataFrame:
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
        logger.warning('Requesting full NASA Exoplanet table from the TAP service. This will take a long time.')
        FullTable = pd.DataFrame(service.search("SELECT * FROM ps"))
    else:
        logger.info('Loading NASA Exoplanet table for planet: '+ planet_name)
        planet_name = "'%s'"%planet_name # To have it in correct format for the search
        FullTable = pd.DataFrame(service.search("SELECT * FROM pscomppars WHERE pl_name = %s"% planet_name))
    return FullTable

def _load_array_from_table(NASATable: pd.DataFrame,
                           keyword: str) -> NDDataArray:
    """
    Load a NDDataArray from NASA Table (either Full or Composite) given a keyword.

    Parameters
    ----------
    NASATable : pd.DataFrame
        NASA Exoplanet Table (Full or Composite) from which to load the keyword
    keyword : str
        Keyword to load.

    Returns
    -------
    NDDataArray
        Array including the uncertainty (if defined for a key), and reference in meta dictionary. NDDataArray supports error propagation.
    """
    if keyword + 'err1' in NASATable.keys():
        logger.debug('Loading key:' + keyword)
        logger.debug('Value:' CompositeTable[keyword])
        logger.debug('Lower error:' CompositeTable[keyword + 'err1'])
        logger.debug('Upper error:' CompositeTable[keyword + 'err2'])
        logger.debug('Reference:' CompositeTable[keyword+'_reflink'])
        parameter = NDDataArray(
            data= CompositeTable[keyword],
            uncertainty= StdDevUncertainty(
                np.max([CompositeTable[keyword + 'err1'],
                        CompositeTable[keyword + 'err2']]
                        )
                ),
            meta= {'reference': CompositeTable[keyword+'_reflink']}
            )
    else:
        logger.debug('Loading key:' + keyword)
        logger.debug('Value:' CompositeTable[keyword])
        logger.debug('Reference:' CompositeTable[keyword+'_reflink'])
        parameter = NDDataArray(
            data= CompositeTable[keyword],
            meta= {'reference': CompositeTable[keyword+'_reflink']}
            )
    return parameter

class _Magnitudes():
    pass

class _Catalogues():
    pass

class _PlanetDiscovery():
    method: str
    year: int
    reference: str
    publication_date: str
    locale: str
    facility: str
    instrument: str
    
    pass


class _DetectionFlags():
    pass
    

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)
class _StellarParameters():
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
                                          CompositeTable: pd.DataFrame):
        """
        Load values from NASA Exoplanet Composite Table into class attributes.

        Parameters
        ----------
        CompositeTable : pd.DataFrame
            NASA Exoplanet Composite Table as loaded through the TAP service.
        """
        self.stellar_type = _load_array_from_table(CompositeTable, 'st_spectype')
        self.temperature = _load_array_from_table(CompositeTable, 'st_teff')
        self.stellar_type = _load_array_from_table(CompositeTable, 'st_spectype')
        self.radius = _load_array_from_table(CompositeTable, 'st_rad')
        self.mass = _load_array_from_table(CompositeTable, 'st_mass')
        self.luminosity = _load_array_from_table(CompositeTable, 'st_lum')
        self.metallicity = _load_array_from_table(CompositeTable, 'st_met')
        self.logg = _load_array_from_table(CompositeTable, 'st_logg')
        self.age = _load_array_from_table(CompositeTable, 'st_age')
        self.density = _load_array_from_table(CompositeTable, 'st_dens')
        self.vsini = _load_array_from_table(CompositeTable, 'st_vsin')
        self.rotation_period = _load_array_from_table(CompositeTable, 'st_rotp')
    
    def print_values(self):
        return
    
    def create_latex_table(self):
        return
    
    pass
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)
class _PlanetParameters():
    name: str = ''
    letter: str = ''
    
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

    discovery: _PlanetDiscovery | None = None
    detection_flags: _DetectionFlags | None = None
    
    def _load_values_from_composite_table(self,
                                          CompositeTable: pd.DataFrame):
        """
        Load planet parameters from NASA Exoplanet Composite Table as loaded through the TAP service.

        Parameters
        ----------
        CompositeTable : pd.DataFrame
            NASA Exoplanet Composite Table holding the planetary values.
        """
        # TODO
        # Replace CompositeTable with Row
        self.radius = _load_array_from_table(CompositeTable, 'pl_radj')
        self.mass = _load_array_from_table(CompositeTable, 'pl_bmassj')
        self.density = _load_array_from_table(CompositeTable, 'pl_dens')
        self.semimajor_axis = _load_array_from_table(CompositeTable, 'pl_orbsmax')
        self.period = _load_array_from_table(CompositeTable, 'pl_orbper')
        self.impact_parameter = _load_array_from_table(CompositeTable, 'pl_imppar')
        self.inclination = _load_array_from_table(CompositeTable, 'pl_orbincl')
        self.eccentricity = _load_array_from_table(CompositeTable, 'pl_orbeccen')
        self.argument_of_periastron = _load_array_from_table(CompositeTable, 'pl_orblper')
        self.a_rs_ratio = _load_array_from_table(CompositeTable, 'pl_ratdor')
        self.rs_a_ratio = 1/ self.a_rs_ratio
        self.keplerian_semiamplitude = _load_array_from_table(CompositeTable, 'pl_rvamp')
        self.insolation_flux = _load_array_from_table(CompositeTable, 'pl_insol')
        self.equilibrium_temperature = _load_array_from_table(CompositeTable, 'pl_eqt')
        self.rprs = _load_array_from_table(CompositeTable, 'pl_ratror')
        self.epoch_periastron = _load_array_from_table(CompositeTable, 'pl_orbtper')
        self.obliquity = _load_array_from_table(CompositeTable, 'pl_projobliq')
        self.trueobliquity = _load_array_from_table(CompositeTable, 'pl_trueobliq')

    
    def print_values(self):
        return
    def create_latex_table(self):
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
    
    
    
    def print_values(self):
        return
    def create_latex_table(self):
        return
    pass

class SystemParameters():
    
    
    pass

class SystemParametersComposite():
    pass


class SystemParametersFull():
    pass

class EphemerisPlanet():
    pass

class CompositeTable():
    pass

class FullTable():
    pass
