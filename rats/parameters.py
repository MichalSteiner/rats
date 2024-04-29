# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 08:34:27 2021

@author: Michal Steiner


"""
#%% Importing libraries
import rats.parameters_subroutines.utilities as parautils
import pyvo as vo
from astropy.nddata import NDData, StdDevUncertainty, NDDataArray
import astropy.time as astime
import astropy.units as u
import datetime
import numpy as np
import os
import re
import dill as pickle
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
        FullTable = pd.DataFrame(service.search("SELECT * FROM ps WHERE pl_name = %s"% planet_name))
    logger.info("Loading finished.")
    return FullTable
#%% Convenience function to extract NDData from Composite table
def _load_array_from_CompositeTable(CompositeTableRow: pd.DataFrame,
                                    keyword: str,
                                    parameter_name: str) -> NDDataArray:
    """
    Load a NDDataArray from NASA Composite Table given a keyword.

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
    
    if (keyword + 'err1' in CompositeTableRow.keys() and
        keyword + '_reflink' in CompositeTableRow.keys()):
        logger.debug('Loading key:' + keyword)
        logger.debug('Value:' +  str(CompositeTableRow[keyword]))
        logger.debug('Lower error:' + str(CompositeTableRow[keyword + 'err1']))
        logger.debug('Upper error:' + str(CompositeTableRow[keyword + 'err2']))
        logger.debug('Reference:' + str(CompositeTableRow[keyword+'_reflink']))
        parameter = NDDataArray(
            data= CompositeTableRow[keyword],
            uncertainty= StdDevUncertainty(
                np.nanmax([CompositeTableRow[keyword + 'err1'],
                        CompositeTableRow[keyword + 'err2']]
                        )
                ),
            meta= {'reference': CompositeTableRow[keyword+'_reflink'],
                   'parameter': parameter_name}
            )
    elif keyword + 'err1' in CompositeTableRow.keys():
        logger.debug('Loading key:' + keyword)
        logger.debug('Value:' +  str(CompositeTableRow[keyword]))
        logger.debug('Lower error:' + str(CompositeTableRow[keyword + 'err1']))
        logger.debug('Upper error:' + str(CompositeTableRow[keyword + 'err2']))
        parameter = NDDataArray(
            data= CompositeTableRow[keyword],
            uncertainty= StdDevUncertainty(
                np.nanmax([CompositeTableRow[keyword + 'err1'],
                        CompositeTableRow[keyword + 'err2']]
                        )
                ),
            meta= {'reference': 'unknown',
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
        logger.debug('Reference:' + str(FullTableRow[keyword[:2]+'_refname']))
        parameter = NDDataArray(
            data= FullTableRow[keyword],
            uncertainty= StdDevUncertainty(
                np.max([FullTableRow[keyword + 'err1'],
                        FullTableRow[keyword + 'err2']]
                        )
                ),
            meta= {'reference': FullTableRow[keyword[:2]+'_refname'],
                   'parameter': parameter_name}
            )
    else:
        logger.debug('Loading key:' + keyword)
        logger.debug('Value:' + str(FullTableRow[keyword]))
        logger.debug('Lower error:' + str(FullTableRow[keyword + 'err1']))
        logger.debug('Upper error:' + str(FullTableRow[keyword + 'err2']))
        logger.debug('Reference:' + str(FullTableRow[keyword[:2]+'_refname']))
        parameter = NDDataArray(
            data= FullTableRow[keyword],
            uncertainty= StdDevUncertainty(
                np.max([FullTableRow[keyword + 'err1'],
                        FullTableRow[keyword + 'err2']]
                        )
                ),
            meta= {'reference': FullTableRow[keyword[:2]+'_refname'],
                   'parameter': parameter_name}
            )
    return parameter
#%% Convenience function for printing parameters
def _extract_reference(reference_string: str) -> str:
    """
    Extract reference string for printing from the loaded keyword of NASA Table.
    
    First, this code removes the HTML tags with the link, and then decodes (unescape) the HTML encoded characters. 

    Parameters
    ----------
    reference_string : str
        Reference string as present in various instances of reference values.

    Returns
    -------
    str
        Reference string suitable for printing.
    """
    pattern = re.compile('<.*?>')
    reference = re.sub(pattern, '', reference_string)
    reference = unescape(reference)
    return reference
#%% Print NDDataArray values and reference
def _print_NDDataArray(Array: NDDataArray):
    """
    Print a NDDataArray through the logging PRINT level (custom defined by rats.utilities).

    Parameters
    ----------
    Array : NDDataArray
        Parameter array including the parameter name, uncertainty and reference.
    """
    # Removes the html link and unescape HTML encoded characters.
    reference = _extract_reference(Array.meta["reference"])
    # If unitless, change string None to "1"
    if Array.unit is None:
        unit = '1'
    else:
        unit = Array.unit
    # Print through logger.print level
    if Array.uncertainty is None:
        logger.print(f'{Array.meta["parameter"]}: {Array.data} [{unit}] | {reference}')
    else:
        logger.print(f'{Array.meta["parameter"]}: {Array.data} ± {Array.uncertainty.array} [{unit}] | {reference}')
    return
#%% Magnitudes class
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class _Magnitudes():
    B: NDDataArray | None = None
    V: NDDataArray | None = None
    J: NDDataArray | None = None
    H: NDDataArray | None = None
    K: NDDataArray | None = None
    u: NDDataArray | None = None
    g: NDDataArray | None = None
    r: NDDataArray | None = None
    i: NDDataArray | None = None
    z: NDDataArray | None = None
    
    w1: NDDataArray | None = None
    w2: NDDataArray | None = None
    w3: NDDataArray | None = None
    w4: NDDataArray | None = None
    gaia: NDDataArray | None = None
    ic: NDDataArray | None = None
    TESS: NDDataArray | None = None
    Kepler: NDDataArray | None = None
    #TODO
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
        self.B = _load_array_from_CompositeTable(CompositeTableRow, 'sy_bmag', 'B (Johnson) Magnitude')
        self.V = _load_array_from_CompositeTable(CompositeTableRow, 'sy_vmag', 'V (Johnson) Magnitude')
        self.J = _load_array_from_CompositeTable(CompositeTableRow, 'sy_jmag', 'J (2MASS) Magnitude')
        self.H = _load_array_from_CompositeTable(CompositeTableRow, 'sy_hmag', 'H (2MASS) Magnitude')
        self.K = _load_array_from_CompositeTable(CompositeTableRow, 'sy_kmag', 'Ks (2MASS) Magnitude')
        self.u = _load_array_from_CompositeTable(CompositeTableRow, 'sy_umag', 'u (Sloan) Magnitude')
        self.g = _load_array_from_CompositeTable(CompositeTableRow, 'sy_gmag', 'g (Sloan) Magnitude')
        self.r = _load_array_from_CompositeTable(CompositeTableRow, 'sy_rmag', 'r (Sloan) Magnitude')
        self.i = _load_array_from_CompositeTable(CompositeTableRow, 'sy_imag', 'i (Sloan) Magnitude')
        self.z = _load_array_from_CompositeTable(CompositeTableRow, 'sy_zmag', 'z (Sloan) Magnitude')

        self.w1 = _load_array_from_CompositeTable(CompositeTableRow, 'sy_w1mag', 'W1 (WISE) Magnitude')
        self.w2 = _load_array_from_CompositeTable(CompositeTableRow, 'sy_w2mag', 'W2 (WISE) Magnitude')
        self.w3 = _load_array_from_CompositeTable(CompositeTableRow, 'sy_w3mag', 'W3 (WISE) Magnitude')
        self.w4 = _load_array_from_CompositeTable(CompositeTableRow, 'sy_w4mag', 'W4 (WISE) Magnitude')
        self.gaia = _load_array_from_CompositeTable(CompositeTableRow, 'sy_gaiamag', 'Gaia Magnitude')
        self.ic = _load_array_from_CompositeTable(CompositeTableRow, 'sy_icmag', 'I (Cousins) Magnitude')
        self.TESS = _load_array_from_CompositeTable(CompositeTableRow, 'sy_tmag', 'TESS Magnitude')
        self.Kepler = _load_array_from_CompositeTable(CompositeTableRow, 'sy_kepmag', 'Kepler Magnitude')

    pass
#%% Catalogues class
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class _Catalogues():
    #TODO
    pass
#%% PlanetDiscovery class
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
#%% Detection Flags class
class _DetectionFlags():
    #TODO
    pass
    
#%% Stellar parameters class
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class _StellarParameters(parautils.StellarModel,
                         parautils.LimbDarkening):
    """
    Class holding the stellar parameters.
    
    Each parameter is either None, if not defined, or a NDDataArray structure, with value, error, units and reference. For magnitudes and catalogues, these are either None if not defined or _Magnitudes, respectivelly _Catalogues classes.
    
    Attributes
    ----------
    name : str | None
        Name of the host star, by default None.
    stellar_type : NDDataArray | None
        Stellar type of the host star, by default None.
    temperature : NDDataArray | None
        Stellar effective temperature of the host star, by default None.
    radius : NDDataArray | None
        Stellar radius of the host star, by default None.
    mass : NDDataArray | None
        Stellar mass of the host star, by default None.
    luminosity : NDDataArray | None
        Stellar luminosity of the host star, by default None.
    metallicity : NDDataArray | None
        Stellar metallicity of the host star, by default None.
    logg : NDDataArray | None
        Stellar logg of the host star, by default None.
    age : NDDataArray | None
        Stellar age of the host star, by default None.
    density : NDDataArray | None
        Stellar density of the host star, by default None.
    vsini : NDDataArray | None
        Stellar vsini of the host star, by default None.
    rotation_period : NDDataArray | None
        Stellar rotation period of the host star, by default None.
    magnitudes : _Magnitudes | None
        Stellar magnitudes of the host star, by default None.
    catalogues : _Catalogues | None
        Other ID numbers in various Catalogues, by default None.
    """
    
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
    magnitudes: _Magnitudes | None = _Magnitudes()
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
        self.temperature.unit = u.K
        self.radius = _load_array_from_CompositeTable(CompositeTableRow, 'st_rad', 'Stellar radius')
        self.radius.unit = u.R_sun
        self.mass = _load_array_from_CompositeTable(CompositeTableRow, 'st_mass', 'Stellar mass')
        self.mass.unit = u.M_sun
        self.luminosity = _load_array_from_CompositeTable(CompositeTableRow, 'st_lum', 'Stellar luminosity')
        self.luminosity.unit = u.Lsun
        self.metallicity = _load_array_from_CompositeTable(CompositeTableRow, 'st_met', 'Stellar metalicity')
        # TODO units
        self.logg = _load_array_from_CompositeTable(CompositeTableRow, 'st_logg', 'Stellar logg')
        self.logg.unit = u.cm / u.s / u.s
        self.age = _load_array_from_CompositeTable(CompositeTableRow, 'st_age', 'Stellar age')
        self.age.unit = u.Gyr
        self.density = _load_array_from_CompositeTable(CompositeTableRow, 'st_dens', 'Stellar density')
        self.density.unit = u.g / u.cm / u.cm / u.cm
        self.vsini = _load_array_from_CompositeTable(CompositeTableRow, 'st_vsin', 'Stellar vsini')
        self.vsini.unit = u.km / u.s
        self.rotation_period = _load_array_from_CompositeTable(CompositeTableRow, 'st_rotp', 'Stellar rotation period')
        self.rotation_period.unit = u.d
        self.magnitudes._load_values_from_composite_table(CompositeTableRow)
        
    def print_values(self):
        """
        Print a default set of parameters using the logging PRINT level (custom defined by rats.utilities)
        """
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
#%% Planet parameters class
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class _PlanetParameters(parautils.CalculationPlanet):
    """
    Class holding planet parameters.
    
    Each parameter is either None, if not defined, or a NDDataArray structure, with value, error, units and reference. 
    
    Attributes
    ----------
    name : str | None
        Name of the planet, by default None.
    letter : str | None
        Letter of the planet, by default None.
    radius : NDDataArray | None
        Radius of the planet, by default None.
    mass : NDDataArray | None
        Mass of the planet, by default None.
    density : NDDataArray | None
        Density of the planet, by default None.
    semimajor_axis : NDDataArray | None
        Semimajor axis of the planetary orbit, by default None.
    period : NDDataArray | None
        Orbital period of the planet, by default None.
    impact_parameter : NDDataArray | None
        Impact parameter of the planetary orbit, by default None.
    inclination : NDDataArray | None
        Inclination of the planetary orbit, by default None.
    eccentricity : NDDataArray | None
        Eccentricity of the planetary orbit, by default None.
    argument_of_periastron : NDDataArray | None
        Argument of periastron of the planetary orbit, by default None.
    a_rs_ratio : NDDataArray | None
        Ratio of semimajor-axis over stellar radii, by default None.
    rs_a_ratio : NDDataArray | None
        Ratio of stellar radii over semimajor axis, by default None.
    keplerian_semiamplitude : NDDataArray | None
        Semiamplitude of the keplerian orbit, by default None.
    insolation_flux : NDDataArray | None
        Insolation flux of the planet, by default None.
    equilibrium_temperature : NDDataArray | None
        Equilibrium temperature of the planet, by default None.
    rprs : NDDataArray | None
        Ratio of planetary radii to stellar radii, by default None.
    epoch_periastron : NDDataArray | None
        Epoch of periastron of the planetary orbit, by default None.
    projected_obliquity : NDDataArray | None
        Projected obliquity of the planetary orbit, by default None.
    true_obliquity : NDDataArray | None
        True obliquity of the planetary orbit, by default None.
    discovery : _PlanetDiscovery | None
        Basic information about discovery of this planet, by default None.
    detection_flags : _DetectionFlags | None
        Detection flags about the planets, by default None.
    """
    
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
    projected_obliquity: NDDataArray | None = None
    true_obliquity: NDDataArray | None = None
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
        self.radius.unit = u.R_jupiter
        self.mass = _load_array_from_CompositeTable(CompositeTableRow, 'pl_bmassj', 'Planetary mass')
        self.mass.unit = u.M_jupiter
        self.density = _load_array_from_CompositeTable(CompositeTableRow, 'pl_dens', 'Planetary density')
        self.density.unit = u.g / (u.cm**3)
        self.semimajor_axis = _load_array_from_CompositeTable(CompositeTableRow, 'pl_orbsmax', 'Planet semimajor-axis')
        self.semimajor_axis.unit = u.au
        self.impact_parameter = _load_array_from_CompositeTable(CompositeTableRow, 'pl_imppar', 'Planetary impact parameter')
        self.impact_parameter.unit = u.dimensionless_unscaled
        self.inclination = _load_array_from_CompositeTable(CompositeTableRow, 'pl_orbincl', 'Planet orbital inclination')
        self.inclination.unit = u.deg
        self.eccentricity = _load_array_from_CompositeTable(CompositeTableRow, 'pl_orbeccen', 'Planetary eccentricity')
        self.eccentricity.unit = u.dimensionless_unscaled
        self.argument_of_periastron = _load_array_from_CompositeTable(CompositeTableRow, 'pl_orblper', 'Planetary argument of periastron')
        self.argument_of_periastron.unit = u.deg
        if self.eccentricity.data == 0:
            self.argument_of_periastron = NDDataArray(
                data= 90,
                unit= u.deg,
                uncertainty= StdDevUncertainty(0),
                meta= {
                    'reference': 'Assumed',
                    'parameter': 'Planetary argument of periastron',
                    'details': 'Assumed 90 degrees as eccentricity is 0'
                    }
            )
        self.a_rs_ratio = _load_array_from_CompositeTable(CompositeTableRow, 'pl_ratdor', 'Planet semimajor-axis to stellar radius ratio')
        self.a_rs_ratio.unit = u.dimensionless_unscaled
        self.keplerian_semiamplitude = _load_array_from_CompositeTable(CompositeTableRow, 'pl_rvamp', 'Keplerian semiamplitude')
        self.keplerian_semiamplitude.unit = u.m / u.s
        self.insolation_flux = _load_array_from_CompositeTable(CompositeTableRow, 'pl_insol', 'Insolation flux')
        # TODO
        logger.warning('Add units for insolation flux')
        self.equilibrium_temperature = _load_array_from_CompositeTable(CompositeTableRow, 'pl_eqt', 'Equilibrium temperature')
        self.equilibrium_temperature.unit = u.K
        self.rprs = _load_array_from_CompositeTable(CompositeTableRow, 'pl_ratror', 'Ratio of planet to star radius')
        self.rprs.unit = u.dimensionless_unscaled
        self.epoch_periastron = _load_array_from_CompositeTable(CompositeTableRow, 'pl_orbtper', 'Epoch of periastron')
        self.epoch_periastron.unit = u.deg
        self.projected_obliquity = _load_array_from_CompositeTable(CompositeTableRow, 'pl_projobliq', 'Projected obliquity')
        self.projected_obliquity.unit = u.deg
        self.true_obliquity = _load_array_from_CompositeTable(CompositeTableRow, 'pl_trueobliq', 'True obliquity')
        self.true_obliquity.unit = u.deg
        self.rs_a_ratio = NDDataArray(1, unit= u.dimensionless_unscaled).divide(self.a_rs_ratio, handle_meta = 'first_found')
        self.rs_a_ratio.meta['parameter'] = 'Ratio of stellar radius to planet semimajor axis'


        self._calculate_gravity_acceleration()
    
    def print_values(self):
        """
        Print a default set of parameters using the logging PRINT level (custom defined by rats.utilities)
        """
        logger.print(f'Planet name: {self.name}')
        
        STANDARD_PLANET_LIST = [self.radius,
                                self.mass,
                                self.density,
                                self.semimajor_axis,
                                self.insolation_flux,
                                self.equilibrium_temperature,
                                self.eccentricity,
                                self.projected_obliquity,
                                self.true_obliquity]
        for array in STANDARD_PLANET_LIST:
            _print_NDDataArray(array)
        
        return
    def create_latex_table(self):
        # TODO
        return
    
    pass
#%% System parameters class
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class _SystemParameters():
    
    # TODO
    
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
    
    def _load_values_from_composite_table(self,
                                          CompositeTableRow: pd.DataFrame):
        """
        Load values from Composite table.

        Parameters
        ----------
        CompositeTableRow : pd.DataFrame
            Row for given system taken from NASA Composite table
        """
        
        
        self.systemic_velocity = _load_array_from_CompositeTable(CompositeTableRow, 'st_radv', 'Systemic velocity')
        self.systemic_velocity.unit = u.km / u.s
        self.distance = _load_array_from_CompositeTable(CompositeTableRow, 'sy_dist', 'Distance to the system')
        self.distance.unit = u.pc
        
        self.number_of_stars = _load_array_from_CompositeTable(CompositeTableRow, 'sy_snum', 'Number of stars')
        self.number_of_planets = _load_array_from_CompositeTable(CompositeTableRow, 'sy_pnum', 'Number of planets')
        self.number_of_moons = _load_array_from_CompositeTable(CompositeTableRow, 'sy_mnum', 'Number of moons')
        
        self.total_proper_motion = _load_array_from_CompositeTable(CompositeTableRow, 'sy_pm', 'Total proper motion')
        self.total_proper_motion.unit = u.mas / u.year
        self.proper_motion_right_ascension = _load_array_from_CompositeTable(CompositeTableRow, 'sy_pmra', 'Proper motion in right ascension')
        self.proper_motion_right_ascension.unit =  u.mas / u.year
        self.proper_motion_declination = _load_array_from_CompositeTable(CompositeTableRow, 'sy_pmdec', 'Proper motion in declination')
        self.proper_motion_declination.unit =  u.mas / u.year
        self.parallax = _load_array_from_CompositeTable(CompositeTableRow, 'sy_plx', 'System paralax')
        self.parallax.unit = u.mas
        self.right_ascension = _load_array_from_CompositeTable(CompositeTableRow, 'ra', 'Right ascension')
        self.declination = _load_array_from_CompositeTable(CompositeTableRow, 'dec', 'Declination')
        
        self.galactic_latitude = _load_array_from_CompositeTable(CompositeTableRow, 'glat', 'Galactic latitude')
        self.galactic_latitude.unit = u.deg
        self.galactic_longitude = _load_array_from_CompositeTable(CompositeTableRow, 'glon', 'Galactic longitude')
        self.galactic_longitude.unit = u.deg
        
        self.ecliptic_latitude = _load_array_from_CompositeTable(CompositeTableRow, 'elat', 'Ecliptic latitude')
        self.ecliptic_latitude.unit = u.deg
        self.ecliptic_longitude = _load_array_from_CompositeTable(CompositeTableRow, 'elon', 'Ecliptic longitude')
        self.ecliptic_longitude.unit = u.deg


#%% Ephemeris parameters class
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class _EphemerisParameters():
    """
    Ephemeris parameters class.
    
    Attributes
    ----------
    transit_center : NDDataArray | None
        Transit center of the planetary transit, by default None.
    period : NDDataArray | None
        Orbital period of the planet, by default None.
    transit_length_partial : NDDataArray | None
        Transit length of the planetary transit (including partial transit, defined by T1 and T4 contact points), by default None.
    transit_length_full : NDDataArray | None
        Transit length of the planetary transit (excluding partial transit, defined by T2 and T3 contact points), by default None.
    transit_depth : NDDataArray | None
        Transit depth of the planetary transit, by default None.
    occultation_depth : NDDataArray | None
        Occultation depth of the planetary eclipse, by default None.
    ttv_flag : NDDataArray | None
        TTV flag for the planet, by default False.
    """
    
    transit_center: NDDataArray | None = None
    period: NDDataArray | None = None
    transit_length_partial: NDDataArray | None = None
    transit_length_full: NDDataArray | None = None
    transit_depth: NDDataArray | None = None
    occultation_depth: NDDataArray | None = None
    ttv_flag: bool = False
    
    def _try_load_single_value_fromFullTable(self,
                                             FullTableRow: pd.DataFrame,
                                             CompositeTableRow: pd.DataFrame,
                                             keyword: str,
                                             parameter_name: str) -> NDDataArray:
        """
        Try to load value from Full NASA table, and no value is found loads the composite value.
        
        Parameters
        ----------
        FullTableRow : pd.DataFrame
            Row with the particular reference of interest.
        CompositeTableRow : pd.DataFrame
            CompositeTableRow for given planet.
        keyword : str
            Which keyword to load.
        parameter_name : str
            What parameter is being loaded.

        Returns
        -------
        parameter : NDDataArray
            Loaded parameter outputed as NDDataArray
        """
        try:
            logger.debug('='*25)
            logger.debug('Try:')
            logger.debug(FullTableRow[keyword])
            logger.debug(_extract_reference(FullTableRow['pl_refname']))
            parameter = _load_array_from_FullTable(FullTableRow= FullTableRow,
                                                   keyword= keyword,
                                                   parameter_name= parameter_name
                                                   )
            if np.isnan(parameter):
                logger.debug('NaN detected, loading from CompositeTable')
                raise # Go to except statement to load Composite value
            logger.debug('Successful load')
        except:
            logger.debug('='*25)
            logger.debug('Exception:')
            logger.debug(CompositeTableRow[keyword])
            if keyword != 'ttv_flag':
                logger.debug(_extract_reference(CompositeTableRow[keyword + '_reflink']))
            parameter = _load_array_from_CompositeTable(CompositeTableRow= CompositeTableRow,
                                                        keyword= keyword,
                                                        parameter_name=parameter_name)
            logger.debug('Failed to load Full table')
        return parameter
    
    def _load_values_from_FullTable(self,
                                    FullTablePlanetRow: pd.DataFrame,
                                    CompositeTableRow: pd.DataFrame):
        """
        Loads values from FullTable or Composite if no value is detected.

        Parameters
        ----------
        FullTablePlanetRow : pd.DataFrame
            Full NASA table for given reference for given planet.
        CompositeTableRow : pd.DataFrame
            Composite NASA table for given planet.
        """
        self.reference = _extract_reference(FullTablePlanetRow['pl_refname'])
        
        self.transit_center = self._try_load_single_value_fromFullTable(
            FullTableRow = FullTablePlanetRow,
            CompositeTableRow= CompositeTableRow,
            keyword = 'pl_tranmid',
            parameter_name= 'Transit center'
            )
        self.transit_center.unit = u.day
        self.period = self._try_load_single_value_fromFullTable(
            FullTableRow = FullTablePlanetRow,
            CompositeTableRow= CompositeTableRow,
            keyword = 'pl_orbper',
            parameter_name= 'Planetary period'
            )
        self.period.unit = u.day
        self.transit_length_partial = self._try_load_single_value_fromFullTable(
            FullTableRow = FullTablePlanetRow,
            CompositeTableRow= CompositeTableRow,
            keyword = 'pl_trandur',
            parameter_name= 'Transit length (partial, T14)'
            )
        self.transit_length_partial.unit = u.hour
        self.transit_depth = self._try_load_single_value_fromFullTable(
            FullTableRow = FullTablePlanetRow,
            CompositeTableRow= CompositeTableRow,
            keyword = 'pl_trandep',
            parameter_name= 'Transit depth'
            )
        self.transit_depth.unit = u.Unit(0.01) # Percentage
        self.transit_depth = self.transit_depth.convert_unit_to(u.dimensionless_unscaled)
        self.occultation_depth = self._try_load_single_value_fromFullTable(
            FullTableRow = FullTablePlanetRow,
            CompositeTableRow= CompositeTableRow,
            keyword = 'pl_occdep',
            parameter_name= 'Occultation depth'
            )
        self.occultation_depth.unit = u.Unit(0.01) # Percentage
        self.occultation_depth = self.occultation_depth.convert_unit_to(u.dimensionless_unscaled)
        self.ttv_flag = self._try_load_single_value_fromFullTable(
            FullTableRow = FullTablePlanetRow,
            CompositeTableRow= CompositeTableRow,
            keyword = 'ttv_flag',
            parameter_name= 'TTV flag'
            )
        return
        
        
    def print_values(self):
        """
        Print a default set of parameters using the logging PRINT level (custom defined by rats.utilities)
        """
        logger.print(f'Ephemeris: ' + self.reference)
        
        STANDARD_EPHEMERIS_LIST = [self.transit_center,
                                   self.period,
                                   self.transit_length_partial,
                                   self.transit_depth]
        for array in STANDARD_EPHEMERIS_LIST:
            _print_NDDataArray(array)
        return
    
    
    def create_latex_table(self):
        # TODO
        return

#%% System parameters Composite class
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False)
class SystemParametersComposite(parautils.CalculationTransitLength,
                                parautils.CalculationSystem,
                                parautils.ModellingLightCurve,
                                # parautils.EquivalenciesTransmission,
                                ):
    """
    Ephemeris parameters class. Inherits multiple calculations methods through parautils.CalculationTransitLenght, parautils.CalculationSystem classes.
    
    Attributes
    ----------
    Star : _StellarParameters
        Stellar parameters saved as a _StellarParameters class
    Planet : _PlanetParameters
        Planet parameters saved as a _PlanetParameters class
    Ephemeris : _EphemerisParameters
        Ephemeris parameters saved as a _EphemerisParametes class
    System : _SystemParameters
        System parameters saved as a _SystemParameters class
    filename : str
        Filename where to save the class data (to avoid rerunning the TAP query). By default, this class is saved in "/saved_data/system_parameters.pkl".
    """
    Star: _StellarParameters = _StellarParameters()
    Planet: _PlanetParameters = _PlanetParameters()
    Ephemeris: _EphemerisParameters = _EphemerisParameters()
    System: _SystemParameters = _SystemParameters()
    
    filename: str = 'system_parameters.pkl'
    
    def _save(self):
        """
        Save the class values in given filename.
        """
        with open(self.filename, 'wb') as output_file:
            self.time = datetime.datetime.now()
            pickle.dump(self.__dict__, output_file)
            logger.info('Saved system parameters in file:')
            logger.info(f'    {self.filename}')
    
    def _load(self):
        """
        Load the class values from given filename.
        """
        with open(self.filename, 'rb') as input_file:
            # FIXME
            self.__dict__  =  pickle.load(input_file)
            logger.info('Loaded system parameters from file:')
            logger.info(f'    {self.filename}')
            
    
    def load_NASA_CompositeTable_values(self,
                                        planet_name: str,
                                        force_load: bool = False):
        """
        Loads values from NASA Composite table. For Ephemeris attribute, full table is used.

        Parameters
        ----------
        planet_name : str
            Planet name as defined by NASA archive.
        force_load : bool
            Whether to force reload through the TAP query. By default, is False, which means no reload is forced.
        """
        # Try to load if not forced and filename exist.
        if (not(force_load) and
            os.path.isfile(self.filename)):
            self._load()
            # If last TAP query happened more than week ago, will rerun again.
            if ((datetime.datetime.now() - self.time).days < 7):
                logger.info('SystemParametersComposite class loaded successfully.')
                return
        
        CompositeTableRow = _load_NASA_CompositeTable(system_name = None,
                                                      planet_name = planet_name).iloc[0]
        FullTablePlanet = _load_NASA_FullTable(planet_name = planet_name)
        
        assert CompositeTableRow['pl_name'] == planet_name, 'Composite table does not include the planet name. This should not happen in any scenario.'
        
        self.Star._load_values_from_composite_table(CompositeTableRow= CompositeTableRow)
        self.Planet._load_values_from_composite_table(CompositeTableRow= CompositeTableRow)
        self.System._load_values_from_composite_table(CompositeTableRow= CompositeTableRow)
        self.EphemerisPlanet = EphemerisPlanet(planet_name= planet_name,
                                           FullTablePlanet= FullTablePlanet,
                                           CompositeTableRow= CompositeTableRow)
        
        
        
        logger.info('Calculated most precise ephemeris to current date.')
        self.Ephemeris = self.EphemerisPlanet.find_most_precise()
        logger.info('    In case of needing different set, either manual set:')
        logger.info('        self.Ephemeris = self.EphemerisPlanet[ind]')
        logger.info('            where ind is indice of the ephemeris set you want')
        logger.info('    or:')
        logger.info('        self.Ephemeris = _EphemerisParameters() with exact arguments as extracted')
        logger.info('    or:')
        logger.info('        self.Ephemeris = self.EphemerisPlanet.find_most_precise(night = observed_night)')
        logger.info('            which will take into account most precise ephemeris for given night')
        
        for ind, FullTablePlanetRow in FullTablePlanet.iterrows():
            self.Ephemeris._load_values_from_FullTable(FullTablePlanetRow= FullTablePlanetRow,
                                                       CompositeTableRow= CompositeTableRow)

            self.Ephemeris.print_values()
            
        self._update_contact_points(
            force_recalculate= False
            )
        self._calculate_all_values()
        
        self._save()
        return
    
    def _calculate_all_values(self):
        """
        Calculate all possible values for the system. Launched automatically after loading the NASA archive.
        """
        # FIXME Calculate missing values after loading
        self._calculate_gravity_acceleration()
        self._calculate_atmospheric_scale_height()
        self._update_contact_points()
        self._calculate_contact_points()
        # self._custom_transmission_units()
        
        
        return
    
    def print_main_values(self):
        """
        Print a default set of parameters using the logging PRINT level (custom defined by rats.utilities)
        """
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
    # TODO
    def __init__(self,
                 planet_name: str,
                 FullTablePlanet: pd.DataFrame | None = None,
                 CompositeTableRow: pd.DataFrame | None = None,
                 ):
        self.Ephemeris_list = []
        self.planet_name = planet_name
        
        if FullTablePlanet is None:
            FullTablePlanet = _load_NASA_FullTable(planet_name = planet_name)
        if CompositeTableRow is None:
            CompositeTableRow = _load_NASA_CompositeTable(system_name = None,
                                                          planet_name = planet_name).iloc[0]
        
        for ind, FullTablePlanetRow in FullTablePlanet.iterrows():
            Ephemeris = _EphemerisParameters()
            Ephemeris._load_values_from_FullTable(
                    FullTablePlanetRow= FullTablePlanetRow,
                    CompositeTableRow= CompositeTableRow
                    )
            self.Ephemeris_list.append(
                Ephemeris
                )
        return
    
    def print_values(self):
        for Ephemeris in self.Ephemeris_list:
            Ephemeris.print_values()
        return
    
    def find_most_precise(self,
                          night: astime.Time | None = None):
        """
        Find the most precise ephemeris given a date.

        Parameters
        ----------
        night : astime.Time | None, optional
            Time where to look for transit window, by default None. If None, select current time.

        Returns
        -------
        BestEphemeris : _EphemerisParameters
            Best ephemeris solution for given night.
        """
        
        if night is None:
            night = astime.Time.now()
        
        BestEphemeris = None
        
        for Ephemeris in self.Ephemeris_list:
            if BestEphemeris is None:
                lowest_uncertainty = self._calculate_uncertainty(Ephemeris, night)
                BestEphemeris = Ephemeris
            else:
                if lowest_uncertainty > self._calculate_uncertainty(Ephemeris, night):
                    BestEphemeris = Ephemeris
        logger.info(f'Best ephemeris reference found: {BestEphemeris.reference}')
        logger.info(f'    with uncertainty of {lowest_uncertainty.to(u.h)} [h]')
        logger.info(f'    with uncertainty of {lowest_uncertainty.to(u.min)} [min]')
        logger.info(f'    with uncertainty of {lowest_uncertainty.to(u.s)} [s]')
        
        return BestEphemeris

    def _calculate_uncertainty(self,
                               Ephemeris: _EphemerisParameters,
                               night: astime.Time):
        
        n, remainder = divmod((night.jd - Ephemeris.transit_center), Ephemeris.period)
        n+=1
        
        if Ephemeris.period.data < 1.5: # To avoid falling in different transit window
            n+=1
        
        # Calculate T_0 and sigma_T_0 for given transit
        current_transit_center = astime.Time(Ephemeris.transit_center + n * Ephemeris.period, format = 'jd')
        uncertainty_transit_window = np.sqrt(
            Ephemeris.transit_center.uncertainty.array**2 + 
            n**2 * Ephemeris.period.uncertainty.array**2
            ) * u.day
        
        logger.print(f'Transit window on night: {night}' )
        logger.print(f'    Calculated center: {current_transit_center.datetime}')
        logger.print(f'    Uncertainty: {uncertainty_transit_window.to(u.h)} [h]')
        logger.print(f'    Uncertainty: {uncertainty_transit_window.to(u.min)} [min]')
        logger.print(f'    Uncertainty: {uncertainty_transit_window.to(u.s)} [s]')
        
        return uncertainty_transit_window

class CompositeTable():
    # TODO
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
    TOI132 = SystemParametersComposite(
        filename= os.getcwd() + '/saved_data/system_parameters.pkl'
        )
    TOI132.load_NASA_CompositeTable_values(planet_name = 'TOI-132 b',
                                           force_load=True
                                           )
    TOI132.Planet._calculate_gravity_acceleration()
    TOI132._update_contact_points(
        force_recalculate= True
    )
    TOI132.print_main_values()
    TOI132.Planet._calculate_atmospheric_scale_height()

    TOI132._phase_fold(bjd= 2600000)
    
    # CompositeTableAll = _load_NASA_CompositeTable(system_name = None)
    # FullTableTOI132 = _load_NASA_FullTable(planet_name = planet_name)
    # FullTableAll = _Load_NASA_Full_Table(planet_name = None)
    logger.info('Test succesful. All Tables were succesfully loaded:')
    
# %%
