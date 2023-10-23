# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 08:34:27 2021

@author: Chamaeleontis

Functions for parametrizing the observations
Includes information on star, planet (masses, radius, velocities etc.)
Includes information on measurement

Main part of this sub-library is system_parameters_class
    This class has several parameters, divided based on whether they are star, planet, system or transit related
    Main usage is to have an object with all the parameters necessary for various calculations, like the planetary velocity
    

"""

#%% Importing libraries
import numpy as np
import os
import astropy.units as u
import pyvo as vo
import batman
from astropy.nddata import NDData,StdDevUncertainty, NDDataArray
import rats.spectra_manipulation as sm
import termcolor as tc
import pandas as pd
import pickle
import re
import specutils as sp
import astropy.io.fits as fits
import datetime
import astropy.constants as con
from joblib import Parallel, delayed
import multiprocessing
from rats.utilities import time_function, save_and_load, progress_tracker, disable_func,logger,skip_function, todo_function
from typing import Callable, Iterator, Union, Optional, Tuple, Any
#%% Type aliases
# Spectrum1D object as defined by specutils
Spectrum1D = sp.spectra.spectrum1d.Spectrum1D
# SpectrumCollection object as defined by specutils
SpectrumCollection = sp.spectra.spectrum_collection.SpectrumCollection
# SpectrumList object as defined by specutils
SpectrumList = sp.spectra.spectrum_list.SpectrumList
# Array object as defined by numpy
Array = np.ndarray
# Header object as defined by astropy.io.fits
Header = fits.header.Header
# System parameter object - Will not throw errors as each function expect different set of parameters only
SystemParameters = Any
#%% find_nearest
@disable_func
def find_nearest(array, value):
    '''
    # TODO: Check if this function is used or not
    Find nearest value index in array
    '''
    array = np.asarray(array,dtype=np.dtype(float))
    idx = (np.abs(array - value)).argmin()
    return idx
#%%
# =============================================================================
# TODO: Redo system-parameter class!
# =============================================================================

#%% system_parameters_class
class attrib:pass
# @todo_function
class system_parameters_class:
    '''
    Class defining system parameters
    
    Parameters divided into several subclasses:
        star ; parameters related to star
        planet ; parameters related to planet
        system ; paramaeters related to system
        transit ; parameters related to transit
    
    List of parameters to initilize
    name - mandatory parameter
                  stellar_radius
                  stellar_mass
                  planet_radius
                  planet_mass
                  semimajor_axis
                  impact_parameter
                  inclination
                  eccentricity
                  period
                  transit_depth
                  argument_of_periastron
                  transit_center
    
    Functions:
        load_nasa_parameters(pl_name=False)
            Loads system parameters from NASA exoplanet archive
            
    
    Note:
        class attrib is here to allow for subattributes notation, eg: parameters.star.radius
        
    Use: 
        parameter_class = system_parameters_class('KELT-10 b', use_NASA_value= True, period = 4)
            This will create a class with parameters for KELT-10 b, using NASA exoplanet archive, and rewriting period value to 4
    '''
    def save(self):
        """save class as self.name.txt"""
        with open('sys_para.pkl', 'wb') as output_file:
            pickle.dump(self.__dict__, output_file)


    def load(self):
        """try load self.name.txt"""
        with open('sys_para.pkl', 'rb') as input_file:
            self.__dict__  =  pickle.load(input_file)
    
    
    # @time_function
    # @progress_tracker
    def load_nasa_parameters(self,pl_name = False,force_load=False):
        '''
        Loading system parameters from NASA exoplanet archive
        
        Input:
            pl_name = False ; name of the planet as given by the pl_name flag in NASA exoplanet archive
        Output:
            none
        Change:
            Updated parameters of the system
            TODO: Setup so it goes through all available data, picks the most precise, adds the reference and error as well
        '''
        try: # Test loading the function
            logger.info('Trying to load NASA table')
            self.load()
            if ((datetime.datetime.now() - self.time ).days >7) or (force_load == True): # If not < week data, reload anyway
                logger.info('Reloading NASA table [new week] or forced loading')
                service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP/") # Initialization of TAP service
                pl_name = "'%s'"%pl_name # To have it in correct format for the search
                # Searching relevant parameters
                self.nasa_table = pd.DataFrame(service.search("SELECT * FROM pscomppars WHERE pl_name = %s"% pl_name))
                self.time = datetime.datetime.now()
                self.save()
        except: # Loading failed, reload the archive - should happen only the first time running the code
            logger.info('Weird, no table found. Should happen on first time running the function only.')
            service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP/") # Initialization of TAP service
            pl_name = "'%s'"%pl_name # To have it in correct format for the search
            # Searching relevant parameters
            self.nasa_table = pd.DataFrame(service.search("SELECT * FROM pscomppars WHERE pl_name = %s"% pl_name))
            self.time = datetime.datetime.now()
            self.save()
            pass
        
        pl_name = "'%s'"%pl_name # To have it in correct format for the search
        service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP/") # Initialization of TAP service
        
        # Local variable assigning (unnecessary, but slightly cleaner)
        nasa_table = self.nasa_table
        
        # TODO:
            # list_reference = [ self.planet.semiamp_sref]
        
        # Defining stellar parameters
        # Basic parameters
        self.star.type = nasa_table['st_spectype'][0]
        self.star.typeref = nasa_table['st_spectype_reflink'][0]
        self.star.teff = nasa_table['st_teff'][0] * u.K
        self.star.teffref = nasa_table['st_teff_reflink'][0]
        self.star.radius = nasa_table['st_rad'][0] * u.R_sun
        self.star.radiusherr = nasa_table['st_raderr1'][0] * u.R_sun
        self.star.radiuslerr = nasa_table['st_raderr2'][0] * u.R_sun
        self.star.radiusref = nasa_table['st_rad_reflink'][0]
        self.star.mass = nasa_table['st_mass'][0] * u.M_sun
        self.star.massherr = nasa_table['st_masserr1'][0] * u.M_sun
        self.star.masslerr = nasa_table['st_masserr2'][0] * u.M_sun
        self.star.massref = nasa_table['st_mass_reflink'][0]
        self.star.luminosity = nasa_table['st_lum'][0] * u.L_sun
        self.star.luminosityref = nasa_table['st_lum_reflink'][0]
        self.star.metallicity = nasa_table['st_met'][0] 
        self.star.metallicityref = nasa_table['st_met_reflink'][0] 
        self.star.metallicity_ratio = nasa_table['st_metratio'][0] 
        self.star.logg = nasa_table['st_logg'][0]
        self.star.loggref = nasa_table['st_logg_reflink'][0]
        self.star.age = nasa_table['st_age'][0] * u.Gyr
        self.star.ageref = nasa_table['st_age_reflink'][0]
        self.star.density = nasa_table['st_dens'][0] * u.g / u.cm/u.cm/u.cm
        self.star.densityref = nasa_table['st_dens_reflink'][0]
        self.star.vsini = nasa_table['st_vsin'][0] * u.km/u.s
        self.star.vsiniref = nasa_table['st_vsin_reflink'][0]
        self.star.rotp = nasa_table['st_rotp'][0] *u.day
        self.star.rotpref = nasa_table['st_rotp_reflink'][0]
        
        # Names and IDs of star
        self.star.name = nasa_table['hostname'][0]
        self.star.hdcatalogue = nasa_table['hd_name'][0]
        self.star.hipcatalogue = nasa_table['hip_name'][0]
        self.star.ticid = nasa_table['tic_id'][0]
        self.star.gaiaid = nasa_table['gaia_id'][0]
        
        # Photomety
        self.star.bmag = nasa_table['sy_bmag'][0] 
        self.star.bmagref = nasa_table['sy_bmag_reflink'][0] 
        self.star.vmag = nasa_table['sy_vmag'][0] 
        self.star.vmagref = nasa_table['sy_vmag_reflink'][0] 
        self.star.jmag = nasa_table['sy_jmag'][0] 
        self.star.jmagref = nasa_table['sy_jmag_reflink'][0] 
        self.star.hmag = nasa_table['sy_hmag'][0] 
        self.star.hmagref = nasa_table['sy_hmag_reflink'][0] 
        self.star.kmag = nasa_table['sy_kmag'][0] 
        self.star.kmagref = nasa_table['sy_kmag_reflink'][0] 
        self.star.umag = nasa_table['sy_umag'][0] 
        self.star.umagref = nasa_table['sy_umag_reflink'][0] 
        self.star.gmag = nasa_table['sy_gmag'][0] 
        self.star.gmagref = nasa_table['sy_gmag_reflink'][0] 
        self.star.rmag = nasa_table['sy_rmag'][0] 
        self.star.rmagref = nasa_table['sy_rmag_reflink'][0] 
        self.star.imag = nasa_table['sy_imag'][0] 
        self.star.imagref = nasa_table['sy_imag_reflink'][0] 
        self.star.zmag = nasa_table['sy_zmag'][0] 
        self.star.zmagref = nasa_table['sy_zmag_reflink'][0] 
        self.star.w1mag = nasa_table['sy_w1mag'][0] 
        self.star.w1magref = nasa_table['sy_w1mag_reflink'][0] 
        self.star.w2mag = nasa_table['sy_w2mag'][0] 
        self.star.w2magref = nasa_table['sy_w2mag_reflink'][0] 
        self.star.w3mag = nasa_table['sy_w3mag'][0] 
        self.star.w3magref = nasa_table['sy_w3mag_reflink'][0] 
        self.star.w4mag = nasa_table['sy_w4mag'][0] 
        self.star.w4magref = nasa_table['sy_w4mag_reflink'][0] 
        self.star.gaiamag = nasa_table['sy_gaiamag'][0] 
        self.star.gaiamagref = nasa_table['sy_gaiamag_reflink'][0] 
        self.star.icmag = nasa_table['sy_icmag'][0] 
        self.star.icmagref = nasa_table['sy_icmag_reflink'][0] 
        self.star.tmag = nasa_table['sy_tmag'][0] 
        self.star.tmagref = nasa_table['sy_tmag_reflink'][0] 
        self.star.kepmag = nasa_table['sy_kepmag'][0] 
        self.star.kepmagref = nasa_table['sy_kepmag_reflink'][0] 
        
        # Defining planetary parameters
        # self.planet.radius = nasa_table['pl_radj'][0] * u.R_jup
        # self.planet.mass = nasa_table['pl_bmassj'][0] * u.M_jup
        self.planet.name = nasa_table['pl_name'][0]
        self.planet.letter = nasa_table['pl_letter'][0]
        
        # Planetary mass and radius
        try:
            self.planet.radius = nasa_table['pl_radj'][0] * u.R_jup
            self.planet.mass = nasa_table['pl_bmassj'][0] * u.M_jup
            self.planet.radiusherr = nasa_table['pl_radjerr1'][0] * u.R_jup
            self.planet.massherr = nasa_table['pl_bmassjerr1'][0] * u.M_jup
            self.planet.radiuslerr = nasa_table['pl_radjerr2'][0] * u.R_jup
            self.planet.masslerr = nasa_table['pl_bmassjerr2'][0] * u.M_jup
            self.planet.radiusref = nasa_table['pl_radj_reflink'][0]
            self.planet.massref = nasa_table['pl_bmassj_reflink'][0]
            self.planet.mass_provenance = nasa_table['pl_bmassprov'][0]
        except:
            try:
                self.planet.radius = nasa_table['pl_rade'][0] * u.R_earth
                self.planet.mass = nasa_table['pl_bmasse'][0] * u.M_earth
                self.planet.radiusref = nasa_table['pl_rade_reflink'][0]
                self.planet.massref = nasa_table['pl_bmasse_reflink'][0]
                self.planet.mass_provenance = nasa_table['pl_bmassprov'][0]
            except:
                pass
        
        self.planet.density = nasa_table['pl_dens'][0] * u.g / u.cm/u.cm/u.cm
        self.planet.densityref = nasa_table['pl_dens_reflink'][0]
        
        self.planet.a = nasa_table['pl_orbsmax'][0] * u.au
        self.planet.aherr = nasa_table['pl_orbsmaxerr1'][0] * u.au
        self.planet.alerr = nasa_table['pl_orbsmaxerr2'][0] * u.au
        self.planet.aref = nasa_table['pl_orbsmax_reflink'][0]
        self.planet.b = nasa_table['pl_imppar'][0] 
        self.planet.bherr = nasa_table['pl_impparerr1'][0] 
        self.planet.blerr = nasa_table['pl_impparerr2'][0] 
        self.planet.bref = nasa_table['pl_imppar_reflink'][0] 
        self.planet.i = nasa_table['pl_orbincl'][0] * u.deg
        self.planet.iherr = nasa_table['pl_orbinclerr1'][0] * u.deg
        self.planet.ilerr = nasa_table['pl_orbinclerr2'][0] * u.deg
        self.planet.iref = nasa_table['pl_orbincl_reflink'][0]
        self.planet.e = nasa_table['pl_orbeccen'][0]
        self.planet.eherr = nasa_table['pl_orbeccenerr1'][0]
        self.planet.elerr = nasa_table['pl_orbeccenerr2'][0]
        self.planet.eref = nasa_table['pl_orbeccen_reflink'][0]
        
        if type(self.planet.e) == np.ma.core.MaskedConstant:
            self.planet.e = 0

        self.planet.omega = nasa_table['pl_orblper'][0] * u.deg
        self.planet.omegaherr = nasa_table['pl_orblpererr1'][0] * u.deg
        self.planet.omegalerr = nasa_table['pl_orblpererr2'][0] * u.deg
        self.planet.omegaref = nasa_table['pl_orblper_reflink'][0]
        if self.planet.e == 0:
            self.planet.omega = 90 * u.deg
            self.planet.omegaherr = 0 * u.deg
            self.planet.omegalerr = 0 * u.deg
            
        self.planet.omega_bar = np.radians(self.planet.omega)
        self.planet.omega_barreef = nasa_table['pl_orbtper_reflink'][0]
        self.planet.P = nasa_table['pl_orbper'][0] * u.day
        self.planet.Pherr = nasa_table['pl_orbpererr1'][0] * u.day
        self.planet.Plerr = nasa_table['pl_orbpererr2'][0] * u.day
        self.planet.Pref = nasa_table['pl_orbper_reflink'][0]
        self.planet.a_rs_ratio = nasa_table['pl_ratdor'][0] 
        self.planet.a_rs_ratioref = nasa_table['pl_ratdor_reflink'][0] 
        self.planet.rs_a_ratio = 1/self.planet.a_rs_ratio
        self.planet.rs_a_ratioref = nasa_table['pl_ratdor_reflink'][0] 
        self.planet.semiamp_s = nasa_table['pl_rvamp'][0] * u.m / u.s
        self.planet.semiamp_sherr = nasa_table['pl_rvamperr1'][0] * u.m / u.s
        self.planet.semiamp_slerr = nasa_table['pl_rvamperr2'][0] * u.m / u.s
        self.planet.semiamp_sref = nasa_table['pl_rvamp_reflink'][0]
        
        
        self.planet.insolation = nasa_table['pl_insol'][0] 
        self.planet.insolationref = nasa_table['pl_insol_reflink'][0]
        self.planet.equilibrium_temperature = nasa_table['pl_eqt'][0]  * u.K
        self.planet.equilibrium_temperatureref = nasa_table['pl_eqt_reflink'][0]
        self.planet.rprs = nasa_table['pl_ratror'][0] * u.dimensionless_unscaled
        self.planet.rprsref = nasa_table['pl_ratror_reflink'][0]
        self.planet.epoch_periastron = nasa_table['pl_orbtper'][0] * u.deg
        self.planet.epoch_periastronref = nasa_table['pl_orbtper_reflink'][0]
        self.planet.obliquity = nasa_table['pl_projobliq'][0] * u.deg
        self.planet.obliquityref = nasa_table['pl_projobliq_reflink'][0]
        self.planet.trueobliquity = nasa_table['pl_trueobliq'][0] * u.deg
        self.planet.trueobliquityref = nasa_table['pl_trueobliq_reflink'][0]
        
        # Planet discovery parameters
        self.planet.discovery.method = nasa_table['discoverymethod'][0]
        self.planet.discovery.year = nasa_table['disc_year'][0]
        self.planet.discovery.reference = nasa_table['disc_refname'][0]
        self.planet.discovery.publicationdate = nasa_table['disc_pubdate'][0]
        self.planet.discovery.locale = nasa_table['disc_locale'][0]
        self.planet.discovery.facility = nasa_table['disc_facility'][0]
        self.planet.discovery.instrument = nasa_table['disc_instrument'][0]
        
        # Planet detections flags
        self.planet.detection.rv = nasa_table['rv_flag'][0]
        self.planet.detection.pulsar_timing = nasa_table['pul_flag'][0]
        self.planet.detection.pulsation_timing = nasa_table['ptv_flag'][0]
        self.planet.detection.transit = nasa_table['tran_flag'][0]
        self.planet.detection.astrometric_variations = nasa_table['ast_flag'][0]
        self.planet.detection.orbital_brightness_modulation = nasa_table['obm_flag'][0]
        self.planet.detection.microlensing = nasa_table['micro_flag'][0]
        self.planet.detection.eclipse_timing = nasa_table['etv_flag'][0]
        self.planet.detection.imaging = nasa_table['ima_flag'][0]
        self.planet.detection.disk_kinematics = nasa_table['dkin_flag'][0]
        

        # System parameters
        self.system.vel_s = nasa_table['st_radv'][0] * u.km / u.s
        self.system.vel_sherr = nasa_table['st_radverr1'][0] * u.km / u.s
        self.system.vel_slerr = nasa_table['st_radverr2'][0] * u.km / u.s
        self.system.vel_sref = nasa_table['st_radv_reflink'][0]
        self.system.distance = nasa_table['sy_dist'][0] * u.pc
        self.system.distanceref = nasa_table['sy_dist_reflink'][0]
        self.system.number_of_stars = nasa_table['sy_snum'][0]
        self.system.number_of_planets = nasa_table['sy_pnum'][0]
        self.system.number_of_moons = nasa_table['sy_mnum'][0]
        
        self.system.totalpropermotion = nasa_table['sy_pm'][0]
        self.system.totalpropermotionref = nasa_table['sy_pm_reflink'][0]
        self.system.propermotionra = nasa_table['sy_pmra'][0]
        self.system.propermotiondec = nasa_table['sy_pmdec'][0]
        self.system.parallax = nasa_table['sy_plx'][0]
        self.system.parallaxref = nasa_table['sy_plx_reflink'][0]
        
        # Position of the system
        self.system.position.rastr = nasa_table['rastr'][0]
        self.system.position.decstr = nasa_table['decstr'][0]
        self.system.position.ra = nasa_table['ra'][0]
        self.system.position.dec = nasa_table['dec'][0]
        self.system.position.galactic_latitude = nasa_table['glat'][0] * u.deg 
        self.system.position.galactic_longitude = nasa_table['glon'][0] * u.deg
        self.system.position.ecliptic_latitude = nasa_table['elat'][0] * u.deg
        self.system.position.ecliptic_longitude = nasa_table['elon'][0] * u.deg
        
        # Defining transit parameters
        self.transit.T_C = nasa_table['pl_tranmid'][0] * u.day
        self.transit.T_Cherr = nasa_table['pl_tranmiderr1'][0] * u.day
        self.transit.T_Clerr = nasa_table['pl_tranmiderr2'][0] * u.day
        self.transit.T_Cref = nasa_table['pl_tranmid_reflink'][0]
        self.transit.delta = nasa_table['pl_trandep'][0] / 100 *u.dimensionless_unscaled
        self.transit.deltaherr = nasa_table['pl_trandeperr1'][0] / 100 *u.dimensionless_unscaled
        self.transit.deltalerr = nasa_table['pl_trandeperr2'][0] / 100 *u.dimensionless_unscaled
        self.transit.deltaref = nasa_table['pl_trandep_reflink'][0]
        self.transit.ttv_flag = nasa_table['ttv_flag'][0]
        self.transit.occulation_depth = nasa_table['pl_occdep'][0] / 100 *u.dimensionless_unscaled
        self.transit.occulation_depthref = nasa_table['pl_occdep_reflink'][0]
        self.transit.T14 = nasa_table['pl_trandur'][0] * u.hour
        self.transit.T14herr = nasa_table['pl_trandurerr1'][0] * u.hour
        self.transit.T14lerr = nasa_table['pl_trandurerr2'][0] * u.hour
        self.transit.T14ref = nasa_table['pl_trandur_reflink'][0]
        
        self.calculate_scale_height()
        self.calculate_ksi()
        
        self.save()
        
        return 
    
    
    # @todo_function
    def __init__(self,
                      name,
                      use_NASA_value = False,
                      force_values = True
                    ):
        '''
        Initilization of arguments
        Apart of name, parameters by default are set to None
            name can be set as anything, however in case of using NASA exoplanet database parameters, it is necessary to be given name of the   planet, as given by the 'pl_name' parameter
            This can be circumvent by using the pl_name parameter
            This is in order to evade multiplicity problems
        Will throw warnings/errors in some other functions if parameters are not properly initilized
        All parameters need to have units included using the astropy.units module
        
        For calculation of transit_length 1-4 and 2-3 formula from Winn (2014) has been used
        '''
        
        try:
            self.load()
        except:
            pass
        if force_values == True:
            # Defining name of the system 
            self.name = name
            self.star = attrib()
            self.planet = attrib()
            self.system = attrib()
            self.system.position = attrib()
            self.star.photometry = attrib()
            self.transit = attrib()
            self.planet.discovery = attrib()
            self.planet.detection = attrib()
            
            self.save()
        return
    
    def create_system_table(self):
        '''
        Prints a latex code to paste into paper for system parameters, uncertainties and refernces.
        It is necessary to add caption with corresponding references (numbering is dealt with). 
        
        TODO:
            Restructure the table
            Better references handling
            Truncate values to specific number of digits
            Delta in percentages

        Returns
        -------
        Prints a code to copy paste into latex file.

        '''
        
        
        import tabulate as ta
        
        list_reference = [self.star.typeref, self.star.massref, self.star.radiusref, self.planet.massref, self.planet.radiusref, self.planet.aref, self.planet.iref, self.planet.eref, self.planet.omegaref, self.system.vel_sref, self.transit.T_Cref, self.transit.T14ref, self.planet.Pref, self.planet.semiamp_sref,self.transit.deltaref]
        
        references, indices = np.unique(list_reference,return_inverse=True)
        
        # Numbering of references feels dumb, could be improved
        table = [
            ['Parameter','Value','Reference'],
            ['System Name',self.name,'--'],
            ['Spectral type',self.star.type,indices[0]+1],
            ['$M_\mathrm{star} [M_{\odot}]$',str(self.star.mass.value)+'$^{+%f}'%self.star.massherr.value + '_{%f}$'%self.star.masslerr.value,indices[1]+1],
            ['$R_\mathrm{star} [R_{\odot}]$',str(self.star.radius.value)+'$^{+%f}'%self.star.radiusherr.value + '_{%f}$'%self.star.radiuslerr.value,indices[2]+1],
            ['Planet name',self.planet.name,'--'],
            ['$M_{p} [M_\mathrm{jup}]$',str(self.planet.mass.value)+'$^{+%f}'%self.planet.massherr.value + '_{%f}$'%self.planet.masslerr.value,indices[3]+1],
            ['$R_{p} [R_\mathrm{jup}]$',str(self.planet.radius.value)+'$^{+%f}'%self.planet.radiusherr.value + '_{%f}$'%self.planet.radiuslerr.value,indices[4]+1],
            ['a [au]',str(self.planet.a.value)+'$^{+%f}'%self.planet.aherr.value + '_{%f}$'%self.planet.alerr.value,indices[5]+1],
            ['i [deg]',str(self.planet.i.value)+'$^{+%f}'%self.planet.iherr.value + '_{%f}$'%self.planet.ilerr.value,indices[6]+1],
            ['e [1]',str(self.planet.e)+'$^{+%f}'%self.planet.eherr + '_{%f}$'%self.planet.elerr,indices[7]+1],
            ['$\omega$ [deg]',str(self.planet.omega.value)+'$^{+%f}'%self.planet.omegaherr.value + '_{%f}$'%self.planet.omegalerr.value,indices[8]+1],
            ['$\gamma$ [km/s]',str(self.system.vel_s.value)+'$^{+%f}'%self.system.vel_sherr.value + '_{%f}$'%self.system.vel_slerr.value,indices[9]+1],
            ['$\delta [1]$',str(self.transit.delta.value)+'$^{+%f}'%float(self.transit.deltaherr.value) + '_{%f}$'%float(self.transit.deltalerr.value) , indices[14]+1],
            
            ['$T_{C} [BJD_\mathrm{TBD}]$',str(self.transit.T_C.value)+'$^{+%f}'%self.transit.T_Cherr.value + '_{%f}$'%self.transit.T_Clerr.value,indices[10]+1],
            ['$T_{14} [h]$',str(self.transit.T14.value)+'$^{+%f}'%self.transit.T14herr.value + '_{%f}$'%self.transit.T14lerr.value,indices[11]+1],
            
            # ['$T_{23} [h]$','{:.2f}'.format(self.transit.T23.to(u.h).value)+'$^{+%f}'%self.planet.omegaherr.value + '_{%f}$'%self.planet.omegalerr.value,'Calculated'],
            
            ['P [d]',str(self.planet.P.value)+'$^{+%f}'%self.planet.Pherr.value + '_{%f}$'%self.planet.Plerr.value,indices[12]+1],
            ['K [m/s]',str(self.planet.semiamp_s.value)+'$^{+%f}'%self.planet.semiamp_sherr.value + '_{%f}$'%self.planet.semiamp_slerr.value,indices[13]+1],
            ]
        print('\nTabulate Latex:\n')
        print(ta.tabulate(table, headers='firstrow', tablefmt='latex_raw'))
        return
    
    
    def update_contact_points(self):
        '''
        Contact points based on the Winn 2014 paper
        '''
        try:
            self.transit.T14
        except:
            self.transit.T14 = self.planet.P / (np.pi * u.rad) * \
                              np.arcsin((self.star.radius / self.planet.a *\
                              np.sqrt((1+self.transit.delta)**2- self.planet.b**2)/\
                              np.sin(self.planet.i)).decompose()) *\
                              np.sqrt(1-self.planet.e**2)/(1+self.planet.e*np.sin(self.planet.omega))
        try:
            self.transit.T23
        except:
            self.transit.T23 = self.planet.P / (np.pi * u.rad)  * \
                              np.arcsin((self.star.radius / self.planet.a *\
                              np.sqrt( (1 - self.transit.delta)**2 - self.planet.b**2) /\
                              np.sin(self.planet.i)).decompose())*\
                              np.sqrt(1-self.planet.e**2)/(1+self.planet.e*np.sin(self.planet.omega))

    def define_light_curve_model(self,coefficients,mode='quadratic'):
        '''
        Define parameters and model for light curve given limb-darkening coefficients
        Input:
            coefficients ; [list] of coefficients
            mode ; limb darkening model
        Output:
            Updating class parameters:
                self.transit_model ; light curve model
                self.transit_parameters ; light curve parameters
        '''
        
        params = batman.TransitParams()       #object to store transit parameters
        params.t0 = self.transit.T_C.value                        #time of inferior conjunction
        params.per = self.planet.P.value                       #orbital period
        params.rp = (self.planet.radius.to(u.R_sun) / self.star.radius).value                       #planet radius (in units of stellar radii)
        params.a = (self.planet.a.to(u.R_sun)/self.star.radius).value                        #semi-major axis (in units of stellar radii)
        params.inc = self.planet.i.value                      #orbital inclination (in degrees)
        params.ecc = self.planet.e                       #eccentricity
        params.w = self.planet.omega.value                        #longitude of periastron (in degrees)
        params.limb_dark = mode    #limb darkening model
        params.u = coefficients      #limb darkening coefficients [u1, u2, u3, u4]
        
        t = np.array([self.transit.T_C.value])
        self.transit.transit_model = batman.TransitModel(params, t)    #initializes model
        self.transit.transit_parameters = params
        return
    
    def get_lc_flux(self,t):
        '''
        Gives a flux value given a time 
        Input:
            t ; time, in BJD - 2 400 000
        Output:
            flux ; flux of the model at given time
        '''
        t = np.array([t.value])
        transit_model =  batman.TransitModel(self.transit.transit_parameters, t)
        flux = transit_model.light_curve(self.transit.transit_parameters)[0]
        return flux
    
    def update_spec_flux(self,spec):
        '''
        Updates meta value related to light curve
        Input:
            spec ; sp.Spectrum1D with meta informationscatter
        Output:
            none
        Update:
            spec.meta['light_curve_flux'] ; light curve at given BJD time
            spec.meta['delta'] ; 1 - light curve
        
        '''
        time = spec.meta['BJD']# - 2400000 *u.day
        spec.meta['light_curve_flux'] = self.get_lc_flux(time) * u.dimensionless_unscaled
        spec.meta['delta'] = (1 - self.get_lc_flux(time)) * u.dimensionless_unscaled
        return
    
    def update_spec_list(self,spec_list):
        '''
        Updates light_curve for entire spec_list
        Input:
            spec_list ; sp.SpectrumList
        Update:
            updated meta in spec_list
        '''
        
        for item in spec_list:
            self.update_spec_flux(item)
        return
    
    def find_phase(self,spec_list,night,line):
        '''
        Updates meta information regarding RM_correction
        Input:
            spec_list ; sp.SpectrumList
            night ; string for given Night 
            line ; 
        '''
        phase = line[0]
        vel = line[1]
        # err_vel = line[2]
        delta = 99999
        for ind,item in enumerate(spec_list):
            if (item.meta['Night'] == night):
                if abs(item.meta['Phase'] - phase) < delta:
                    delta = abs(item.meta['Phase']- phase)
                else:
                    spec_list[ind-1].meta['vel_stel_loc'] = (vel * u.km / u.s).to(u.m/u.s)
                    spec_list[ind-1].meta['RM_velocity'] = True
                    return
        return
    
    def load_stel_vel(self,file):
        '''
        Loads file of local stellar velocities
        Input:
            file ; string - location of the rv file
        Output:
            night_rv ; list - list with array of local stellar velocities for each night
        '''
        with open(file) as f:
            lines= f.readlines()
        
        night_rv = []
        ind_mark = []
        night = []
        for ind,line in enumerate(lines):
            if 'Night' in line:
                ind_mark.append(ind)
                night.append(line[7:17])
        
        for ii,mark in enumerate(ind_mark):
            if (ii+1) != len(ind_mark):
                night_rv.append([night[ii],np.genfromtxt(file,skip_header=mark+2,skip_footer=(len(lines)-ind_mark[ii+1]))])
            if (ii+1) == len(ind_mark):
                night_rv.append([night[ii],np.genfromtxt(file,skip_header=mark+2)])
        return night_rv
    
    def get_loc_stel_vel_test(self,file,spec_list):
        '''
        Loads local stellar velocities and updates the meta values in spec_list
        Input:
            file ; string - location of the rv file
            spec_list ; sp.SpectrumList - spectrum list for which to update meta
        Change:
            spec_list.meta - update vel_stel_loc keyword where applyable
        
        '''
        night_rv = self.load_stel_vel(file)
        phases_field=[]
        for ind,item in enumerate(spec_list):
            
            phases_field.append([ind,item.meta['Night'],item.meta['Phase']])

        for night,array in night_rv:
            filter_arr = []
            for element in phases_field:
                if element[1]== night:
                    filter_arr.append(True)
                else:
                    filter_arr.append(False)
            phases = np.asarray(phases_field)[filter_arr]
            idx = find_nearest(phases[:,2], array.transpose()[0][0])
            for ni,item in enumerate(spec_list):
                if night == item.meta['Night']:
                    st_night = ni
                    break

            for ii,element in enumerate(array):
                spec_list[idx+ii+st_night].meta['vel_stel_loc'] = element[1] * u.km/u.s
                spec_list[idx+ii+st_night].meta['RM_velocity'] = True
            
        return
    
    def load_CONAN_output(self,filename):
        '''
        Loads system parameters from the CONAN output file
        Input:
            filename ; string - path to filename
        Output:
            None
        
        '''
        text_file = open(filename, "r")
        lines = text_file.read().split('\n')
        for line in lines:
            line = re.split("\s+",line)
            if 'T_0' in line: # Mid-transit
                self.transit.T_C = float(line[1]) * u.day
            elif 'Period_[d]' in line: # Period
                self.planet.P = float(line[1]) * u.day
            elif 'Rp_[Rjup]' in line: # Planetary radius
                self.planet.radius = float(line[1]) * u.R_jup
            elif 'a_[au]' in line: # Semi-major axis
                self.planet.a = float(line[1]) * u.au
            elif 'inclination_[deg]' in line: # Inclination in degrees
                self.planet.i = float(line[1]) * u.deg
            elif 'Rs_[Rsun]' in line: # Radius of star 
                self.star.radius = float(line[1]) * u.R_sun
            elif 'Transit_dur' in line: # Transit duration T14
                self.transit.T14 = float(line[1]) * u.day
        return
    
    def read_line(self,line):
        array = np.fromstring(line,sep=' ')
        return array
    
    def get_loc_stel_vel(self,file,spec_list):
        with open(file) as f:
            lines = f.readlines()
        
        night_ind = []
        for ind,line in enumerate(lines):
            if 'Night' in line:
                cur_night = line[7:17]
                night_ind.append([cur_night,ind])
        
        num_nights = len(night_ind)
        indices = np.zeros((num_nights,2),dtype=int)
        
        for ind,night in enumerate(night_ind):
            indices[ind,0] = night[1] + 3
            if ind != num_nights-1:
                indices[ind,1] = night_ind[ind+1][1] - 3
            else:
                indices[ind,1] = len(lines)
        for ni,array in enumerate(indices):
            for ii in range(array[0],array[1]):
                array = self.read_line(lines[ii])
                self.find_phase(spec_list,night_ind[ni][0],array)
        return
    
    def calculate_ksi(self):
        '''
        \ksi = \frac{T_{eq}}{1000K}\frac{g}{g_{J}}
        
        g =GM/r
        can be simplified 
        TODO:
            simplify
            
        '''
        g_J = con.G * (con.M_jup).to(u.kg)/((con.R_jup).to(u.m)**2)
        
        try:
            g = con.G * (self.planet.mass).to(u.kg)/((self.planet.radius).to(u.m)**2)
            self.planet.ksi= (self.planet.equilibrium_temperature/1000/u.K) * (g/g_J)
        except:
            pass
                
        return
    
    def calculate_scale_height(self,mol_mass:float = 2.3):
        '''
        Calculates scale height of planets assuming mol_mass. H = \frac{k_b T}{g\mu}

        Parameters
        ----------
        mol_mass : float, optional
            Molecular mass in units of u.kg/u.mol. The default is 2.3 (H+He dominated).

        Returns
        -------
        None. Adds a row with ['H'] in the table in meters

        '''
        
        try:
            g = con.G * (self.planet.mass).to(u.kg)/((self.planet.radius).to(u.m)**2)
            self.planet.H = (con.N_A *self.planet.equilibrium_temperature  * con.k_B / (mol_mass*u.kg/u.mol) / g).decompose().value
        except:
            pass
        return
    
    
    def print_values(self):
        '''
        Prints basic system parameters used

        Returns
        -------
        None.

        '''
        tmp_message = 'Name of the system:',self.name
        tc.cprint(tmp_message, 'grey','on_green')
        tmp_message = 'Stellar parameters:'
        tc.cprint(tmp_message, 'red','on_yellow',attrs=['bold'])
        tmp_message = 'Stellar radius:',self.star.radius
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Stellar mass:', self.star.mass
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Planetary parameters:'
        tc.cprint(tmp_message, 'red','on_yellow',attrs=['bold'])
        tmp_message = 'Planetary radius:',self.planet.radius
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Planetary mass:',self.planet.mass
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'System parameters:'
        tc.cprint(tmp_message, 'red','on_yellow',attrs=['bold'])
        tmp_message = 'Semi-major axis:',self.planet.a
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Impact parameter:', self.planet.b
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Inclination:',self.planet.i
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Eccentricity',self.planet.e
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Period of the planet:',self.planet.P
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Argument of periastron:',self.planet.omega,' or ',self.planet.omega_bar
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Systemic velocity:',self.system.vel_s
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Semimajor-axis to stellar radius ratio:',self.planet.a_rs_ratio
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'RV semiamplitude:',self.planet.semiamp_s
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Transit parameters:'
        tc.cprint(tmp_message, 'red','on_yellow',attrs=['bold'])
        tmp_message = 'Transit center:',self.transit.T_C
        tc.cprint(tmp_message, 'grey','on_white')
        tmp_message = 'Transit depth:',self.transit.delta
        tc.cprint(tmp_message, 'grey','on_white')
    
    def RM_results(self,obliquity,vsini):
        '''
        Add values from RM analysis to the class
        Input:
            obliquity ; value *u.deg- value of obliquity
            vsini ; value *u.km/u.s - value of vsini
        '''
        self.planet.obliquity = obliquity *u.deg
        self.star.vsini = vsini*u.km/u.s
    
    def calculate_v_stel_loc(self,spec_list):
        '''
        Calculates the local stellar velocities based on the formula provided by Omar
        Using LaTeX: 
            RV(\phi)= (\frac{a}{R_s})(sin(2 \pi \phi) cos(\lambda) + cos(2 \pi \phi) cos (i) sin(\lambda) vsin(i)) 
        Input:
            spec_list ; spec_list to which to calculate the stel_loc_vel
        Output:
            None
        Change:
            Updated meta parameters:
                vel_stel_loc ; value
                RM_velocity ; True/False
        '''
        for item in spec_list:
            if item.meta['Transit']:
                item.meta['vel_stel_loc'] = (self.planet.a/self.star.radius).decompose() * (np.sin(2 * np.pi *u.rad * item.meta['Phase'])*np.cos(self.planet.obliquity) +np.cos(2*np.pi*u.rad*item.meta['Phase'])*np.cos(self.planet.i)*np.sin(self.planet.obliquity))*self.star.vsini
                item.meta['RM_velocity'] =True
        return
#%%
# =============================================================================
# TODO: Rewrite transit_value
# Current issue is that it is not clear in how it is written.
# Check for Transit_full and Transit values in phase space
# =============================================================================
# =============================================================================
# TODO: Check that the code runs correctly
# =============================================================================
@todo_function
def transit_value(BJD:float,
                      star_para:SystemParameters
                      ) -> tuple[bool,bool]:
    '''
    Calculates whether given time is during (partial-) transit or out of transit.

    Parameters
    ----------
    BJD : float
        Time of observation.
    star_para : SystemParameters
        System parameters.

    Returns
    -------
    tuple(bool,bool)
        Transit_full : bool
            Spectrum fully in transit
        Transit_partial : bool
            Spectrum partially in transit

    '''
    phase = (((BJD - star_para.transit.T_C) % star_para.planet.P)/star_para.planet.P).value
    
    if phase > 0.5: # Go from [0,1] interval to [-0.5, 0.5] interval
        phase = phase-1
    
    t1,t4 = (-star_para.transit.T14/star_para.planet.P/2).decompose().value,\
            (+star_para.transit.T14/star_para.planet.P/2).decompose().value
    t2,t3 = (-star_para.transit.T23/star_para.planet.P/2).decompose().value,\
            (+star_para.transit.T23/star_para.planet.P/2).decompose().value
    
    if (phase<t1) or (phase>t4): # Out-of-transit
        Transit_full = False
        Transit_partial = False
    elif (phase<t2) or (phase > t3): # Partial transit
        Transit_full = False
        Transit_partial = True
    else: # Full in transit
        Transit_full = True
        Transit_partial = True
    
    return Transit_full, Transit_partial

#%% transit_value
@todo_function
def transit_value_old(BJD,star_para):
    '''Gives 2 parameters whether is spectrum in/out of transit and during ingress/egress
    Input : 
        header ; spectrum header
        star_para ; class of parameters
        
    Output : 
        Transit_full ; spectrum fully in Transit
        Transit_value ; spectrum atleast partly in Transit
    '''
    t1 = (star_para.transit.T_C - (star_para.transit.T14/2))%star_para.planet.P
    t2 = (star_para.transit.T_C - (star_para.transit.T23/2))%star_para.planet.P
    t3 = (star_para.transit.T_C + (star_para.transit.T23/2))%star_para.planet.P
    t4 = (star_para.transit.T_C + (star_para.transit.T14/2))%star_para.planet.P
    
    warn = False
    if t4 < t3 or t4 <t2 or t4<t1:
        t4 = t4 + star_para.planet.P
        warn=True
    if t3< t2 or t3<t1:
        t3 = t3 + star_para.planet.P
        warn=True
    if t2<t1:
        t2 = t2 + star_para.planet.P
        warn=True

    
    if (BJD) %star_para.planet.P<star_para.planet.P/2 and warn:
        BJD = BJD%star_para.planet.P + star_para.planet.P
    else:
        BJD = BJD%star_para.planet.P
    
    if BJD >t1 and BJD <t4: 
        if BJD>t2 and BJD <t3: 
                Transit_full = True
                Transit_value = True
        else:
            Transit_full = False
            Transit_value = True
    else:
        Transit_full = False
        Transit_value = False
    return Transit_full,Transit_value
#%% pre_post_transit
def pre_post_transit(meta: dict):
    '''
    Updates meta parameters for keywords Preingress, Postingress, Transit and Transit full

    Parameters
    ----------
    meta : dict
        Meta information for given spectrum.

    Returns
    -------
    None.

    '''
    if not(meta['Transit_full']):
        if meta['Phase']<0:
            meta['Preingress'] = True
            meta['Postegress'] = False
        if meta['Phase']>0:
            meta['Preingress'] = False
            meta['Postegress'] = True
    else:
        meta['Preingress'] = False
        meta['Postegress'] = False
    return 
#%% update_transit
@progress_tracker
@skip_function
def update_transit(spec_list,sys_para,force_skip=False):
    '''
    Updates the spec_list meta dictionary for in/out of transit values
    Input:
        spec_list ; sp.SpectrumList - spectrum list to update
        sys_para ; system_parameters_class - class with system parameters defined at the beginning
    Output:
        None
    Changes:
        spec_list[:].meta now have 'Transit' and 'Transit_full' values, which are defined by T1-T4 and T2-T3, respectively
    '''
    for item in spec_list:
        Transit_full,Transit_value = transit_value(item.meta['BJD'],sys_para)
        
        item.meta.update({
            'Transit_full':Transit_full,
            'Transit':Transit_value,
            })
        pre_post_transit(item.meta)
    return

#%% observation
# =============================================================================
# TODO: Dataclass usage!
# =============================================================================
class observation:
    '''
    Class of single observation
    Has several parameters for given spectrum describing its parameters like:
        Airmass
        Seeing
        S_N (Signal-to-Noise ratio)
        Transit_part (partial transit)
        Transit_full (full transit)
        index (to identify given spectrum while filtering)
    
    '''
    @todo_function
    def __init__(self,spectrum):
        self.airmass = spectrum.meta['Airmass']
        self.seeing = spectrum.meta['Seeing']
        self.s_n = spectrum.meta['S_N']
        self.exptime = spectrum.meta['Exptime']
        self.transit_part = spectrum.meta['Transit_part']
        self.transit_full = spectrum.meta['Transit_full']
        self.index = spectrum.meta['spec_num']
        self.mask = spectrum.meta
        
        return
#%% observation_night
# =============================================================================
# TODO: Dataclass usage? Simplify
# =============================================================================
class observation_night:
    @todo_function
    def __init__(self,spec_list,orders):
        '''
        Creates a observation_night class
        Includes information for observations in single night
        Input:
            spec_list ; sp.SpectrumList - sublist of spectra for one night
            orders ; List - list of orders to include in SN
        
        Has these attributes:
            night - Night of observation
            night_num - Number of night
            exp_num - Number of exposures
            in_num - Number of in Transit exposures
            out_num - Number of out of transit exposures
            SN - Min-Mean-Max SN for night for each order (ordered by order field)
            Airmass - Min-Mean-Max Airmass for night
            Seeing - Min-Mean-Max Airmass for night
            SN_array - SN for each exposure in night
            Airmass_array - Airmass for each exposure in night
            Seeing_array - Seeing for each exposure in night
            BJD - BJD for each exposure in night
            Phase - Phase for each exposure in night
        '''
        self.night = spec_list[0].meta['Night']
        self.night_num = spec_list[0].meta['Night_num']
        self.exp_num = len(spec_list)
        self.in_num = 0
        self.out_num = 0 
        self.SN = {}
        self.order_list = orders
        SN = []
        Airmass = []
        Seeing = []
        BJD = []
        Phase = []
        Exptime = []
            
        for item in spec_list:
            self.in_num += item.meta['Transit']
            self.out_num += ~item.meta['Transit']
            Airmass.append(item.meta['Airmass'])
            Seeing.append(item.meta['Seeing'])
            BJD.append(item.meta['BJD'])
            Phase.append(item.meta['Phase'])
            Exptime.append(item.meta['Exptime'].value)

            
        if len(orders) !=0:
            for order in orders:
                SN_sgl_order = []
                for item in spec_list:
                    SN_sgl_order.append(item.meta['S_N_all'][order])
                self.SN[order] = "{:.0f}".format(np.min(SN_sgl_order)) + '-' + "{:.0f}".format(np.mean(SN_sgl_order)) + '-' + "{:.0f}".format(np.max(SN_sgl_order))
                SN.append(SN_sgl_order)
        else:
            SN_sgl_order = []
            for item in spec_list:
                SN_sgl_order.append(
                    pd.Series(item.meta['S_N_all'][1:]).quantile(0.8))
                
            self.SN['Median'] = "{:.0f}".format(np.min(SN_sgl_order)) + '-' + "{:.0f}".format(np.mean(SN_sgl_order)) + '-' + "{:.0f}".format(np.max(SN_sgl_order))
            SN.append(SN_sgl_order)
        

        self.Airmass = "{:.2f}".format(np.min(Airmass)) + '-' +"{:.2f}".format(np.mean(Airmass)) + '-' + "{:.2f}".format(np.max(Airmass))
        self.Seeing = "{:.2f}".format(np.min(Seeing)) + '-' + "{:.2f}".format(np.mean(Seeing)) + '-' + "{:.2f}".format(np.max(Seeing))
        self.SN_array = SN
        self.Airmass_array = Airmass
        self.Seeing_array = Seeing
        self.BJD = BJD
        self.Phase = Phase
        self.Exptime = Exptime
        return
#%% observations_log
class observations_log:
    '''
    Class for observations information
    
    Parameters:
        night - list of nights with observation_night class
    '''

    def __init__(self,spec_list,orders):

        self.night = []
        for ni in range(spec_list[0].meta['Night_num'],spec_list[-1].meta['Night_num']+1):
            sublist = sm.get_sublist(spec_list, 'Night_num', ni)
            self.night.append(observation_night(sublist,orders))
        return
#%% observed_exoplanets
# =============================================================================
# Fix several functions within class
# =============================================================================
class observed_exoplanets:
    '''
    Class including all observed exoplanets in the NASA exoplanet archive
    
    '''
    def save(self):
        """save class as self.name.txt"""
        with open('known_exo.pkl', 'wb') as output_file:
            pickle.dump(self.__dict__, output_file)


    def load(self):
        """try load self.name.txt"""
        with open('known_exo.pkl', 'rb') as input_file:
            self.__dict__  =  pickle.load(input_file)
    
    def calculate_insolation_flux_single(self,index,row):
        print(row['pl_insol'].flags)
        if np.isnan(row['pl_insol']):
            if np.isnan(row['st_lum']) or np.isnan(row['pl_orbsmax']):
                pass
            else:
                self.nasa_table.loc[index,'pl_insol'] = 10**row['st_lum'] / row['pl_orbsmax']**2
                row['pl_insol'] = 10**row['st_lum'] / row['pl_orbsmax']**2
        return
    # inputs = arguments_over_which_to_iterate
    @todo_function
    def calculate_insolation_flux_new(self):
        '''
        Tries to recalculate insolation flux for NaNs and adds proper units to the table
        TODO: Add units
        '''
        num_cores = multiprocessing.cpu_count()
        result = Parallel(n_jobs=num_cores)(delayed(self.calculate_insolation_flux_single)(index,row) for index,row in self.nasa_table.iterrows())
        
        return
    @todo_function
    def calculate_insolation_flux(self):
        '''
        Tries to recalculate insolation flux for NaNs and adds proper units to the table
        TODO: Add units
        '''
        ii=0
        for index, row in self.nasa_table.iterrows():
            if np.isnan(row['pl_insol']):
                if np.isnan(row['st_lum']) or np.isnan(row['pl_orbsmax']):
                    ii +=1
                    pass
                else:
                    self.nasa_table.loc[index,'pl_insol'] = 10**row['st_lum'] / row['pl_orbsmax']**2
                    row['pl_insol'] = 10**row['st_lum'] / row['pl_orbsmax']**2
        print('Failed to calculate insolation flux for %i'%(ii)+' planets')
        return
    
    
    def calculate_ksi(self):
        '''
        \ksi = \frac{T_{eq}}{1000K}\frac{g}{g_{J}}
        
        g =GM/r
        can be simplified 
        TODO:
            simplify
            
        '''
        ii = 0
        g_J = con.G * (con.M_jup).to(u.kg)/((con.R_jup).to(u.m)**2)
        
        try:
            self.nasa_table['ksi']
        except:
            self.nasa_table['ksi'] = [np.nan]*len(self.nasa_table)
            
        for index, row in self.nasa_table.iterrows():

            
            try:
                if np.isnan(row['pl_bmassj']) or np.isnan(row['pl_radj']) or np.isnan(row['pl_eqt']):
                    ii += 1
                else:
                    g = con.G * (row['pl_bmassj']*u.M_jup).to(u.kg)/((row['pl_radj']*u.R_jup).to(u.m)**2)
                    self.nasa_table.loc[index,'ksi'] = (row['pl_eqt']/1000) * (g/g_J)
            except:
                ii +=1
                
        print('Failed to calculate ksi parameter for %i'%(ii)+' planets')
        return
    
    
    def calculate_scale_height(self,mol_mass:float = 2.3):
        '''
        Calculates scale height of planets assuming mol_mass. H = \frac{k_b T}{g\mu}

        Parameters
        ----------
        mol_mass : float, optional
            Molecular mass in units of u.kg/u.mol. The default is 2.3 (H+He dominated).

        Returns
        -------
        None. Adds a row with ['H'] in the table in meters

        '''
        ii = 0
        try:
            self.nasa_table['H']
        except:
            self.nasa_table['H'] = [np.nan]*len(self.nasa_table)

        for index, row in self.nasa_table.iterrows():
            try:
                if np.isnan(row['pl_bmassj']) or np.isnan(row['pl_radj']) or np.isnan(row['pl_eqt']):
                    ii += 1
                else:
                    g = con.G * (row['pl_bmassj']*u.M_jup).to(u.kg)/((row['pl_radj']*u.R_jup).to(u.m)**2)
                    self.nasa_table.loc[index,'H'] = (con.N_A *row['pl_eqt'] * u.K  * con.k_B / (mol_mass*u.kg/u.mol) / g).decompose().value
            except:
                ii +=1
        print('Failed to calculate scale height parameter for %i'%(ii)+' planets')
        return
    
    @todo_function
    def __init__(self):
        try: # Test loading the function
            print('Trying to load NASA table')
            self.load()
            if (datetime.datetime.now() - self.time ).days >7: # If not < week data, reload anyway
                print('Reloading NASA table [new day]')
                service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP/") # Initialization of TAP service
                # Searching relevant parameters
                self.nasa_table = pd.DataFrame(service.search("SELECT * FROM pscomppars"))
                self.time = datetime.datetime.now()
                self.calculate_insolation_flux()
                self.calculate_ksi()
                self.calculate_scale_height()
                self.save()
        except: # Loading failed, reload the archive - should happen only the first time running the code
            print('Weird, no table found')
            service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP/") # Initialization of TAP service
            # Searching relevant parameters
            self.nasa_table = pd.DataFrame(service.search("SELECT * FROM pscomppars"))
            self.time = datetime.datetime.now()
            self.calculate_insolation_flux()
            self.calculate_ksi()
            self.calculate_scale_height()
            self.save()
            pass
        return
    
    
