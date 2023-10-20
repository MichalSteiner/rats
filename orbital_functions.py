# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:26:53 2020

orbit functions to calculate all parameters
needed to correct for star and planet motion
from A. Wyttenbach (adapted from V. Bourrier and J. Seidel)

@author: Chamaeleontis
"""
#%% Importing libraries
import astropy.units as u
import numpy as np
import math
from scipy.optimize import newton
from rats.utilities import time_function, save_and_load, progress_tracker, disable_func,default_logger_format,skip_function, todo_function
import logging
import specutils as sp
from typing import Callable, Iterator, Union, Optional, Tuple, Any
#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

#%% Kepler_func
def Kepler_func(ecc_anom,mean_anom,ecc):
    """
    Find the Eccentric anomaly using the mean anomaly and eccentricity
    - M = E - e sin(E)
    """
    delta=ecc_anom-ecc*np.sin(ecc_anom)-mean_anom
    return delta
#%% calc_true_anom
def calc_true_anom(sys_para, phase):
    #Circular orbit
    if math.isclose(sys_para.planet.e,0.,abs_tol=1e-4) == True:
        true_anom=2.*np.pi* phase*u.rad
        #Eccentric anomaly
        ecc_anom=None
        #Eccentric orbit
        #  - by definition the ascending node is where the object goes toward the observer through the plane of sky
        #  - omega_bar is the angle between the ascending node and the periastron, in the orbital plane (>0 counterclockwise)
    else:
        #True anomaly of the planet at mid-transit:
        #    - angle counted from 0 at the perisastron, to the star/Earth LOS
        #    - >0 counterclockwise, possibly modulo 2pi
        true_anom_TR=(np.pi*0.5*u.rad)-sys_para.planet.omega_bar

        #True anomaly at the time of the transit
        #    - corresponds to 'dt_transit' (in years), time from periapsis to transit center
        #    - atan(X) is in -PI/2 ; PI/2
        ecc_anom_TR=2.*np.arctan(np.tan(true_anom_TR*0.5)*np.sqrt((1.-sys_para.planet.e)/(1.+sys_para.planet.e)))

        mean_anom_TR=ecc_anom_TR-sys_para.planet.e*np.sin(ecc_anom_TR)*u.rad
        if (mean_anom_TR<0.):
            mean_anom_TR= mean_anom_TR+2.*np.pi*u.rad

        #Mean anomaly
        #  - time origin of t_mean at the periapsis (t_mean=0 <-> M=0 <-> E=0)
        #  - M(t_mean)=M(dt_transit)+M(t_simu)
        mean_anom=2.*np.pi* phase*u.rad +mean_anom_TR

        #Eccentric anomaly :
        #  - M = E - e sin(E)
        #    - >0 counterclockwise
        #  - angle, with origin at the ellipse center, between the major axis toward the periapsis and the
        #line crossing the circle with radius 'a_Rs' at its intersection with the perpendicular to
        #the major axis through the planet position
        ecc_anom=newton(Kepler_func,mean_anom.value,args=(mean_anom.value,sys_para.planet.e))

        #True anomaly of the planet at current time
        true_anom=2.*np.arctan(np.sqrt((1.+sys_para.planet.e)/(1.-sys_para.planet.e))*np.tan(ecc_anom/2.)) *u.rad
    return true_anom
#%% calc_stellar_vel_sgl
def calc_stellar_vel_sgl(sys_para,phase,multi_pl=False):
    '''
    Calculates stellar velocity for single phase using stellar parameters
    
    Input:
        sys_para ; rats.parameters.system_parameters_class
        phase ; Quantity, dimensionless - phase of the spectrum
    Output:
        vel_star ; Quantity, u.km/u.s - velocity of the star
    '''
    omega_bar = sys_para.planet.omega_bar
    true_anomaly = calc_true_anom(sys_para, phase)
    vel_star = sys_para.planet.semiamp_s.to(u.m/u.s) * \
        (np.cos(true_anomaly+omega_bar) + sys_para.planet.e * np.cos(omega_bar))
    return vel_star
#%% planet_velocity_sgl
# =============================================================================
# Make it work for eccentric orbits
# =============================================================================
@todo_function
def planet_velocity_sgl(sys_para,phase):
    '''
    Calculates planetary velocity. Assumes circular orbit.
    
    Input:
        sys_para ; rats.parameters.system_parameters_class
        phase ; Quantity, dimensionless - phase of the spectrum
    Output:
        vel_plan ; Quantity, u.km/u.s - velocity of the planet
    '''
    vel_plan = (2. * np.pi * sys_para.planet.a.to(u.m)/\
                       (sys_para.planet.P.to(u.s))) * \
        np.sin(2.*np.pi * u.rad *phase) * \
            np.sin(np.radians(sys_para.planet.i))
    return vel_plan

#%% update_phase_and_vel
@progress_tracker
@skip_function
def update_phase_and_vel(spec_list: sp.SpectrumList,
                         star_para: Any,
                         force_skip: bool = False) -> sp.SpectrumList:
    '''
    Updates phases and planetary + stellar velocities in the spectrum list object

    Parameters
    ----------
    spec_list : SpectrumList
        Spectrum list to update meta parameters in.
    star_para : SystemParameters
        System parameters.
    force_skip : bool, optional
        Skip this function with skip_function decorator. The default is False.

    Returns
    -------
    SpectrumList
        Updated SpectrumList with velocities and phases for each observation.

    '''
    for item in spec_list:
        phase = sm.phase_fold(item.meta['BJD'],star_para)
        vel_pl = of.planet_velocity_sgl(star_para,phase)
        vel_st = of.calc_stellar_vel_sgl(star_para,phase,multi_pl=False)
        item.meta.update({
                'Phase':phase,
                'vel_pl':vel_pl,
                'vel_st':vel_st,
                'vel_sys':star_para.system.vel_s
                }
            )
    return spec_list
