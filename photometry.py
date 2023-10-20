#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:08:31 2022

@author: chamaeleontis
"""
#%% Import libraries
import os
import numpy as np
import termcolor as tc
import re
import astropy.units as u
import batman
import matplotlib.pyplot as plt
import rats.spectra_manipulation as sm
import seaborn as sns
from rats.utilities import disable_func
#%% load_lc
@disable_func
def load_lc(lc_directory,list_filters,alias_filters):
    '''
    Load each light-curve analyzed in the lc_directory and returns a list of lightcurve class
    Lightcurves come from the output files created by CONAN with name 'lc'_out.dat
    These should be detrended by the parameters in the setup file
    Limb-darkening coefficients are used the same as in CONAN
    
    Input:
        lc_directory ; string - location of directory with the lightcurves
            Lightcurves name should have the notation yyyymmdd.filter.instrument
        list_filters ; list - list of filters used
            name of the filter is based on the filename (the filter string should be in the filename of )
        
    Output:
        updates the class with all lightcurves used in CONAN
    '''
    tmp_message = 'Check that there are no duplicate names in filename and filter,\n so each lightcurve is loaded with correct filter'
    tc.cprint(tmp_message, 'grey','on_green')
    list_lc = []
    for item in os.listdir(lc_directory): # Go through entire directory
        for ind,lc_filter in enumerate(list_filters): # Go through all filters used
            if '.'+lc_filter+'.' in item: # Find filter in name
                print(item,lc_filter)
                lc = np.loadtxt(lc_directory + '/' + item,skiprows=1)
                list_lc.append([lc,lc_filter,alias_filters[ind],item])
                break
    return list_lc
#%% load_batman_model
@disable_func
def load_batman_model(CONAN_output,lc,star_para,ecc=0,ax=None,value=0):
    '''
    TODO:
        Implement eccentricity output
    Input:
        CONAN_output ; string - filename of CONAN output
        lc ; list - list of lightcurve to model
    Output:
        params ; batman.TransitParams()
    '''
    text_file = open(CONAN_output, "r")
    lines = text_file.read().split('\n')
    for line in lines:
        line = re.split("\s+",line)
        if 'T_0' in line:
            T0 = float(line[1])
        elif 'Period_[d]' in line:
            P = float(line[1])
        elif 'Rp_[Rjup]' in line:
            Rp = ((line[1] * u.R_jup).to(u.R_sun))
        elif 'a_[au]' in line:
            a = ((line[1] * u.au).to(u.R_sun))
            
        elif 'inclination_[deg]' in line:
            inc = float(line[1])
        elif lc[2]+'_c1' in line:
            c1 = float(line[1])
        elif lc[2]+'_c2' in line:
            c2 = float(line[1])
        elif 'Rs_[Rsun]' in line:
            Rs = float(line[1])
        elif 'Transit_dur' in line:
            T14 = float(line[1])
    if ecc == 0:
        w = 90
    Rp = Rp / Rs
    a = a / Rs
    # (self.planet.radius.to(u.R_sun) / self.star.radius).value
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = T0                        #time of inferior conjunction
    params.per = P                       #orbital period
    params.rp = Rp                       #planet radius (in units of stellar radii)
    params.a = a                        #semi-major axis (in units of stellar radii)
    params.inc = inc                      #orbital inclination (in degrees)
    params.ecc = ecc                       #eccentricity
    params.w = w                        #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"        #limb darkening model
    params.u = [c1, c2]      #limb darkening coefficients [u1, u2, u3, u4]
    
    date_new = (((lc[0][:,0] - T0)/P))%1
    for ind,date in enumerate(date_new):
        if date >0.5:
            date_new[ind]= date-1
    
    gaps = np.diff(date_new)
    median_gaps = np.median(gaps)
    ind_mask = np.where(gaps>10*median_gaps)
    
    date_new = np.ma.array(date_new)
    date_new[ind_mask] = np.ma.masked
    
    ax.errorbar(date_new,lc[0][:,1]-value*0.025,yerr=lc[0][:,2],fmt='.',color='grey')
    ax.plot(date_new,lc[0][:,5]-value*0.025,color=sns.color_palette("dark")[value],zorder=999)
    return params

