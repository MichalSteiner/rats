#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:58:42 2022

@author: chamaeleontis
"""

#%% Importing libraries
import rats.utilities as util
import rats.spectra_manipulation as sm

# from astroquery.nist import Nist
# from astroquery.atomic import AtomicLineList
# from astroquery.hitran import Hitran
import astropy
from astropy import units as u
import specutils as sp
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as con
#%% save_ccf
def save_ccf(x,y,z,element,data_directory):
    '''
    Saves the output of the CCF file
    '''
    util.save_pickle([x,y,z], data_directory+'/' + element+'.pkl')
    return


#%%
def find_nearest(array, value):
    '''
    Find nearest value index in array
    '''
    array = np.asarray(array,dtype=np.dtype(float))
    idx = (np.abs(array - value)).argmin()
    return idx
# %% Vacuum to Air and viceversa conversion
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

#%% load_line_list_hitran
def load_line_list_hitran():
    '''
    TODO:
        Write the function

    Returns
    -------
    None.

    '''
    return

#%%
def create_weigthed_template(spectral_axis: sp.SpectralAxis,
                             line_list: list,
                             strength_list: list,
                             velocity: u.Quantity,
                             vacuum: bool = True,
                             ) -> sp.Spectrum1D:
    
    if vacuum:
        line_list = vactoair(np.asarray(line_list))
    else:
        line_list = np.asarray(line_list)
    
    strength_list = np.asarray(strength_list)
    strength_list = np.log(strength_list)
    strength_list = strength_list / np.mean(strength_list)
    line_list = line_list*(1+velocity/con.c)
    
    line_list = line_list[line_list>spectral_axis[0].value] # Cut line list to used wavelength range
    line_list = line_list[line_list<spectral_axis[-1].value] # Cut line list to used wavelength range
    
    flux = np.zeros(spectral_axis.shape) # Allocate flux array to zeros (no-line)
    
    for item, strength in zip(line_list, strength_list): # For line in line list
        idx = find_nearest(spectral_axis.value,item) # Find nearest pixel for given line
        flux[idx] = strength # Assign strength
        
    template_spectrum = sp.Spectrum1D(
        spectral_axis = spectral_axis,
        flux = flux*u.dimensionless_unscaled,
        uncertainty = astropy.nddata.StdDevUncertainty(np.full_like(flux, 0)),
        wcs = sm.add_spam_wcs(1),
        meta = {'velocity':velocity},
        )
    return template_spectrum

#%% load_line_list_nist
def load_line_list_nist(spectral_axis,element):
    '''
    Load line list from astroquery.nist 
    
    Line strength descriptors at: https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html#OUTRELINT
    
    
    
    '''
    table = Nist.query(spectral_axis[0], spectral_axis[-1], linename=element,wavelength_type='vacuum')
    
    
    line_list = []
    strength_list = []
    for line, strength in zip(table['Observed'], table['Rel.']):
        print(line,strength)
        try:
            float(strength) # Test for line strength
            float(line) # Test for line position
        except:
            continue
        
        if float(strength) > 100: # Filter by stronger lines
            line_list.append(float(line))
            strength_list.append(float(strength))
    
    
    return line_list, strength_list
#%% load_line_list_atomic
def load_line_list_atomic(spectral_axis,element):
    '''
    Load line list from astroquery.atomic
    
    Input:
        spectral_axis ; sp.SpectralAxis - spectral axis for the mask
        element ; str - which element to look for
    Output:
        template_mask = sp.Spectrum1D - binary mask of line lists
    '''
    
    wavelength_range = (spectral_axis[0], spectral_axis[-1])
    
    table = AtomicLineList.query_object(wavelength_range=wavelength_range, wavelength_type='Air',
                                element_spectrum=element)
    line_list = table['LAMBDA VAC ANG'].value

    return line_list

#%% create_template
def create_template(spectral_axis, element):
    '''
    Create templates for calculating CCF
    
    Input:
        spectral_axis ; sp.SpectralAxis - wavelength range to which to calculate the template
        list_elements ; list of str - lists of elements to search for
    Output:
        template_masks ; sp.SpectrumList - list of templates masks for given element
    
    '''
    line_lists = []
    try:
        line_lists.append(['NIST',load_line_list_nist(spectral_axis,element)])
    except:
        pass
    try:
        line_lists.append(['Atomic',load_line_list_atomic(spectral_axis,element)])
    except:
        pass

    return line_lists

#%% create_template_list
def create_template_list(spectral_axis,list_elements):
    '''
    Create list of templates for list of elements
    
    Input:
        spectral_axis ; sp.SpectralAxis - wavelength range to which to calculate the template
        list_elements ; list of str - lists of elements to search for
    Output:
        template_list ; sp.SpectrumList - list of templates 
        
    '''
    elements_list = []
    for element in list_elements:
        elements_list.append([element,create_template(spectral_axis,element)])

    return elements_list
#%% create_flux_mask
def create_flux_mask(spectral_axis,
                     line_list,
                     vel = 0*u.m/u.s,
                       ):
    '''
    Finds the indices of nearest pixel on spectral_axis
    Assumes values in \AA
    
    Input:
        spectral_axis ; sp.Spectral_axis,
        line_list - input line_list['Observed'].value for NIST
        vel - velocity by which to shift the line list
    Output:
        flux
    
    '''
    line_list = line_list*(1+vel/con.c)
    
    line_list = line_list[line_list>spectral_axis[0].value] # Cut line list to used wavelength range
    line_list = line_list[line_list<spectral_axis[-1].value] # Cut line list to used wavelength range
    
    
    flux = np.zeros(spectral_axis.shape) # Allocate flux array to zeros (no-line)
    
    for item in line_list: # For line in line list
        idx = find_nearest(spectral_axis.value,item) # Find nearest pixel for given line
        flux[idx] = -1 # Assign -1
        
    template_spectrum = sp.Spectrum1D(
        spectral_axis = spectral_axis,
        flux = flux*u.dimensionless_unscaled,
        uncertainty = astropy.nddata.StdDevUncertainty(np.full_like(flux, 0)),
        wcs = sm.add_spam_wcs(1),
        meta = {'velocity':vel},
        )
    return template_spectrum # return template_spectrum
#%%
def write_CCF(x,y,z):
    
    CCF = sp.Spectrum1D(
        spectral_axis = x,
        flux = z,
        meta = {},
        mask = np.isnan(z),
        wcs = sm.add_spam_wcs(),
        
        
        )
    return

#%%
# def calculate_CCF_value(spectrum, template, velocity):
    
    
#     return


# from joblib.externals.loky import set_loky_pickler
# from joblib import parallel_backend
# from joblib import Parallel, delayed
# import multiprocessing
# num_cores = multiprocessing.cpu_count()
# new_spec_list = sp.SpectrumList(
#     Parallel(n_jobs=num_cores)(delayed(interpolate2commonframe)(i,new_spectral_axis) for i in spec_list)
#     )

# z_array = [
#     Parallel(n_jobs=num_cores)(delayed(calculate_CCF_value)(i,new_spectral_axis) for i in spec_list)
#     ]

# set_loky_pickler('pickle')


#%% crosscorrelate
def crosscorrelate(spec_list:sp.SpectrumList,
                   template: sp.Spectrum1D,
                   ):
    '''
    Crosscorrelate spectrum and template mask
    
    Input:
        spec_list ; sp.SpectrumList - spectra on which to crosscorrelate the mask
            Needs to be on common wavelength axis
        template_mask ; sp.Spectrum1D - template mask to use
    Output:
        ccf_spectrum ; sp.Spectrum1D - cross-correlated spectrum
    '''
    x_axis = np.arange(-200,200,0.5)
    y_axis = np.arange(len(spec_list))
    z_axis = np.zeros((len(x_axis),len(y_axis)))
    
    
    for ind,velocity in enumerate(x_axis):
        print('Calculating crosscorrelation for velocity: ',velocity)
        print('Progress :',(ind*100/len(x_axis)),'%')

        shifted_template = sm.shift_spectrum(template, velocities = [-velocity * u.km/u.s])
        for ii, spectrum in enumerate(spec_list):
            nan_ind = np.logical_and(np.isfinite(spectrum.flux),
                                     np.isfinite(spectrum.uncertainty.array)
                                     )
            correlated_values = spectrum.flux[nan_ind] *\
                                shifted_template.flux[nan_ind] *\
                                (1/spectrum.uncertainty.array[nan_ind]**2)
            
            z_axis[ind,ii] = np.sum(correlated_values) / -np.sum(shifted_template.flux[nan_ind]*
                                                                 (1/spectrum.uncertainty.array[nan_ind]**2)
                                                                 )
            
            assert np.isfinite(z_axis[ind, ii])
            
    return x_axis,y_axis,z_axis

#%% cross_correlate_multiple_elements
def cross_correlate_multiple_elements(spec_list,elements_lists):
    '''
    Crosscorrelate spectra on each of the elements
    
    '''
    data_ccf = []
    
    for element in elements_lists:
        for line_list in element[1]:
            data_ccf.append([element[0],line_list[0],crosscorrelate(spec_list,line_list[1])])
    
    
    return data_ccf


