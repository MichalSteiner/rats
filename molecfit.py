# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:13:38 2021

@author: Chamaeleontis
"""

#%% Importing libraries
import numpy as np 
import astropy.io.fits as fits
import astropy.units as u
import specutils as sp
import os as os
import astropy
import rats.eso as eso
import rats.spectra_manipulation as sm
from rats.utilities import time_function, save_and_load, progress_tracker, disable_func, skip_function, todo_function

#%% molec_fit_table_fits
def molec_fit_table_fits(spec_coll):
    '''
    Creates a table.fits file compatible with molecfit
    Input:
        spectrum ; sp.Spectrum1D - spectrum to correct for
    Output:
        Creates in specified directory table.fits file that is compatible with molecfit
    '''
    for ii in range(len(spec_coll)):
        spectrum = spec_coll[ii]
        name = spectrum.meta['file_fiber_A']
        name_directory = name[:name.find('raw')]
        os.makedirs(name_directory  + 'molecfit_input/'+'%i'%(ii), mode = 0o777, exist_ok = True) 
        os.makedirs(name_directory  + 'molecfit_output/'+'%i'%(ii), mode = 0o777, exist_ok = True) 
        header = spectrum.meta['header']
        flux_data = spectrum.flux.value
        err_data = spectrum.uncertainty.array
        wavelength = spectrum.spectral_axis.value
        wave_data = fits.Column(name='wavelength', format = '1D', array = wavelength)
        flux_data = fits.Column(name='flux',format = '1D', array = flux_data)
        err_data = fits.Column(name='err_flux', format = '1D', array = err_data)
        cols = fits.ColDefs([wave_data, flux_data, err_data])
        t = fits.BinTableHDU.from_columns(cols)
        prihdr = fits.Header(copy=True)
        prihdr =  header
        prihdu = fits.PrimaryHDU(header=prihdr)
        thdulist = fits.HDUList([prihdu, t])
        thdulist.writeto(name.replace('raw','molecfit_input/%i'%(ii)))
    return

#%% molecfit_new_output
# =============================================================================
# Likely bugged with telluric profiles being in different rest frame
# =============================================================================
@todo_function
@progress_tracker
def molecfit_new_output(instrument_directory: str,
                        spec_type: str = 'S1D',
                        mask_threshold: float = 0,
                        )->[sp.SpectrumList, sp.SpectrumList, sp.SpectrumList]:
    '''
    Load molecfit output files.

    Parameters
    ----------
    instrument_directory : str
        Location of data for instrument
    spec_type : str, optional
        Type of the spectra (defines the spec_type directory). The default is 'S1D'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    spec_list = sp.SpectrumList()
    spec_list_profile = sp.SpectrumList()
    spec_list_noncorrected = sp.SpectrumList()
    
    for ind,night in enumerate(os.listdir(instrument_directory)): # Looping through each night
        molecfit_output_directory = instrument_directory + '/' + night + '/Fiber_A/S1D/molecfit/molecfit_output'
        
        for ii,item in enumerate(os.listdir(molecfit_output_directory)): # Looping through each file
            if item.startswith('SCIENCE'): # Only SCIENCE fits files
            
                f = fits.open(molecfit_output_directory + '/' + item) 
                meta = {'header':f[0].header}
                meta.update(eso.load_meta_from_header_new_pipeline(f[0].header))
                meta.update(eso.set_meta_parameters())
                meta.update(eso.set_meta_velocity_corrections())
                meta.update({
                                'Night':night,
                                'Night_num': ind+1,
                                'Night_spec_num':ii+1,
                    })
                
                eso.set_filename_header(meta,molecfit_output_directory + '/' + item)
                meta['blaze_corrected'] = True
                meta['telluric_corrected'] = True
                meta['S_N']= meta['Average S_N']
                meta.update({
                                'Night':night,
                                'Night_num': ind+1,
                                'Night_spec_num':ii+1,
                    })
                mask_ind = np.where(f[2].data<mask_threshold)
                flux = f[1].data['flux']
                error = f[1].data['error']
                flux[mask_ind] = np.nan
                error[mask_ind] = np.nan
                
                tmp_spec = sp.Spectrum1D( # Define the spectrum
                    spectral_axis = f[1].data['wavelength_air']*u.AA,
                    flux = flux*u.ct,
                    uncertainty = astropy.nddata.StdDevUncertainty(error),
                    meta = meta,
                    mask = np.isnan(flux),
                    wcs = sm.add_spam_wcs(),
                    )
                spec_list.append(tmp_spec) # Append the spectrum
                
                tmp_spec_profile = sp.Spectrum1D( # Define the spectrum
                    spectral_axis = f[1].data['wavelength_air']*u.AA,
                    flux = f[2].data*u.ct,
                    wcs = sm.add_spam_wcs(),
                    )
                spec_list_profile.append(tmp_spec_profile) # Append the spectrum
                
                tmp_spec_noncorrected = sp.Spectrum1D( # Define the spectrum
                    spectral_axis = f[3].data['wavelength_air']*u.AA,
                    flux = f[3].data['flux']*u.ct,
                    uncertainty = astropy.nddata.StdDevUncertainty(f[3].data['error']),
                    meta = meta,
                    mask = np.isnan(f[3].data['flux']),
                    wcs = sm.add_spam_wcs(),
                    )
                spec_list_noncorrected.append(tmp_spec_noncorrected) # Append the spectrum
                
                del(tmp_spec) # Delete the variable
                del(tmp_spec_profile)
                
    eso.set_meta_numbering(spec_list)
    eso.set_meta_numbering(spec_list_noncorrected)
    return spec_list, spec_list_profile,spec_list_noncorrected # Returns


#%%
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

#%% import_spectra_output
def import_spectra_output_e2ds(root_directory_spec,order):
    '''
    Input:
        root_directory_spec ; directory of spectra data
    Output:
        spec_list ; sp.SpectrumList - telluric corrected
    '''
    # Allocation of spectrum lists
    # All spectra
    spec_list = sp.SpectrumList()
    # Spectra from one night
    tmp_spec_list = sp.SpectrumList()
    profile_list = sp.SpectrumList()
    # For cycle
    for instrument in os.listdir(root_directory_spec):
        # Going through all instrument, assigning night directory
        night_directory = root_directory_spec + '/' + instrument
        for ind,night in enumerate(os.listdir(night_directory)):
            # Going through spectra from one night
            # Where is data saved
            telluric_corrected_directory = night_directory + '/' + night + '/Fiber_A/e2ds/telluric_corrected/%i'%(order) 
            # Cleaning one night spec_list
            tmp_spec_list = sp.SpectrumList()
            try:
                os.listdir(telluric_corrected_directory)
            except:
                break

            for spectra in os.listdir(telluric_corrected_directory):
                # For cycle for all spectra in directory
                filename = telluric_corrected_directory + '/' + spectra
                # Open table
                telluric_table_out = fits.open(filename)
                # Get flux, wavelength and error
                tmp_flux = telluric_table_out[1].data['tacflux'] * u.ct
                tmp_spectral_axis = telluric_table_out[1].data['wavelength'] * u.AA
                tmp_uncertainty = astropy.nddata.StdDevUncertainty(telluric_table_out[1].data['err_flux'] ,unit='ct')
                # Copy uncertainty * 0 for telluric profile (assuming perfect telluric profile)
                tmp_uncertainty_profile = astropy.nddata.StdDevUncertainty(telluric_table_out[1].data['err_flux'] *0 ,unit=u.dimensionless_unscaled)
                # Updating header for the spectrum
                header = telluric_table_out[0].header
                meta  = eso.load_meta_from_header_old_pipeline(header)
                meta.update(eso.set_meta_parameters())
                meta.update({'header':header})
                meta['blaze_corrected'] = True
                meta['telluric_corrected'] = True
                # Appending to one night spec_list
                tmp_spec_list.append(sp.Spectrum1D(
                spectral_axis = tmp_spectral_axis,
                flux = tmp_flux,
                uncertainty = tmp_uncertainty,
                meta = meta,
                mask = np.isnan(tmp_flux),
                wcs = sm.add_spam_wcs(),
                ))
                # Appending telluric profile list 
                profile_list.append(sp.Spectrum1D(
                flux = telluric_table_out[1].data['mtrans'] * u.dimensionless_unscaled,
                spectral_axis = telluric_table_out[1].data['wavelength'] * u.AA,
                uncertainty = tmp_uncertainty_profile, 
                mask = None,
                meta = None,
                wcs = sm.add_spam_wcs(),
                ))
            print(tmp_spectral_axis)
            # Update one night meta data
            eso.set_meta_night_parameters(tmp_spec_list, night, (ind+1))
            # Append to resulting spectrum list
            spec_list.extend(tmp_spec_list)
    # Update spectrum numbering
    eso.set_meta_numbering(spec_list)
    return spec_list,profile_list

