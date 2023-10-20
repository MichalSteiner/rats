# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:11:31 2021

@author: Chamaeleontis
"""

'''
One-time use functions to setup tree directory to put data in

Functions:
    set_tree_directory ; set a tree directory for given planet
    make_tree_directory_spectra_night ; creates a subdirectory for given spectra night observation
    find_fiber ; finds a fiber from filename
    find_type_of_spec ; finds a spectral type of spectra
    setup_directory ; setups main directory and moves data to it
    setup_routine ; automated routine for start of analysis
    setup_directory_paths ; returns directory paths used in other functions

Usage:
    data_directory = 'path-to-data' # Change to path where your data is
    main_directory = 'path-to-main-directory' # Change to path where your main directory is
    setup_routine(data_directory,main_directory,instruments)
    # This will do everything necessary for setup



'''
#%% Importing libraries
import os
import shutil
import termcolor as tc
import re
import tarfile
import astropy.io.fits as fits
from rats.utilities import logger

#%% Open tar.gz file
def open_tarfile(directory: str,
                 filename: str)-> None:
    """
    Open a tar.gz zip file downloaded from DACE.

    This is useful when working on filesystem (e.g., FAT) that don't allow ":" in the name, as these are commonly used in HARPS, ESPRESSO and NIRPS names. In that case, it will automatically replace the ":" with "_", which is usable in most common filesystems.
    
    Parameters
    ----------
    directory : str
        The directory where the tar file is located
    filename : str
        The filename of the tar.gz file

    Returns
    -------
    None
    """
    assert filename.endswith('tar.gz'), "Not a suitable tar.gz file"
    
    file = tarfile.open(filename)
    # Rename files to valid name
    for ind,member in enumerate(file.getmembers()):
        member.name = member.name.replace(':','_')
    file.extractall(directory)
    file.close()

    return None

#%% Filter files inside the folder
def set_tree_directory(original_directory: str,
                       main_directory: str,
                       file_types: list = ['S1D', 'S2D', 'CCF']):
    """
    Set the basic tree directory.

    Parameters
    ----------
    original_directory : str
        Original directory where the data is saved.
    main_directory : str
        Main directory of the project
    file_types : list, optional
        List of file types, currently expecting the S1D, S2D and CCF formats, by default ['S1D', 'S2D', 'CCF'].
    """
    if len(os.listdir(original_directory)) == 1 and os.listdir(original_directory)[0].endswith('tar.gz'):
        filename = os.listdir(original_directory)[0]
        logger.info('Found only tar.gz file. Opening it.')
        open_tarfile(original_directory + '/' + filename)
    
    for night_folder in os.listdir(original_directory):
        if night_folder.endswith('.tar.gz'):
            continue
        logger.info('Found night folder: \n'+str(original_directory + '/' + night_folder))
        
        for file in os.listdir(original_directory +
                               '/' +
                               night_folder):
            create_folder_general(main_directory= main_directory)
            
            create_folder_spectra(main_directory= main_directory,
                          night=  night_folder,
                          full_filepath= original_directory + '/' +
                               night_folder + '/' +
                               file,
                          filename = file,
                          file_type= file_types)
    return

#%% Create general folders
def create_folder_general(main_directory: str):
    """
    Create a general tree directory.

    Parameters
    ----------
    main_directory : str
        Target path of the root directory.
    """
    os.makedirs(main_directory + '/' +
                'code',
                mode = 0o777,
                exist_ok = True) 
    os.makedirs(main_directory + '/' +
                'spectroscopy_data',
                mode = 0o777,
                exist_ok = True) 
    os.makedirs(main_directory + '/' +
                'photometry_data',
                mode = 0o777,
                exist_ok = True)
    os.makedirs(main_directory + '/' +
                'figures',
                mode = 0o777,
                exist_ok = True)
    os.makedirs(main_directory + '/' +
                'tables',
                mode = 0o777,
                exist_ok = True)
    os.makedirs(main_directory + '/' +
                'presentation',
                mode = 0o777,
                exist_ok = True)
    os.makedirs(main_directory + '/' +
                'paper/figures',
                mode = 0o777,
                exist_ok = True)
    return


#%% Create folder for each spectrum
def create_folder_spectra(main_directory: str,
                  night: str,
                  full_filepath: str,
                  filename: str,
                  file_type: list):
    """
    Create a tree directory for given set of observations.

    Parameters
    ----------
    main_directory : str
        Root target directory. Everything will be set there.
    night : str
        The night folder currently worked on.
    full_filepath : str
        Full filepath to given fits file.
    filename : str
        Filename of the fits file currently worked on
    file_type : list
        Which type of files to consider
    """
    for fits_type in file_type:
        if fits_type in full_filepath:
            header = fits.open(full_filepath)[0].header
            instrument = header['INSTRUME']
            fiber = 'Fiber_' + full_filepath.replace('.fits','')[-1]
            start_ind = full_filepath.find(fits_type)
            end_ind = full_filepath.find('_' + full_filepath.replace('.fits','')[-1])
            
            subformat = full_filepath[start_ind:end_ind]
            if subformat == fits_type:
                subformat = 'raw'
            os.makedirs(main_directory + '/' +
                        'spectroscopy_data' + '/' +
                        instrument + '/' + 
                        night + '/' +
                        fiber + '/' +
                        fits_type + '/' +
                        subformat,
                        mode = 0o777,
                        exist_ok = True)
            if subformat in ['S1D_SKYSUB']:
                create_molecfit_folders(header= header,
                                        target= main_directory + '/' +
                                        'spectroscopy_data' + '/' +
                                        instrument + '/' + 
                                        night + '/' +
                                        fiber + '/' +
                                        fits_type + '/'
                                        )
                copy_files(origin= full_filepath,
                           target= main_directory + '/' +
                           'spectroscopy_data' + '/' +
                           instrument + '/' + 
                           night + '/' +
                           fiber + '/' +
                           fits_type + '/' +
                           'molecfit/molecfit_input/' +
                           filename)
            
            
            copy_files(origin= full_filepath,
                       target= main_directory + '/' +
                        'spectroscopy_data' + '/' +
                        instrument + '/' + 
                        night + '/' +
                        fiber + '/' +
                        fits_type + '/' +
                        subformat + '/' +
                        filename)
    return
#%% copy files
def copy_files(origin:str,
               target:str):
    """
    Copy files from origin to target.

    Parameters
    ----------
    origin : str
        Origin path of the file. 
    target : str
        Target path for the file
    """
    
    logger.info('Moving file:\n'+ origin)
    logger.info('to:\n' +target)
    logger.info('='*50)
    # shutil.copy(origin,
    #             target)
    
    return

def create_molecfit_folders(header: fits.header.Header,
                            target: str):
    """
    Create a molecfit folder extracted from the FOLDERS_molecfit based on instrument.

    Parameters
    ----------
    header : fits.header.Header
        Header of the fits file.
    target : str
        Target destination for the directory tree folder.
    """
    os.makedirs(target,
                mode = 0o777,
                exist_ok = True)
    instrument = header['INSTRUME']
    UT = ''
    if instrument == 'ESPRESSO':
        telescope = header['TELESCOP']
        UT = '_' + ''.join(filter(str.isdigit, telescope))
    
    source = os.path.dirname(__file__) + '/FOLDERS_molecfit/' + instrument + UT
    shutil.copytree(source, target, dirs_exist_ok = True)
    return

#%% setup_routine
def setup_routine(original_directory: str,
                  main_directory: str,
                  file_types: list = ['S1D', 'S2D', 'CCF']):
    """
    Full setup for a new dataset

    Parameters
    ----------
    original_directory : str
        Original directory - data are saved there (zipped or unzipped)
    main_directory : str
        Project directory - data will be moved there, including the new tree directory.
    file_types : list, optional
        Which type of file types to check for, by default ['S1D', 'S2D', 'CCF']
    """
    set_tree_directory(original_directory,
                       main_directory,
                       file_types = file_types)
    return

if __name__ == '__main__':
    logger.info(os.getcwd())
    logger.critical('Please enter the original directory with data.')
    original_directory = input()
    logger.info(original_directory)
    setup_routine(original_directory= str(original_directory),
                  main_directory= os.getcwd(),
                  file_types = ['S1D', 'S2D', 'CCF'])
    
    
