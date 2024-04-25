# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:11:31 2021

@author: Chamaeleontis
"""

'''
One-time use functions to setup tree directory to put data in from a DACE downloaded data folder.

TODO:
    Check that filesystem allows : in filename. If yes, no need to rename the files.
    Add a check that this function was already run.


'''
#%% Importing libraries
import os
import shutil
import tarfile
import astropy.io.fits as fits
from rats.utilities import default_logger_format
import logging

# Setup logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

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

    file = tarfile.open(directory + '/' + filename)
    # Rename files to valid name
    logger.info('Opening tar gz file:')
    logger.info('   ' + directory + '/' + filename)
    logger.warning('This operation can take long time (~10minutes) if the tar.gz file is large.')
    logger.warning('    Length of the file is: ' + str(len(file.getmembers())))
        
    for ind,member in enumerate(file.getmembers()):
        member.name = member.name.replace(':','_')
    file.extractall(directory)
    file.close()

    return None

#%% Filter files inside the folder
def _set_tree_directory(original_directory: str,
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
        open_tarfile(directory= original_directory,
                     filename= filename)
    
    for night_folder in os.listdir(original_directory):
        if night_folder.endswith('.tar.gz'):
            continue
        logger.info('Found night folder: \n'+str(original_directory + '/' + night_folder))
        
        for file in os.listdir(original_directory +
                               '/' +
                               night_folder):
            _create_folder_general(main_directory= main_directory)
            
            _create_folder_spectra(main_directory= main_directory,
                          night=  night_folder,
                          full_filepath= original_directory + '/' +
                               night_folder + '/' +
                               file,
                          filename = file,
                          file_type= file_types)
    return

#%% Create general folders
def _create_folder_general(main_directory: str):
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
    os.makedirs(main_directory + '/' +
                'figures/whitemode_normal',
                mode = 0o777,
                exist_ok = True)
    os.makedirs(main_directory + '/' +
            'figures/darkmode_normal',
            mode = 0o777,
            exist_ok = True)
    os.makedirs(main_directory + '/' +
            'figures/whitemode_presentation',
            mode = 0o777,
            exist_ok = True)
    os.makedirs(main_directory + '/' +
            'figures/darkmode_presentation',
            mode = 0o777,
            exist_ok = True)
    os.makedirs(main_directory + '/' +
                'saved_data',
                mode = 0o777,
                exist_ok = True)
    return


#%% Create folder for each spectrum
def _create_folder_spectra(main_directory: str,
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
                _create_molecfit_folders(header= header,
                                        target= main_directory + '/' +
                                        'spectroscopy_data' + '/' +
                                        instrument + '/' + 
                                        night + '/' +
                                        fiber + '/' +
                                        fits_type + '/'
                                        )
                _copy_files(origin= full_filepath,
                           target= main_directory + '/' +
                           'spectroscopy_data' + '/' +
                           instrument + '/' + 
                           night + '/' +
                           fiber + '/' +
                           fits_type + '/' +
                           'molecfit/molecfit_input/' +
                           filename)
            
            
            _copy_files(origin= full_filepath,
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
def _copy_files(origin:str,
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
    
    logger.debug('Moving file:\n'+ origin)
    logger.debug('to:\n' +target)
    logger.debug('='*50)
    shutil.copy(origin,
                target)
    
    return

def _create_molecfit_folders(header: fits.header.Header,
                             target: str):
    """
    Create a molecfit folder extracted from the FOLDERS_molecfit based on instrument.

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
        UT = '_UT' + ''.join(filter(str.isdigit, telescope))
    
    source = os.path.dirname(__file__) + '/FOLDERS_molecfit/' + instrument + UT
    shutil.copytree(source, target, dirs_exist_ok = True)
    return

#%% setup_routine
def setup_routine(original_directory: str,
                  main_directory: str,
                  file_types: list = ['S1D', 'S2D', 'CCF'],
                  rerun:bool = False):
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
    if os.path.exists(main_directory+'/spectroscopy_data'):
        if not(rerun) and len(os.listdir(main_directory+'/spectroscopy_data')) != 0:
            return
    
    logger.info('Preparing the default tree directory for the dataset.')
    _set_tree_directory(original_directory,
                        main_directory,
                        file_types = file_types)
    return


if __name__ == '__main__':
    logger.info('Testing setup for rats.single_use module')
    
    original_directory = '/media/chamaeleontis/Observatory_main/Data_all_raw/test2_TOI132'
    main_directory = '/media/chamaeleontis/Observatory_main/Analysis_dataset/rats_test'
    
    setup_routine(original_directory= original_directory,
                  main_directory= main_directory,
                  file_types= ['S1D','S2D','CCF'])
    
    logger.info('Test succesful. Full setup done in directory:')
    logger.info('    ' + main_directory)