

import subprocess
from rats.setup_filenames import RASSINE_location, RASSINE_config, RASSINE_command
import os
import specutils as sp
import pandas as pd
from fileinput import FileInput 
import numpy as np
import rats.spectra_manipulation as sm
from rats.utilities import progress_tracker, default_logger_format
import astropy.units as u
import pickle
import logging

# %% Setup logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

#%%
@progress_tracker
def rassine_normalization_wrapper(spectrum_list: sp.SpectrumList,
                    ) -> sp.SpectrumList:
    
    RASSINE_save_directory = os.getcwd() + '/saved_data/RASSINE/'
    os.makedirs(RASSINE_save_directory,
                mode = 0o777,
                exist_ok = True)

    new_spectrum_list = sp.SpectrumList()
    
    for ind in np.unique(np.asarray([spectrum.meta['Night_num'] for spectrum in spectrum_list])):
        sublist = sm.get_sublist(spectrum_list, 'Night_num', ind, mode='equal')

        night_list = _prepare_single_night(sublist,
                              RASSINE_save_directory,
                              sublist[0].meta['Night'])
        
        new_spectrum_list.extend(night_list)
    return new_spectrum_list


def _prepare_single_spectrum(spectrum: sp.Spectrum1D | sp.SpectrumCollection,
                             RASSINE_night_save_directory: str,
                             ):
    """
    Prepares a single spectrum for RASSINE in csv file.

    Parameters
    ----------
    spectrum : sp.Spectrum1D | sp.SpectrumCollection
        Spectrum to save in csv file. For now, only Spectrum1D object works.
    RASSINE_night_save_directory : str
        Save directory for given night.

    Raises
    ------
    NotImplementedError
        For SpectrumCollection, this function is not implemented yet.
    """
    
    if type(spectrum) == sp.SpectrumCollection:
        raise NotImplementedError('Spectrum Collection is yet unsupported for RASSINE normalization')
    
    spectrum_number = str(spectrum.meta['Spec_num'])
    filename = RASSINE_night_save_directory + 'before_normalization_' + spectrum_number + '.csv'
    
    data_spectrum = pd.DataFrame(
        data= {
            'pixel': np.arange(len(spectrum.spectral_axis)),
            'wave': spectrum.spectral_axis.to(u.AA).value,
            'flux': spectrum.flux.value,
        }
    )
    data_spectrum.to_csv(filename)
    return filename

def _prepare_single_night(spectrum_list: sp.SpectrumList,
                          RASSINE_save_directory: str,
                          night: str) -> str:
    """
    Prepares for running RASSINE on single night dataset. 
    
    This creates a list of csv files containing the spectra. Furthermore, creates a separate directory for each night. Finally, it modifies the RASSINE_trigger.py file for specific night.

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list to run RASSINE on.
    RASSINE_save_directory : str
        Save directory where to save data
    night : str
        Night on which RASSINE is run on

    Returns
    -------
    str
        Directory where the data is saved by RASSINE
    """
    RASSINE_night_save_directory = RASSINE_save_directory + night + '/'
    os.makedirs(RASSINE_night_save_directory,
            mode = 0o777,
            exist_ok = True)
    
    first_spectrum = spectrum_list[0]
    
    night_list = sp.SpectrumList()
    
    for spectrum in spectrum_list:
        filename = _prepare_single_spectrum(spectrum= spectrum,
                                            RASSINE_night_save_directory= RASSINE_night_save_directory)
        
        output_filename = filename.replace('before', 'RASSINE_before').replace('.csv', '.p')
        
        if os.path.isfile(output_filename): # Don't rerun RASSINE
            with open(output_filename, 'rb') as input_file:
                RASSINE_output = pickle.load(input_file)
            night_list.append(_normalize_with_RASSINE_output(spectrum, RASSINE_output))
            continue
        
        # Adapt trigger file for RASSINE for given instrument
        with FileInput(RASSINE_config,
                    inplace=True,
                    backup='.bak' # The last configuration is saved in backup file
                    ) as file: 
            
            for line in file:
                if line.startswith('spectrum_name'):
                    print(line.replace(line, f"spectrum_name = '{filename}'"), end='\n')
                elif line.startswith('output_dir'):
                    print(line.replace(line, f"output_dir = '{RASSINE_night_save_directory}'"), end='\n')
                # elif line.startswith('dlambda'):
                #     print(line.replace(line, f"dlambda = {dlambda}"), end='\n')
                else:
                    print(line, end='')
        
        # Running RASSINE through terminal
        cwd = os.getcwd()
        os.chdir(RASSINE_location)
        logger.info(f'Running RASSINE on file {filename}')
        subprocess.call(RASSINE_command, shell=True)
        logger.info(f'RASSINE finished, file saved in {output_filename}')
        os.chdir(cwd)
        
        with open(output_filename, 'rb') as input_file:
            RASSINE_output = pickle.load(input_file)
        night_list.append(_normalize_with_RASSINE_output(spectrum, RASSINE_output))
        
        
    return night_list

def _normalize_with_RASSINE_output(spectrum: sp.Spectrum1D,
                                   RASSINE_output: dict):
    
    continuum = sp.Spectrum1D(
        spectral_axis= RASSINE_output['wave'] * u.AA,
        flux= RASSINE_output['output']['continuum_cubic'] * spectrum.flux.unit
    )
    
    normalized_spectrum = spectrum.divide(continuum, handle_meta='first_found')
    normalized_spectrum = normalized_spectrum.divide(np.nanmedian(normalized_spectrum.flux), handle_meta='first_found')
    
    normalized_spectrum.meta['normalization'] = True
    
    return normalized_spectrum
