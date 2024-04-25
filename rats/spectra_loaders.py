import astropy.io.fits as fits
import specutils as sp
import numpy as np
from astropy.table import Table
import os
from copy import deepcopy
from specutils.io.registers import custom_writer
import logging
from rats.utilities import default_logger_format


#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)


#%%
@custom_writer("rats-writer")
def rats_fits_writer(
    spectrum: sp.Spectrum1D,
    file_name: str,
    ):
    """
    Writes a spectrum into a fits file using a simple custom format.
    
    It is possible to use the "rats_writer" format for the sp.Spectrum1D.write() | sp.SpectrumCollection.write() function
    The output format has a total of 6 HDUs (1 Primary + 5 ImageHDU):
    PrimaryHDU:
        Empty HDU needed for initialization of the fits file.
    InstrumentHeader:
        HDU holding the header of the instrument.
    RATSHeader:
        HDU holding meta parameters introduced by RATS pipeline.
    Spectral_axis:
        HDU holding the wavelength information.
    Flux:
        HDU holding the flux information.
    Uncertainty:
        HDU holding the uncertainty information.
    
    Parameters
    ----------
    spectrum : sp.Spectrum1D
        Spectrum to save
    file_name : str
        Filename where to save the data.
    """
    # TODO: Branch for 
    logger.info(f'Writing spectrum into: {file_name}')
    
    instrument= spectrum.meta['instrument']
    
    hdu_primary_header = fits.PrimaryHDU()
    hdu_instrument_header = fits.ImageHDU(data= np.arange(5)+1,
                                          header= spectrum.meta['header'])
    meta_copied = deepcopy(spectrum.meta)
    meta_copied.pop('header')
    hdu_rats_header = _create_BINTABLE_from_dictionary(meta_copied)
    
    # TODO: Implement units handling
    hdu_spectral_axis = fits.ImageHDU(spectrum.spectral_axis.value, name= 'Spectral axis', unit=spectrum.spectral_axis.unit)
    hdu_flux = fits.ImageHDU(spectrum.flux.value, name= 'Flux', unit=spectrum.flux.unit)
    hdu_uncertainty = fits.ImageHDU(spectrum.uncertainty.array, name= 'Uncertainty')
    
    hdu_instrument_header.header = spectrum.meta['header']
    modified_meta = deepcopy(spectrum.meta)
    _ = modified_meta.pop('header', None)
    hdu_rats_header.header = modified_meta
    
    # FIXME: For some reason, the INSTRUMENT header HDU is named the same as the RATS HDU.
    hdu_instrument_header.name = 'Instrument meta'
    hdu_rats_header.name = 'RATS meta'

    # Fix the non-FITS compliant parts
    hdu_instrument_header.verify('fix')
    hdu_rats_header.verify('fix')
    
    hdu_list = fits.HDUList([
        hdu_primary_header,
        # hdu_instrument_header,
        # hdu_rats_header,
        hdu_spectral_axis,
        hdu_flux,
        hdu_uncertainty]
        )
    hdu_list.writeto(file_name)
    
    logger.info(f'Writing completed')
    return

def rats_fits_loader(fits_filename: str) -> sp.Spectrum1D | sp.SpectrumCollection:
    
    hdu_list = fits.open(fits_filename)
    
    # instrument_header = hdu_list[1].header
    # rats_header = hdu_list[2].header
    spectral_axis = hdu_list['SPECTRAL AXIS'].data
    flux = hdu_list['FLUX'].data
    uncertainty = hdu_list['UNCERTAINTY'].data
    

    
    return

def rats_fits_writer_list(spectrum_list: sp.SpectrumList,
                          directory_name: str,
                          custom_filename: str = '',
                          ):
    os.makedirs(directory_name,
                mode = 0o777,
                exist_ok = True)
    
    logger.info(f'About to write spectrum list into {directory_name}')
    for ind, spectrum in enumerate(spectrum_list):
        spectrum.write(directory_name + '/' + custom_filename + f'_{ind}',
                       format= 'rats-writer'
                       )
    logger.info('Writing completed')
    return

def rats_fits_loader_list(fits_directory_name: str):
    raise NotImplementedError
    return

def _create_BINTABLE_from_dictionary(dictionary: dict):
    
    # Your dictionary with objects of different sizes

    # Create a list of FITS column definitions
    columns = []
    for key, value in dictionary.items():
        # Convert each value to a numpy array if it's not already
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        # Determine the format based on the data type
        if np.issubdtype(value.dtype, np.integer):
            column_format = 'J'  # 32-bit signed integer
        elif np.issubdtype(value.dtype, np.floating):
            column_format = 'E'  # 32-bit floating point
        elif np.issubdtype(value.dtype, np.bool_):
            column_format = 'L'  # Boolean
        elif np.issubdtype(value.dtype, np.character):
            if np.isscalar(value):
                column_format = 'A10'  # Use a fixed length for scalar strings
            else:
                column_format = 'A{}'.format(value.itemsize)  # variable-length string
        else:
            raise ValueError(f"Unsupported data type for key '{key}'")
        
        # Create a FITS column definition for each key-value pair
        column = fits.Column(name=key, format=column_format, array=value)
        columns.append(column)
    
    return fits.BinTableHDU.from_columns(columns)

@custom_writer("rats-writer-template")
def _rats_writer_template(
    spectrum: sp.Spectrum1D,
    file_name: str
    ):
    raise NotImplementedError
    return

def _rats_loader_template():
    raise NotImplementedError
    
    return