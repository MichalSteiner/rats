from rats.StarRotator.StarRotator import StarRotator
import matplotlib.pyplot as plt
import numpy as np
from rats.parameters import SystemParametersComposite
from rats.utilities import default_logger_format, save_and_load, progress_tracker
import logging
import astropy.units as u
import os
import pysme
import specutils as sp
# Setup logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

@progress_tracker
@save_and_load
def run_StarRotator_pySME(SystemParameters: SystemParametersComposite,
                          linelist: str,
                          exposures: int | sp.SpectrumList = 40,
                          abundances: dict = {},
                          force_load: bool = False,
                          force_skip: bool = False,
                          pkl_name: str = 'RM_model.pkl'
                          ) -> StarRotator:
    """
    StarRotator wrapper that uses pySME for master out spectra generation corrected for RM+CLV effect. This wrapper simplifies the input for the StarRotator, however, it limits the usable options.

    Parameters
    ----------
    SystemParameters : SystemParametersComposite
        System parameters under which the model needs to be calculated
    linelist : str
        Location of the file with line list, as obtained through the VALD database.
    exposures : int | sp.SpectrumList, optional
        Exposures for which to consider the model, by default 40. If integer, it refers to number of spectra to be modelled linearly spaced in phase. If SpectrumList, the entire list of spectra will be modeled, including out-of-transit data.
    force_load : bool, optional
        Whether to load the output, by default False. If true, the 'pkl_name' will be opened and loaded, and the function will be skipped.
    force_skip : bool, optional
        Whether to skip the function completely, by default False. If true, the function will not be run and no output will be provided.
    pkl_name : str, optional
        Name of the pickle file, by default 'RM_model.pkl'

    Returns
    -------
    StarRotator
        StarRotator object, which holds all information from the modelling. Refer to StarRotator documentation to see more details.
    """

    if type(exposures) == sp.SpectrumList:
        phases = [spectrum.meta['Phase'].data for spectrum in exposures]
        start_wlg, end_wlg = exposures[0].spectral_axis[0].to(u.nm).value, exposures[0].spectral_axis[-1].to(u.nm).value #type: ignore
    else:
        t1 = -SystemParameters.Ephemeris.transit_length_partial.convert_unit_to(u.d).data / SystemParameters.Ephemeris.period.data / 2 #type: ignore
        t4 = -t1
        phases = np.linspace(t1, t4, exposures) #type: ignore
        start_wlg, end_wlg = 586.0,592.0
    
    input_dictionary = {
        'veq': SystemParameters.Star.vsini.data, #type: ignore
        'stelinc': 90.0,
        'drr': 0.0,
        'T': SystemParameters.Star.temperature.data, #type: ignore
        'FeH': SystemParameters.Star.metallicity.data, #type: ignore
        'logg': SystemParameters.Star.logg.data, #type: ignore
        'u1': SystemParameters.Star.LimbDarkening_u1, #type: ignore
        'u2': SystemParameters.Star.LimbDarkening_u2, #type: ignore
        'R': 140000, # FIXME
        'mus': 10,
        'model': 'pySME',
        'sma_Rs': SystemParameters.Planet.a_rs_ratio.data, #type: ignore
        'e': SystemParameters.Planet.eccentricity.data, #type: ignore
        'omega': SystemParameters.Planet.argument_of_periastron.data, #type: ignore
        'inclination': SystemParameters.Planet.inclination.data, #type: ignore
        'obliquity': SystemParameters.Planet.projected_obliquity.data, #type: ignore
        'RpRs': SystemParameters.Planet.rprs.data, #type: ignore
        'P': SystemParameters.Ephemeris.period.data, #type: ignore
        'phases': phases,
        'grid_model': '',
        'abund': abundances,
        'linelist_path': linelist
        }
    input_dictionary = _correct_NaNs(input_dictionary)
    RM_model = StarRotator(start_wlg,end_wlg,200.0,input=input_dictionary)
    return RM_model


def _correct_NaNs(input_dictionary: dict) -> dict:
    """
    Correct NaN numbers for eccentricity and argument of periastron, replacing them with default value.

    Parameters
    ----------
    input_dictionary : dict
        _description_

    Returns
    -------
    dict
        _description_
    """
    if np.isnan(input_dictionary['e']):
        input_dictionary['e'] = 0
    if np.isnan(input_dictionary['omega']):
        input_dictionary['omega'] = 90
    
    return input_dictionary

if __name__ == '__main__':
    
    os.chdir('/media/chamaeleontis/Observatory_main/Analysis_dataset/rats_test')
    logger.info('Starting test of rats.modeling_CCF:')
    logger.info('Loading parameters of KELT-10 b system')
    
    # Testing for KELT-10 b system
    KELT10b = SystemParametersComposite(
        filename= os.getcwd() + '/saved_data/system_parameters.pkl'        
        )
    KELT10b.load_NASA_CompositeTable_values(planet_name = 'KELT-10 b',
                                            force_load= True)
    KELT10b.print_main_values()
    
    # TODO: Fix the related function to work as intended
    KELT10b.Ephemeris.transit_length_partial = KELT10b.EphemerisPlanet.Ephemeris_list[1].transit_length_partial
    KELT10b.Ephemeris.transit_length_full = KELT10b.Ephemeris.transit_length_partial
    from astropy.nddata import StdDevUncertainty, NDDataArray
    KELT10b.Planet.projected_obliquity = NDDataArray(data= -5.2, # -5.2
                                           uncertainty= StdDevUncertainty(3.4))
    KELT10b.Star.vsini = NDDataArray(data= 2.58,
                                     uncertainty= StdDevUncertainty(0.12))
    
    linelist = "/media/chamaeleontis/Observatory_main/Code/rats/LINELIST_VALD/MichalSteiner.015070"
    RM_model = run_StarRotator_pySME(SystemParameters= KELT10b,
                                     linelist= linelist,
                                     exposures= 15)
    
    logger.info('Test succesful')