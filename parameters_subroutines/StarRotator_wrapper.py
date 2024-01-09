from rats.StarRotator.StarRotator import StarRotator
import matplotlib.pyplot as plt
import numpy as np
from rats.parameters import SystemParametersComposite
from rats.utilities import default_logger_format
import logging
import astropy.units as u
import os


# Setup logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)


def run_StarRotator_pySME(SystemParameters: SystemParametersComposite,
                          linelist: str,
                          number_of_exposures: int = 40,
                          ) -> StarRotator:
    
    t1 = -SystemParameters.Ephemeris.transit_length_full.convert_unit_to(u.d).data / SystemParameters.Ephemeris.period.data / 2
    t4 = -t1
    input_dictionary = {
        'veq': SystemParameters.Star.vsini.data,
        'stelinc': 90.0,
        'drr': 0.0,
        'T': SystemParameters.Star.temperature.data,
        'FeH': SystemParameters.Star.metallicity.data,
        'logg': SystemParameters.Star.logg.data,
        'u1': 0.43180061,# FIXME
        'u2': 0.27686728, # FIXME
        'R': 140000, # FIXME
        'mus': 10,
        'model': 'pySME',
        'sma_Rs': SystemParameters.Planet.a_rs_ratio.data,
        'e': SystemParameters.Planet.eccentricity.data,
        'omega': SystemParameters.Planet.argument_of_periastron.data,
        'inclination': SystemParameters.Planet.inclination.data,
        'obliquity': SystemParameters.Planet.projected_obliquity.data,
        'RpRs': SystemParameters.Planet.rprs.data,
        'P': SystemParameters.Ephemeris.period.data,
        'phases': np.linspace(t1, t4, number_of_exposures),
        'grid_model': '',
        'abund': {},
        'linelist_path': linelist
        }
    
    input_dictionary = _correct_NaNs(input_dictionary)
    RM_model = StarRotator(586.0,592.0,200.0,input=input_dictionary)
    
    return RM_model
def _correct_NaNs(input_dictionary: dict) -> dict:
    
    if np.isnan(input_dictionary['e']):
        input_dictionary['e'] = 0
    if np.isnan(input_dictionary['omega']):
        input_dictionary['omega'] = 0
    
    return input_dictionary

def _get_LD_coefficients():
    
    
    return

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
                                     number_of_exposures= 15)
    
    logger.info('Test succesful')