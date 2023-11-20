#%% Importing libraries
import astropy.constants as con
import astropy.units as u
import logging
from rats.utilities import default_logger_format

#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)




#%%
class CalculationPlanet:
    
    def _calculate_gravity_acceleration(self):
        logger.info('Calculation of gravity acceleration:')
        self.gravity_acceleration = (con.G * self.mass / (self.radius.multiply(self.radius))).decompose()
        logger.info(f'    {self.gravity_acceleration}')
        return
    
    def _calculate_atmospheric_scale_height(self):
        try:
            self.gravity_acceleration
        except:
            self._calculate_gravity_acceleration
        
        logger.info('Calculation of atmospheric scale height:')
        H = k * self.equilibrium_temperature / mu /self.gravity_acceleration
        logger.info('')
        
        return
    
    def _calculate_TSM(self):
        logger.info('Calculation of gravity acceleration:')
        
        logger.info('')
        
        return
    
    