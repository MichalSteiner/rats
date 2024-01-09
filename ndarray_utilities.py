import logging
from rats.utilities import default_logger_format
from astropy.nddata import NDDataArray, StdDevUncertainty
import astropy.units as u
from uncertainties import ufloat
import numpy as np

#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

def _round_value_more_than_one(value: float,
                               uncertainty: float) -> [float, float]:
    """
    Rounds the value and uncertainty to 2-nonzero digit precision. This version works for numbers above 1.

    Parameters
    ----------
    value : float
        Value which should be cut.
    uncertainty : float
        Uncertainty based on which the value will be cut. The uncertainty will also be rounded

    Returns
    -------
    [float, float]
        Rounded value and uncertainty to two non-zero digit
    """
    ind = 1
    position = 1
    while divmod(uncertainty, ind)[0] != 0:
        ind *= 10
        position -= 1
    position += 1
    
    return round(value, position), round(uncertainty, position)

def _round_value_less_than_one(value: float,
                               uncertainty:float) -> [float, float]:
    """
    Rounds the value and uncertainty to 2-nonzero digit precision. This version works for numbers below 1.

    Parameters
    ----------
    value : float
        Value which should be cut.
    uncertainty : float
        Uncertainty based on which the value will be cut. The uncertainty will also be rounded

    Returns
    -------
    [float, float]
        Rounded value and uncertainty to two non-zero digit
    """
    ind = 1
    position = 1
    while divmod(uncertainty, ind)[0] == 0:
        ind /= 10
        position += 1
    
    return round(value, position), round(uncertainty, position)

#%% Cut array values to precision
def _round_to_precision(array: NDDataArray) -> NDDataArray:
    """
    Rounds the NDDataArray to 2-nonzero digit precision.

    Parameters
    ----------
    array : NDDataArray
        Array which to cut.

    Returns
    -------
    new_array : NDDataArray
        Modified array.
    """
    value = array.data
    uncertainty = array.uncertainty.array
    
    if np.isnan(value) or np.isnan(uncertainty): 
        logger.debug('Rounding to precision failed due to NaN values.')
        return array
    
    logger.debug('Rounding values to two non-zero digits:')
    logger.debug('Original values:')
    logger.debug(f'    {value}')
    logger.debug(f'    {uncertainty}')
    
    if divmod(uncertainty,1)[0] != 0:
        # More than 1
        value, uncertainty = _round_value_more_than_one(value, uncertainty)
    else:
        # Less than 1
        value, uncertainty = _round_value_less_than_one(value, uncertainty)
    logger.debug('New values:')
    logger.debug(f'    {value}')
    logger.debug(f'    {uncertainty}')
    new_array = NDDataArray(
        data = value,
        uncertainty = StdDevUncertainty(uncertainty),
        meta= array.meta,
        unit = array.unit
    )
    return new_array

#%% NDDataArray**(power)
def _power_NDDataArray(array: NDDataArray,
                       power: float) -> NDDataArray:
    """
    Workaround for using powers on NDDataArray class.
    
    Uses propagation of uncertainties from the uncertainties package.

    Parameters
    ----------
    array : NDDataArray
        Array which to put to power.
    power : float
        Power to which to raise array to.

    Returns
    -------
    new_array : NDDataArray:
        Array raised to given power.
    """
    
    unit = array.unit
    if unit is None:
        unit = u.dimensionless_unscaled
    array = ufloat(array.data, array.uncertainty.array)
    new_array = array**(power)
    new_array = NDDataArray(data= new_array.n, uncertainty= StdDevUncertainty(new_array.s), unit = unit**power)
    
    return new_array

#%% sin(NDDataArray)
def _sin_NDDataArray(array: NDDataArray) -> NDDataArray:
    """
    Workaround for using sin on NDDataArray class. Using np.sin on NDDataArray will reduce it to just a standard numpy array, with no uncertainty being propagated.

    Parameters
    ----------
    array : NDDataArray
        Array for which we want to calculate sin.

    Returns
    -------
    new_array : NDDataArray
        sin(array).
    """
    from uncertainties.umath import sin
    array = array.convert_unit_to(u.rad)
    array = ufloat(array.data, array.uncertainty.array)
    new_array = sin(array)
    new_array = NDDataArray(data= new_array.n, uncertainty= StdDevUncertainty(new_array.s))
    return new_array
#%% cos(NDDataArray)
def _cos_NDDataArray(array: NDDataArray) -> NDDataArray:
    """
    Workaround for using cos on NDDataArray class. Using np.cos on NDDataArray will reduce it to just a standard numpy array, with no uncertainty being propagated.

    Parameters
    ----------
    array : NDDataArray
        Array for which we want to calculate cos.

    Returns
    -------
    new_array : NDDataArray
        cos(array).
    """
    from uncertainties.umath import cos
    array = array.convert_unit_to(u.rad)
    array = ufloat(array.data, array.uncertainty.array)
    new_array = cos(array)
    new_array = NDDataArray(data= new_array.n, uncertainty= StdDevUncertainty(new_array.s))
    return new_array

#%% tan(NDDataArray)
def _tan_NDDataArray(array: NDDataArray) -> NDDataArray:
    """
    Workaround for using tan on NDDataArray class. Using np.tan on NDDataArray will reduce it to just a standard numpy array, with no uncertainty being propagated.

    Parameters
    ----------
    array : NDDataArray
        Array for which we want to calculate tan.

    Returns
    -------
    new_array : NDDataArray
        tan(array).
    """
    from uncertainties.umath import tan
    array = array.convert_unit_to(u.rad)
    array = ufloat(array.data, array.uncertainty.array)
    new_array = tan(array)
    new_array = NDDataArray(data= new_array.n, uncertainty= StdDevUncertainty(new_array.s))
    return new_array


#%% arcsin(NDDataArray)
def _arcsin_NDDataArray(array: NDDataArray) -> NDDataArray:
    """
    Workaround for using arcsin on NDDataArray class.

    Parameters
    ----------
    array : NDDataArray
        Array on which to apply the arcsin function.

    Returns
    -------
    new_array : NDDataArray
        Result of arcsin(array).
    """
    from uncertainties.umath import asin
    array = array.convert_unit_to(u.dimensionless_unscaled)
    array = ufloat(array.data, array.uncertainty.array)
    new_array = asin(array)
    new_array = NDDataArray(data= new_array.n, uncertainty= StdDevUncertainty(new_array.s), unit=u.rad)
    
    return new_array

#%% arctan(NDDataArray)
def _arctan_NDDataArray(array: NDDataArray) -> NDDataArray:
    """
    Workaround for using arctan on NDDataArray class.

    Parameters
    ----------
    array : NDDataArray
        Array on which to apply the arctan function.

    Returns
    -------
    new_array : NDDataArray
        Result of arctan(array).
    """
    from uncertainties.umath import atan
    array = array.convert_unit_to(u.dimensionless_unscaled)
    array = ufloat(array.data, array.uncertainty.array)
    new_array = atan(array)
    new_array = NDDataArray(data= new_array.n, uncertainty= StdDevUncertainty(new_array.s), unit=u.rad)
    
    return new_array
