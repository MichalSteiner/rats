

#%% Importing libraries
import numpy as np
import specutils as sp
import rats.parameters as para


#%%
def _assert_first_order_rm_clv_effect_conditions(system_parameters: para.SystemParametersComposite):
    """
    Assert that all the necessary system parameters are included in the first order RM+CLV estimate.

    Parameters
    ----------
    system_parameters : para.SystemParametersComposite
        System parameters for which the RM and CLV effect is estimated.
    """
    
    assert system_parameters.Star.vsini is not None, 'The v sin i of the host star is not defined'
    assert system_parameters.Planet.projected_obliquity is not None, 'The projected obliquity is not defined'
    assert system_parameters.Ephemeris.transit_depth is not None, 'The transit depth is not defined'
    
    return

#%% first_order_rm_clv_effect
def first_order_rm_clv_effect(master_out_list: sp.SpectrumList,
                              spectrum_list: sp.SpectrumList,
                              system_parameters: para.SystemParametersComposite
                              ):
    
    system_parameters.Ephemeris.transit_depth
    
    
    return
#%% second_order_rm_clv_effect
def second_order_rm_clv_effect():
    return

