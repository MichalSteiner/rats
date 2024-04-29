# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 02:51:24 2021

@author: Chamaeleontis

Functions to create and export various tables

"""

#%% Import libraries
import latextable
import texttable
import specutils as sp
import numpy as np

#%% Observation log of spectra
def observation_log(spectrum_list: sp.SpectrumList):
    """
    Prints out latex table output of observation log for given dataset.
    
    This can be copy-pasted in to paper draft. The columns for Seeing, Airmass and S/N are done in min-median-max format. S/N are taken from the 'Average_S_N' keyword, which is by default average over all orders. 

    Parameters
    ----------
    spectrum_list : sp.SpectrumList
        Spectrum list 
    """
    # Array allocation
    exposure_time = np.asarray([spectrum.meta['Exptime'].value for spectrum in spectrum_list])
    seeing = np.asarray([spectrum.meta['Seeing'] for spectrum in spectrum_list])
    airmass = np.asarray([spectrum.meta['Airmass'] for spectrum in spectrum_list])
    snr = np.asarray([spectrum.meta['Average_S_N'] for spectrum in spectrum_list])
    Night_num = np.asarray([spectrum.meta['Night_num'] for spectrum in spectrum_list])
    Nights = np.asarray([spectrum.meta['Night'] for spectrum in spectrum_list])
    Instruments = np.asarray([spectrum.meta['instrument'] for spectrum in spectrum_list])
    
    night_list, indices = np.unique(np.asarray(Night_num), return_index=True)
    # Create table
    Observation_log_table = texttable.Texttable()
    Observation_log_table.set_cols_align(['l','l','l','r','r','r','r'])
    # Create header
    table_rows = [["Night", "Night \#", "Instrument", 'Exposure time [s]', 'Seeing ["]', 'Airmass','S/N']]
    # Loop over all nights and extract values for given row
    for ind, night_num in enumerate(night_list):
        night = Nights[indices[night_num-1]]
        mean_exposure_time = round(np.nanmean(exposure_time[Nights==night]),-1)
        
        min_seeing, median_seeing, max_seeing = (round(np.nanmin(seeing[Nights==night]), 2),
                                                 round(np.nanmedian(seeing[Nights==night]), 2),
                                                 round(np.nanmax(seeing[Nights==night]), 2))
        
        min_airmass, median_airmass, max_airmass = (round(np.nanmin(airmass[Nights==night]), 2),
                                                    round(np.nanmedian(airmass[Nights==night]), 2),
                                                    round(np.nanmax(airmass[Nights==night]), 2))
        
        min_sn, median_sn, max_sn = (round(np.nanmin(snr[Nights==night]), 0),
                                     round(np.nanmedian(snr[Nights==night]), 0),
                                     round(np.nanmax(snr[Nights==night]), 0))

        night_number = Night_num[indices[ind]]
        instrument = Instruments[indices[ind]]
        # Append a row for iterated night properly formated
        table_rows.append(
            [
            str(night),
            str(night_number),
            str(instrument),
            str(mean_exposure_time),
            ('{:.2f}'.format(min_seeing) + ' - ' + '{:.2f}'.format(median_seeing) + ' - ' + '{:.2f}'.format(max_seeing)),
            ('{:.2f}'.format(min_airmass)+' - ' + '{:.2f}'.format(median_airmass) + ' - ' + '{:.2f}'.format(max_airmass)),
            ('{:.0f}'.format(min_sn) + ' - ' + '{:.0f}'.format(median_sn) + ' - ' + '{:.0f}'.format(max_sn))
            ]
        )
    # Add all rows to table and print the output
    Observation_log_table.add_rows(table_rows)
    print(latextable.draw_latex(Observation_log_table,
                                caption="Observation log table. The columns from left to right are: Date of observing night, the number of night, used instrument, seeing, airmass, average S/N. Note that for seeing airmass and average S/N, we provide the value in min-median-max format. S/N is average over all orders for given instrument.",
                                label="tab:observation_log",
                                use_booktabs=True)
          )
    return