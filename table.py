# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 02:51:24 2021

@author: Chamaeleontis

Functions to create and export various tables

"""

#%% Import libraries
import pandas as pd
import numpy as np
import os

#%% table_obs_log
def table_obs_log(obs_log,orders):
    '''
    Creates table of observational log given observations_log class and orders
    Input:
        obs_log ; observations_log
        orders ; list of numbers
    Output:
        table ; latex table - also saved in tables directory
    
    '''
    num_nights = len(obs_log.night)
    
    table = pd.DataFrame({'Night':[],'#':[],'Spec_num':[],'Airmass':[],'Seeing':[]})
    if len(orders)!=0:
        for order in orders:
            table['SN%i'%(order)] = []
    else:
        table['Median S/N'] = []
    meta_all = []
    for ii in range(num_nights):
        SN = {}
        if len(orders)!=0:
            for ind,order in enumerate(obs_log.night[ii].order_list):
                SN.update({'SN%i'%(order):obs_log.night[ii].SN[order]})
            dict_meta = {
                'Night': obs_log.night[ii].night,
                '#': obs_log.night[ii].night_num,
                'Spec_num':str(obs_log.night[ii].exp_num) + '(' + str(obs_log.night[ii].in_num) + ')',
                'Airmass':obs_log.night[ii].Airmass,
                'Seeing':obs_log.night[ii].Seeing,
                'Exposure time [s]':str("{:.0f}".format(round(np.mean(obs_log.night[ii].Exptime),-1)))
                }
        else:
            dict_meta = {
                'Night': obs_log.night[ii].night,
                '#': obs_log.night[ii].night_num,
                'Spec_num':str(obs_log.night[ii].exp_num) + '(' + str(obs_log.night[ii].in_num) + ')',
                'Airmass':obs_log.night[ii].Airmass,
                'Seeing':obs_log.night[ii].Seeing,
                'Exposure time [s]':str("{:.0f}".format(round(np.mean(obs_log.night[ii].Exptime),-1)))
                }
            SN.update({'80% quantile S/N': obs_log.night[ii].SN['Median']})
            pass
        dict_meta.update(SN)
        meta_all.append(dict_meta)
    table = pd.DataFrame(meta_all,)
    path =  os.getcwd() + '/tables/observational_log.tex'
    caption = 'Observational log: The columns corresponds (from left to right) to: Night of observation; Reference number of night; Number of spectra during night and number of in transit spectra in parentheses; Airmass; Seeing; Exptime and Signal-to-Noise ratio (SNR) for each order. Note that for the Airmass, Seeing and SNR columns the format is min-mean-max during the night.'
    label = 'tab_obs_log'
    
    latex_table = table.to_latex(caption = caption, label = label,index=False)
    
    with open(path,'w') as tf:
        tf.write(latex_table)
    return table
