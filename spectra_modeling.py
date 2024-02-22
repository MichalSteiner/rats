#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 09:20:55 2022

@author: chamaeleontis
"""
#%% Import libraries
import rats.spectra_manipulation as sm
import itertools
import multiprocessing
import matplotlib as mpl
import specutils as sp
import astropy.units as u
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy
import rats.plot_spectra as ps
import rats.plot_functions as pf
import astropy.constants as con
import rats.orbital_functions as orb
import seaborn as sns
# %% Plot settings
color_pallete_dark = sns.color_palette("dark")
color_pallete_pastel = sns.color_palette("pastel")
color_pallete_muted = sns.color_palette("muted")
color_pallete_deep = sns.color_palette("deep")
color_pallete_bright = sns.color_palette("bright")
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_pallete_dark)
mpl.rcParams["figure.autolayout"] = True
mpl.rcParams['lines.linewidth'] = 2
# Fullscreen for 1920x1080 resolution with default DPI settings
plt.rcParams["figure.figsize"] = 18.52, 11.24
mpl.rcParams['axes.titlesize'] = 36
mpl.rcParams['axes.labelsize'] = 36
mpl.rcParams['xtick.labelsize'] = 36
mpl.rcParams['ytick.labelsize'] = 36
mpl.rcParams['legend.fontsize'] = 24
mpl.rcParams['xtick.major.size'] = 36
mpl.rcParams['ytick.major.pad'] = 36
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
#%%
'''Shifting air to vac and vac to air wavelength'''
def airtovac(wlnm):
    wlA=wlnm*10.0
    s = 1e4 / wlA
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return(wlA*n/10.0)

def vactoair(wlnm):
    wlA = wlnm*10.0
    s = 1e4/wlA
    f = 1.0 + 5.792105e-2/(238.0185e0 - s**2) + 1.67917e-3/( 57.362e0 - s**2)
    return(wlA/f/10.0)

#%%
class StellarModel:
    
    def __loadPHOENIXmodel__(self,filename='/media/chamaeleontis/Observatory_main/KELT-10/data/stellar_model/lte05900-4.50+0.5.7'):
        '''
        Loads the PHOENIX stellar model

        Parameters
        ----------
        filename : str
            Where to find the stellar model file.

        Returns
        -------
        None.

        '''
        filename = '/media/chamaeleontis/Observatory_main/KELT-10/data/stellar_model/lte05900-4.50+0.5.7'
        f = open('/media/chamaeleontis/Observatory_main/KELT-10/data/stellar_model/lte05900-4.50+0.5.7', "r")
        
        all_lines = f.readlines()
        test = np.zeros((2,len(all_lines)))
        for ind,line in enumerate(all_lines):
            test[0,ind],test[1,ind] = line.split()[0].replace('D','E'),line.split()[1].replace('D','E')
        
        sorted_test = test[0][np.argsort(test[0])][450000:460000],test[1][np.argsort(test[0])][450000:460000] # Na doublet
        self.spectrum = sp.Spectrum1D(
            spectral_axis = vactoair(sorted_test[0])*u.AA,
            flux = 10**sorted_test[1]/np.mean(10**sorted_test[1])*u.dimensionless_unscaled,
            uncertainty = astropy.nddata.StdDevUncertainty(np.full_like(sorted_test[0], 0))
            )
        self.spectrum = sm.normalize_list([self.spectrum],quantile=.99,linfit=False)[0]
        
        return
    
    def input_array_values(self,nrows,ncols):
        if (np.sqrt((nrows-self.x0)**2+(ncols-self.y0)**2)<self.ncells/2):
            self.grid[nrows,ncols] = (
                                      (nrows-self.x0)/self.rs, #x
                                      (ncols-self.y0)/self.rs, #y 
                                      np.sqrt((nrows-self.x0)**2+(ncols-self.y0)**2)/self.rs, # r
                                      np.cos(np.arcsin(np.sqrt((nrows-self.x0)**2+(ncols-self.y0)**2)/(self.rs))), #mu
                                      True ,#in star
                                      
                                      np.cos(np.arcsin((ncols-self.y0)/(self.rs))) * \
                                      (nrows-self.x0)/(self.rs)*self.vsini, # local stellar velocity
                                      np.arcsin((ncols-self.x0)/(self.rs)), # longitude
                                      np.arcsin((nrows-self.y0)/(self.rs)), # latitude
                                      )
        else:
            self.grid[nrows,ncols] = (
                                      (nrows-self.x0)/self.rs, #x
                                      (ncols-self.y0)/self.rs, #y
                                      np.sqrt((nrows-self.x0)**2+(ncols-self.y0)**2)/self.rs, #r
                                      np.nan, # mu
                                      False , # in star
                                      np.nan, # local stellar velocity
                                      np.nan, # longitude
                                      np.nan, # latitude
                                      )
    

    
    def __init__(self,
                 ncells:int,
                 sys_para,
                 ):
        '''
        Constructor of stellar model
        Creates a grid (ncells x ncells) of spectra with spectral model for modeling of effects on transmission spectrum

        Parameters
        ----------
        ncells : int
            Number of cells in stellar grid across one dimension. The grid will then be of shape = (ncells,ncells)
        sys_para : rats.parameters.system_parameters_class
            Class with system parameters to simulate

        Returns
        -------
        None.

        '''
        
        
        
        self.x0,self.y0 = (ncells-1)/2,(ncells-1)/2
        self.rs = (ncells-1)/2
        self.ncells = ncells
        self.grid = np.empty((ncells,ncells),
                             dtype=[('x', 'float'),
                                    ('y', 'float'),
                                    ('r', 'float'),
                                    ('mu', 'float'),
                                    ('instar', 'bool'),
                                    ('v_stel_loc', 'float'),
                                    ('longitude', 'float'),
                                    ('latitude', 'float')
                                    ])
        self.rprs = (sys_para.planet.radius/sys_para.star.radius).decompose().value
        self.obliquity = sys_para.planet.obliquity.to(u.deg).value
        self.vsini = sys_para.planet.vsini.to(u.km/u.s).value
        self.impact_parameter = sys_para.system.b
        self.a = sys_para.system.a.to(u.au).value
        self.radius_star = (sys_para.star.radius).value
        self.rs_a = sys_para.system.rs_a_ratio
        
        for nrows in range(ncells):
            for ncols in range(ncells):
                self.input_array_values(nrows,ncols)
        return
    
    def plot_system_colormap(self):
        '''
        Plots the system transit across the range of stellar local velocities
        TODO:
            check the impact parameter is well implemented

        Returns
        -------
        None.

        '''
        cmap = mpl.cm.get_cmap("coolwarm").copy()
        cmap.set_bad(color='black')
        
        fig, ax = plt.subplots(1,figsize=(8,8))
        c = ax.pcolormesh(self.grid['x'],self.grid['y'],self.grid['v_stel_loc'],cmap=cmap)
        ax.axline((0,-self.impact_parameter),slope=np.tan(np.radians(self.obliquity)),color='darkgreen',ls='--')
        
        cbar = plt.colorbar(c)
        
        
        return
    
    def RM_effect_cell(self,
                       cell):
        '''
        Simulates the effect of Rossiter-McLaughlin on the cell

        Parameters
        ----------
        cell : self.grid[nrows,ncolls]
            Cell of the stellar grid.

        Returns
        -------
        spectrum_effect : sp.Spectrum1D
            Affected spectrum, shifted by stellar local velocity.

        '''
        spectrum_effect = sm._shift_spectrum(self.spectrum,
                                             [cell['v_stel_loc']*u.km/u.s])
        
        return spectrum_effect
        
    
    def RM_effect(self):
        '''
        Shifts the grid stellar spectra by given local stellar velocity

        Returns
        -------
        None.

        '''
        occulted_disk = self.occulted_disk
        
        for col_ind in range(self.ncells): # Rowindices
            print('row_ind:',col_ind)
            for row_ind in range(self.ncells): # Column indices
                if self.grid[row_ind,col_ind]['instar']: # Is the cell inside the star?
                    shifted_spectrum = sm._shift_spectrum(self.spectrum, [self.grid[row_ind,col_ind]['v_stel_loc']*u.km/u.s]) # Shift spectrum by local stellar velocity
                    for time_ind,occulted_time in enumerate(occulted_disk): # add the spectrum to all  
                        if occulted_time['inplan'][row_ind][col_ind]:
                            occulted_time['corrected_spectrum'] = occulted_time['corrected_spectrum'].add(shifted_spectrum)
                            occulted_time['uncorrected_spectrum'] = occulted_time['uncorrected_spectrum'].add(shifted_spectrum)
                            occulted_time['num_corr']+=1
                            occulted_time['num_unco']+=1
                        else:
                            occulted_time['uncorrected_spectrum'] = occulted_time['uncorrected_spectrum'].add(shifted_spectrum)
                            occulted_time['num_unco']+=1
                            
        for time_ind,occulted_time in enumerate(occulted_disk): # Normalize to 1
            occulted_time['corrected_spectrum'] = occulted_time['corrected_spectrum'].divide(
                sp.Spectrum1D(
                    spectral_axis = occulted_time['corrected_spectrum'].spectral_axis,
                    flux = np.full_like(occulted_time['corrected_spectrum'].spectral_axis.value,occulted_time['num_corr'])*u.dimensionless_unscaled,
                    uncertainty = astropy.nddata.StdDevUncertainty(np.full_like(occulted_time['corrected_spectrum'].spectral_axis.value, 0))
                    )
                )
            occulted_time['uncorrected_spectrum'] = occulted_time['uncorrected_spectrum'].divide(
                sp.Spectrum1D(
                    spectral_axis = occulted_time['uncorrected_spectrum'].spectral_axis,
                    flux = np.full_like(occulted_time['uncorrected_spectrum'].spectral_axis.value,occulted_time['num_unco']*u.dimensionless_unscaled)*u.dimensionless_unscaled,
                    uncertainty = astropy.nddata.StdDevUncertainty(np.full_like(occulted_time['corrected_spectrum'].spectral_axis.value, 0))
                    )
                )
        
        # correction_factor = corrected_spectrum.divide(corr_num) / uncorrected_spectrum.divide(unco_num)
        
        
        
        return 
    
    def CLV_effect(self):
        '''
        TODO:
            To be implemented

        Returns
        -------
        None.

        '''
        
        
        
        
        return
    
    def Calculate_effect_transit(self,
                                RM=True,
                                CLV=False
                                 ):
        '''
        Calculates effect on the transit and transmission spectrum

        Parameters
        ----------
        RM : bool, optional
            Correct for the RM effect. The default is True.
        CLV : bool, optional
            Correct for the CLV effect. The default is False.

        Returns
        -------
        None.

        '''
        print('Start of transit effect calculations')
        for row_ind in range(self.grid.shape[0]): # Iterate over x-values
            print('Calculate over column number: %i'%(row_ind),'/%i'%(self.grid.shape[0]))
            for col_ind in range(self.grid.shape[1]): # Iterate over y- values
                cell = self.grid[row_ind,col_ind]
                if cell['instar'] == True: # Check that the cell is in star
                    affected_spectrum = self.RM_effect_cell(cell) # Calculate affected spectrum by RM
                    print('Working on column %i'%(row_ind),' and row %i'%(col_ind),' with shift:', cell['v_stel_loc'])
                    for time_ind, time in enumerate(self.occulted_disk): # Calculate over all times of transit
                        if (RM == True) and (CLV == False):
                            if time['inplan'][row_ind,col_ind] == True:
                                
                                self.occulted_disk['corrected_spectrum'][time_ind] =                                  self.occulted_disk['corrected_spectrum'][time_ind].add(
                                    affected_spectrum
                                    )
                                self.occulted_disk['uncorrected_spectrum'][time_ind] =                                  self.occulted_disk['uncorrected_spectrum'][time_ind].add(
                                    affected_spectrum
                                    )
                                time['num_corr']+=1
                                time['num_unco']+=1
                            else:
                                self.occulted_disk['uncorrected_spectrum'][time_ind] =                                  self.occulted_disk['uncorrected_spectrum'][time_ind].add(
                                    affected_spectrum
                                    )
                                time['num_unco']+=1
        
        for time_ind, time in enumerate(self.occulted_disk): # Divide by number of spectra added
            self.occulted_disk['corrected_spectrum'][time_ind] = self.occulted_disk['corrected_spectrum'][time_ind].divide(sp.Spectrum1D(
                spectral_axis = time['corrected_spectrum'].spectral_axis,
                flux = np.full_like(time['corrected_spectrum'].spectral_axis.value,
                                    time['num_corr'])*u.dimensionless_unscaled,
                uncertainty = astropy.nddata.StdDevUncertainty(np.full_like(time['corrected_spectrum'].spectral_axis.value, 0)
                                                               )
                ))
            self.occulted_disk['uncorrected_spectrum'][time_ind] = self.occulted_disk['uncorrected_spectrum'][time_ind].divide(sp.Spectrum1D(
                spectral_axis = time['uncorrected_spectrum'].spectral_axis,
                flux = np.full_like(time['uncorrected_spectrum'].spectral_axis.value,
                                    time['num_unco'])*u.dimensionless_unscaled,
                uncertainty = astropy.nddata.StdDevUncertainty(np.full_like(time['uncorrected_spectrum'].spectral_axis.value, 0)
                                                               )
                ))
        
        return
    
    def Transit(self,
                phasing=100,
                ):
        '''
        Simulates transit over the stellar grid

        Returns
        -------
        None.

        '''
        t = np.linspace(-1-2*self.rprs,1+2*self.rprs,phasing)
        self.occulted_disk = np.empty(
            shape=(phasing),
            dtype=[
                ('x','float'),
                ('y','float'),
                ('inplan','object'),
                ('corrected_spectrum','object'),
                ('uncorrected_spectrum','object'),
                ('num_corr','float'),
                ('num_unco','float'),
                ('Phase','float')
                ]
            )
        for ind,value in enumerate(self.occulted_disk):
            self.occulted_disk[ind] = (
                t[ind],# X values
                self.impact_parameter + t[ind]*np.tan(np.radians(self.obliquity)), # Y values
                np.where(np.sqrt((t[ind]-self.grid['x'])**2+(self.impact_parameter + t[ind]*np.tan(np.radians(self.obliquity))-self.grid['y'])**2)<self.rprs,False,True), # Which indices are not occulted
                sp.Spectrum1D(
                    spectral_axis = self.spectrum.spectral_axis,
                    flux = np.full_like(self.spectrum.spectral_axis.value, 0) *u.dimensionless_unscaled,
                    uncertainty = astropy.nddata.StdDevUncertainty(np.full_like(self.spectrum.spectral_axis.value, 0))
                    
                    ),
                sp.Spectrum1D(
                    spectral_axis = self.spectrum.spectral_axis,
                    flux = np.full_like(self.spectrum.spectral_axis.value, 0) *u.dimensionless_unscaled,
                    uncertainty = astropy.nddata.StdDevUncertainty(np.full_like(self.spectrum.spectral_axis.value, 0))
                    ),
                0,
                0,
                (-np.arccos(t[ind]* self.radius_star*self.rs_a)+np.pi/2)/2/np.pi
                                  )
        return
    
    def Transmission_simulation(self,
                                RM=True,
                                CLV=False):
        '''
        Simulates transmission spectrum calculation without planetary atmosphere, to simulate the RM and CLV effects

        Returns
        -------
        None.

        '''
        spec_list = sp.SpectrumList()
        new_spec_list = sp.SpectrumList()
        for corrected, uncorrected, phase in zip(new_test.occulted_disk['corrected_spectrum'],new_test.occulted_disk['uncorrected_spectrum'],new_test.occulted_disk['Phase']):
            spec_list.append(uncorrected.divide(corrected))
            new_spec_list.append(new_test.spectrum.divide(corrected))
        for intransit in self.occulted_disk:
            if RM:
                spec_list.append(
                self.RM_effect(self,intransit['inplan'])
                )
            if CLV:
                self.CLV_effect(self,intransit['inplan'])
        
        return
#%%
import load.eso as eso
import parameters_old as para
import rats.single_use as su
import rats.plot_spectra as ps
import rats.plot_functions as pf
import rats.spectra_manipulation as sm
import rats.table as tab

#%% Loading parameters
# Loading parameters from NASA archive (Kuhn et al. 2016) + Gaia for R_sun
sys_para = para.system_parameters_class('KELT-10 b')
sys_para.load_nasa_parameters('KELT-10 b',force_load=True)
sys_para.print_values()

#%% Input CONAN parameters
# Stellar parameters
sys_para.star.radius = 1.204 * u.R_sun
sys_para.star.mass = 1.110 * u.M_sun
# Planet parameters
sys_para.planet.radius = 1.409 * u.R_jup
sys_para.planet.mass = 0.680 * u.M_jup
# System parameters
sys_para.system.a = 0.0508 * u.au
sys_para.system.b = 0.344 # b_tra (should b be used instead?)
sys_para.system.i = 87.82 * u.deg
sys_para.system.e = 0
sys_para.system.P = 4.1662449 * u.day
sys_para.system.omega = 90 * u.deg
sys_para.system.omega_bar = np.radians(sys_para.system.omega)
sys_para.system.vel_s = 31.9 * u.km / u.s # * u.km / u.s # From Kuhn et al. 2016
sys_para.system.vel_s = 31.9601  * u.km / u.s # From Omar's analysis
# Transit parameters
sys_para.transit.T_C = 2457175.04197 * u.day
sys_para.transit.delta = 0.01435
sys_para.transit.semiamp_s = 80 * u.m / u.s
sys_para.update_contact_points()
sys_para.transit.T14 = 0.15700069 * u.day
sys_para.define_light_curve_model([],mode = 'uniform')
sys_para.print_values()
#%% Omar analysis - RM
sys_para.RM_results(5.2,2.58)

#%%
new_test = StellarModel(121,sys_para)
new_test.__loadPHOENIXmodel__()
new_test.Transit(100)
#%%
new_test.Calculate_effect_transit()
#%%
spec_list = sp.SpectrumList()


for corrected, uncorrected, phase in zip(new_test.occulted_disk['corrected_spectrum'],new_test.occulted_disk['uncorrected_spectrum'],new_test.occulted_disk['Phase']):
    spec_list.append(
        # sp.Spectrum1D(
        #     spectral_axis = corrected.spectral_axis,
        #     flux = np.full_like(corrected.spectral_axis.value, 0) *u.dimensionless_unscaled,
        #     uncertainty = astropy.nddata.StdDevUncertainty(np.full_like(corrected.spectral_axis.value, 0))
        #     ).subtract(
        corrected.divide(uncorrected)
        # )
        # uncorrected.divide(corrected)

                )

#%%
for ind,phase in enumerate(new_test.occulted_disk['Phase']):
    spec_list[ind].meta['Phase'] = phase
    spec_list[ind].meta['vel_planet'] = orb.planet_velocity_sgl(sys_para,phase*u.dimensionless_unscaled)
    spec_list[ind].meta['vel_star'] = orb.calc_stellar_vel_sgl(sys_para,phase*u.dimensionless_unscaled)
    if (phase>-(sys_para.transit.T14/sys_para.system.P).decompose()/2) \
        and (phase<(sys_para.transit.T14/sys_para.system.P).decompose()/2):
        spec_list[ind].meta['Transit'] = True
        spec_list[ind].meta['Night'] = 1
        spec_list[ind].meta['Night_num'] = 1
        spec_list[ind].meta['normalization'] = True
        spec_list[ind].meta['RF'] = 'Star'
    else:
        spec_list[ind].meta['Transit'] = False
        spec_list[ind].meta['Night'] = 1
        spec_list[ind].meta['normalization'] = True
        spec_list[ind].meta['RF'] = 'Star'
        spec_list[ind].meta['Night_num'] = 1

shifted_spec_list = sp.SpectrumList()
for item in spec_list:
    shifted_spec_list.append(
        sm._shift_spectrum(item,
                         [-item.meta['vel_star'],item.meta['vel_planet']])
        )

master_in = sm.calculate_master_list(shifted_spec_list,key = 'Transit',value =True,sn_type='None')


phases = new_test.occulted_disk['Phase']
rv = np.zeros(new_test.occulted_disk['Phase'].shape)
for ind, phase in enumerate(new_test.occulted_disk['Phase']):
    rv[ind] = orb.planet_velocity_sgl(sys_para,phase*u.dimensionless_unscaled).value
lam_1 =  5895.924 * (1+rv/con.c.value)
lam_2 = 5889.950 * (1+rv/con.c.value)
#%%
fig, ax = plt.subplots(1)
x,y,z = ps.extract_colormap_value(shifted_spec_list)
c = ax.pcolormesh(x,y,z,vmin=0.995,vmax=1.005,cmap='coolwarm')
ax.set_xlim(5886,5900)
ax.set_xlim(5888,5892)
# ax.set_xlim(5894,5898)
ax.axhline(-(sys_para.transit.T14/sys_para.system.P).decompose()/2)
ax.axhline((sys_para.transit.T14/sys_para.system.P).decompose()/2)

offset = 0.6495/2
ax.axvline(5895.924,color='black')
ax.axvline(5889.950,color='black')

ax.axvline(5895.924-offset,color='black',ls='--')
ax.axvline(5889.950-offset,color='black',ls='--')

ax.axvline(5895.924+offset,color='black',ls='--')
ax.axvline(5889.950+offset,color='black',ls='--')

# ax.plot(lam_1,phases,color='black')
# ax.plot(lam_2,phases,color='black')
# ax.plot(lam_1-offset,phases,color='black',ls='--')
# ax.plot(lam_2-offset,phases,color='black',ls='--')
# ax.plot(lam_1+offset,phases,color='black',ls='--')
# ax.plot(lam_2+offset,phases,color='black',ls='--')


pf.add_lines_na(ax)
cbar = plt.colorbar(c)
#%%


fig,ax = plt.subplots(1)
ax.plot( master_in[0].spectral_axis, (master_in[0].flux))
