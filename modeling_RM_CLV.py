# TODO:
"""
Main idea:
    Create a hexagonal grid to simulate the planet transit.
    Create fine sampled grid on the planetary track
    Each cell should be scaled by limb-darkening and shifted by local stellar velocity.
        How to do this?
    
    First order approximation creates a grid such that the planet is in center of a cell, and the next spectrum is in the center of the next cell.
        As such, the RM+CLV effect is full grid - the one occulted cell
    Second order approximation creates a fine grid where the position of the planet changes over the exposure. 
        The RM+CLV effect is than a combination of cell subtractions, normalized to planet area.
    Third order approximation simulates the entire exposure, with movement of the planet.
        The RM+CLV effect is than sampled more finely in time than the exposure.
"""
#%%
# TODO: Create a grid of linelists
#    - For each stellar type
#    - For a grid of metallicities
#    - For a grid of logg
#    - Macroturbulences




#%% Importing libraries
from hexalattice.hexalattice import *
import matplotlib.pyplot as plt
from astropy.nddata import NDData, StdDevUncertainty, NDDataArray
import numpy as np
import rats.spectra_manipulation as sm
import matplotlib as mpl
import astropy.units as u
import rats.parameters as para
from dataclasses import dataclass
from rats.utilities import default_logger_format
from pysme.sme import SME_Structure as SME_Struct
import specutils as sp

from pysme.linelist.vald import ValdFile
from pysme.abund import Abund

from pysme.solve import Synthesizer
import logging

# Setup logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)


# This solves the loading of subpackages in VSCode, however is quite dumb. 
# CLEANME: Make sure this is not needed 
import os,sys
sys.path.append('./')

# #%%
# radius = 10
# impact_parameter = 0.5
# length_at_transit_chord = np.sqrt(radius**2 - (radius*impact_parameter)**2) 
# N_exposures = 33
# number_of_cells = round(N_exposures * radius / length_at_transit_chord)+1
# hex_diameter = radius/number_of_cells*2
# fig, ax = plt.subplots(1)

# stellar_circle = mpl.patches.Circle(
#     (0,0),
#     radius = radius,
#     color = 'yellow'
#     )
# ax.add_patch(stellar_circle)

# hex_centers, ax = create_hex_grid(
#     nx= number_of_cells,
#     ny= number_of_cells*1.2,
#     min_diam= hex_diameter,
#     do_plot= False,
#     h_ax= ax
#     )
# # TODO




# ax.axhline(radius * impact_parameter, color='darkred')
# ax.axhline(-radius * impact_parameter, color= 'darkgreen')

# stellar_cells = []
# for item in hex_centers:
#     hex_x, hex_y = item
#     hex_radius = np.sqrt(hex_x**2 + hex_y**2)
    
#     # TODO fix the condition to include all the grids touching the 
#     if ((hex_x)**2 + (hex_y)**2) - (hex_diameter)**2 >  radius**2:
#         print(f'Drop hex_x: {hex_x} and hex_y: {hex_y} because radius: {hex_radius} is larger then selected radius of the star {radius}')
        
#     else:
#         stellar_cells.append(
#             [hex_x,
#             hex_y]
#             )
#         plot_single_lattice(coord_x= np.asarray([hex_x]),
#                             coord_y= np.asarray([hex_y]),
#                             face_color= None,
#                             edge_color= None,
#                             min_diam= hex_diameter,
#                             plotting_gap= 0,
#                             rotate_deg=0.,
#                             h_ax=ax,
#                             )

# ax.set_xlim(-radius*1.1,radius*1.1)
# ax.set_ylim(-radius*1.1,radius*1.1)

# fig.show()
#%% 

# from pysme.sme import SME_Structure as SME_Struct

# sme = SME_Struct()

# from pysme.abund import Abund
# sme.teff, sme.logg, sme.monh = 5948, 4.319, 0.090
# sme.vsini = 0 # Vsini set to 0 since SME calculates spectra in annuli, not allowing specific cell spectrum
# sme.deltav = 0.5

# # Needs to handle the center cell
# sme.mu = 0.05
# sme.abund = Abund.solar()

# # SME comes with a few model atmospheres see Atmosphere section
# sme.atmo.source = "marcs2012p_t1.0.sav"
# sme.atmo.method = "grid"
# sme.atmo.geom = "PP"
# sme.wran = [[3000, 8000]]
# sme.specific_intensities_only = False

# from pysme.solve import Synthesizer
# logger.info('About to load synthesizer')
# synth = Synthesizer()
# logger.info('About to synthesize spectrum')


# sme = synth.synthesize_spectrum(sme=sme)

# logger.info('About to plot final plot')
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1)
# ax.plot(sme.wave[0], sme.synth[0])

# ax.legend()
# ax.set_xlim(5889,5891)
# fig.show()
# logger.info('Done')


#%%
@dataclass
class Hex():
    """
    Contains information of Hexes inside the stellar grid.
    
    Attributes
    ----------
    x_coordinate : float
        x-coordinate on the grid. Hex-center is saved.
    y_coordinate : float
        y-coordinate on the grid. Hex-center is saved.
    mu : float
        mu = cos(phi), a measure of how far from the center of the stellar disc we are (0 = center, 1 = limb) at given center of the Hex.
    local_stellar_velocity : float
        Local stellar velocity at given center of the Hex.
    area_of_cell_in_star : float
        What fraction of area is inside the star. Scales the disk-integrated spectrum.
    local_transit_depth : float
        The modeled transit depth we expect from the star at given phase/exposure. Is only used when asking for light-curve weighted model.
    
    """
    
    x_coordinate: float
    y_coordinate: float
    
    def _calculate_stellar_velocity(self,
                                    SystemParameters: para.SystemParametersComposite,
                                    ):
        # Calculate the T23 if not present
        # FIXME:
        # SystemParameters._update_contact_points(force_recalculate=True)
        obliquity = SystemParameters.Planet.obliquity.data
        vsini = SystemParameters.Star.vsini.data
        
        self.local_stellar_velocity = (self.x_coordinate/SystemParameters.Star.radius.data * np.cos(obliquity * np.pi/180) - 
                                       self.y_coordinate/SystemParameters.Star.radius.data * np.sin(obliquity * np.pi/180)) * vsini
        
        logger.debug(f'Stellar velocity: {self.local_stellar_velocity} km/s at x: {self.x_coordinate} and y: {self.y_coordinate}')
        return
    
    def _calculate_mu(self,
                      stellar_radius: float):
        """
        Calculates the mu for given cell.

        Parameters
        ----------
        stellar_radius : float
            Stellar radius in the grid.
        """
        
        hex_radius = np.sqrt(self.x_coordinate**2 +
                             self.y_coordinate**2)
        self.hex_radius = hex_radius
        # Check if correct and shouldn't be opposite
        self.mu = hex_radius / stellar_radius
        if self.mu > 1:
            self.mu = 1
    
    def _synthetize_spectrum(self,
                             SystemParameters: para.SystemParametersComposite,
                             delta_v = 0.5,
                             wavelength_range = [[5880, 5890]]):
        

        sme = SME_Struct()

        sme.teff, sme.logg, sme.monh = SystemParameters.Star.temperature.data, SystemParameters.Star.logg.data, SystemParameters.Star.metallicity.data
        sme.vsini = 0 # Vsini set to 0 since SME calculates spectra in annuli, not allowing specific cell spectrum
        sme.deltav = delta_v

        sme.mu = self.mu
        if sme.mu == 0:
            sme.mu += 0.01
        sme.abund = Abund.solar()
        #TODO - Fix for different stellar types
        sme = self._loadlinelist(sme)
        
        sme.atmo.source = "marcs2012p_t1.0.sav"
        sme.atmo.method = "grid"
        sme.atmo.geom = "PP"
        sme.wran = [[5880, 5900]]
        sme.specific_intensities_only = False

        logger.info('About to load synthesizer')
        synth = Synthesizer()
        logger.info('About to synthesize spectrum')
        sme = synth.synthesize_spectrum(sme=sme)
        
        self.local_spectrum = sp.Spectrum1D(
            spectral_axis= sme.wave[0] * u.AA,
            flux= sme.synth[0] * u.dimensionless_unscaled, 
            uncertainty= np.zeros_like(sme.wave[0])
            )
        self.local_spectrum = sm._shift_spectrum(self.local_spectrum,
                                                velocities= [self.local_stellar_velocity * u.km/u.s])
        return

    def _loadlinelist(self,
                      sme):
        
        
        vald = ValdFile("/media/chamaeleontis/Observatory_main/Code/rats/LINELIST_VALD/MichalSteiner.015070")
        sme.linelist = vald
        logger.info('Loaded Line list')
        return sme



#%%
class ModelGrid:
    """
    Model grid for the star. Can simulate a transit event to estimate the RM+CLV effect.
    """
    pass

    def plot_stellar_grid(self):
        """
        Plots a stellar grid, including the transit path.
        """
        
        fig, ax = plt.subplots(1)

        stellar_circle = mpl.patches.Circle(
            (0,0),
            radius = self.radius,
            color = 'black',
            fill = False
            )
        ax.add_patch(stellar_circle)
        
        ax.axhline(self.radius * self.impact_parameter, color='darkred')
        ax.axhline(-self.radius * self.impact_parameter, color= 'darkgreen')

        cmap = mpl.cm.get_cmap('seismic')
        
        for single_hex in self.stellar_cells:
            plot_single_lattice(coord_x= np.asarray([single_hex.x_coordinate]),
                                coord_y= np.asarray([single_hex.y_coordinate]),
                                face_color= cmap(single_hex.local_stellar_velocity/ self.SystemParameters.Star.vsini.data +0.5),
                                edge_color= None,
                                min_diam= self.hex_diameter,
                                plotting_gap= 0,
                                rotate_deg=0.,
                                h_ax=ax,
                                )

        ax.set_xlim(-self.radius*1.1,self.radius*1.1)
        ax.set_ylim(-self.radius*1.1,self.radius*1.1)
        fig.show()
        return
    
    def _create_initial_grid(self,
                             SystemParameters: para.SystemParametersComposite,
                             N_exposures: int = 15,
                             ) -> None:
        """
        Create an initial hexagonal grid. This is later cut by self._filter_stellar_cells() to include only the cells inside star.

        Parameters
        ----------
        SystemParameters : para.SystemParametersComposite
            System parameters for which to create the stellar grid. Scales the size of the grid to be in units of solar radii.
        N_exposures : int, optional
            Number of exposures for given night, by default 15. This scales the grid in a way that the transit cord is covered by exactly the number of exposures of cells. This is done for first approximation, where the RM+CLV effect is simulated as one hex-cell missing in the stellar grid.
        """
        self.SystemParameters = SystemParameters
        
        self.radius = SystemParameters.Star.radius.data
        self.impact_parameter = SystemParameters.Planet.impact_parameter.data
        self.half_length_at_transit_chord = np.sqrt(self.radius**2 - (self.radius*self.impact_parameter)**2) 
        
        self.number_of_cells = round(N_exposures * self.radius / self.half_length_at_transit_chord)+1
        self.hex_diameter = self.radius/self.number_of_cells*2
        
        self.hex_centers, _ = create_hex_grid(
            nx= self.number_of_cells,
            ny= self.number_of_cells*2,# TODO: Fix the scaling to be actually correct
            min_diam= self.hex_diameter,
            do_plot= False,
            h_ax= None
            )
        self._filter_stellar_cells()
        
        return
    
    def _filter_stellar_cells(self) -> None:
        """
        Filters the hexagonal grid to only include cells inside the star.
        """

        self.stellar_cells = []
        for item in self.hex_centers:
            hex_x, hex_y = item
            hex_radius = np.sqrt(hex_x**2 + hex_y**2)
            
            # TODO fix the condition to include all the grids touching the stellar radius
            if ((hex_x)**2 + (hex_y)**2) - (self.hex_diameter)**2 >  self.radius**2:
                logger.debug(f'Drop hex_x: {hex_x} and hex_y: {hex_y} because radius: {hex_radius} is larger then selected radius of the star {self.radius}')
                
            else:
                new_Hex = Hex(
                    x_coordinate= hex_x,
                    y_coordinate= hex_y,
                    )
                new_Hex._calculate_mu(self.radius)
                new_Hex._calculate_stellar_velocity(self.SystemParameters)
                # new_Hex._synthetize_spectrum(SystemParameters= self.SystemParameters)
                self.stellar_cells.append(new_Hex)

    def _calculate_light_curve(self):
        
        
        
        return
    
    def scale_by_cross_section():
        # TODO - On the edge, cut the grid to circular one.
        return
    
    def _disk_integrate_stellar_spectrum(self):
        
        disk_integrated_spectrum = sp.Spectrum1D(
            spectral_axis= self.stellar_cells[0].local_spectrum.spectral_axis,
            flux= np.zeros_like(self.stellar_cells[0].local_spectrum.spectral_axis),
            uncertainty = StdDevUncertainty(np.zeros_like(self.stellar_cells[0].local_spectrum.spectral_axis))
            
           )
        for hex in self.stellar_cells:
            disk_integrated_spectrum += hex.local_spectrum
        self.disk_integrated_spectrum = disk_integrated_spectrum
        return
    
    
    
    
    def simulate_transit():
        
        
        return

if __name__ == '__main__':
    os.chdir('/media/chamaeleontis/Observatory_main/Analysis_dataset/rats_test')
    logger.info('Starting test of rats.modeling_CCF:')
    logger.info('Loading parameters of KELT-10 b system')
    
    # Testing for KELT-10 b system
    KELT10b = para.SystemParametersComposite(
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
    
    logger.info('Creating Model grid on which to simulate the RM+CLV effect')
    RM_CLV_Model = ModelGrid()
    RM_CLV_Model._create_initial_grid(
        SystemParameters= KELT10b,
        N_exposures= 45
        )
    logger.info('Plot stellar grid')
    RM_CLV_Model.plot_stellar_grid()
    
    logger.info('Test of modeling RM+CLV effects succesful')
    
    