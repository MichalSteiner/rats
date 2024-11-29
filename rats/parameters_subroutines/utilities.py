#%% Importing libraries
import astropy.constants as con
import astropy.units as u
import astropy
import specutils as sp
from astropy.nddata import NDData, StdDevUncertainty, NDDataArray
import logging
from rats.utilities import default_logger_format, time_function, save_and_load
import rats.ndarray_utilities as ndutils
import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
#%% Setting up logging
logger = logging.getLogger(__name__)
logger = default_logger_format(logger)

#%%

#%% Kepler_func
def _Kepler_func(ecc_anom,mean_anom,ecc):
    """
    Find the Eccentric anomaly using the mean anomaly and eccentricity
    - M = E - e sin(E)
    """
    delta=ecc_anom-ecc*np.sin(ecc_anom)-mean_anom
    return delta
#%% calc_true_anom


#%% Calculations for specific planet
class CalculationPlanet:
    """
    A class to calculate planetary properties. This class is intended to be inherited by other classes
    that define the `mass` and `radius` attributes.
    """
    
    @property
    def mass(self) -> NDDataArray:
        """
        Mass of the planet. This property should be defined in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this property.")
    
    @property
    def radius(self) -> NDDataArray:
        """
        Radius of the planet. This property should be defined in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this property.")
    
    @property
    def equilibrium_temperature(self) -> NDDataArray:
        """
        Radius of the planet. This property should be defined in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this property.")
    
    def _calculate_gravity_acceleration(self):
        """
        Calculate the gravity acceleration at the planetary surface.
        """
        
        logger.info('Calculation of gravity acceleration:')
        # g = G * M / R / R
        self.gravity_acceleration = NDDataArray(con.G, unit= con.G.unit).multiply( #type:ignore
            self.mass.divide(
                ndutils._power_NDDataArray(self.radius, 2)
                )
            )
        self.gravity_acceleration = self.gravity_acceleration.convert_unit_to(u.m/u.s/u.s) #type:ignore
        self.gravity_acceleration.meta = {
            'parameter': 'Gravity acceleration',
            'Reference': 'calculated'
        }
        self.gravity_acceleration = ndutils._round_to_precision(self.gravity_acceleration)
        logger.info(f'    {self.gravity_acceleration}')
        return
    
    def _calculate_atmospheric_scale_height(self, molecular_mass = 2.33):
        """
        Calculate atmospheric scale height, assuming mean molecular mass

        Parameters
        ----------
        molecular_mass : float, optional
            Mean molecular mass, by default 2.33
        """
        try:
            self.gravity_acceleration   
        except:
            logger.warning('Gravity acceleration was not calculated before, recalculating.')
            self._calculate_gravity_acceleration()
        
        logger.info('Calculation of atmospheric scale height:')
        self.atmospheric_scale_height = (NDDataArray(con.N_A).multiply( #type:ignore
            self.equilibrium_temperature
            ).multiply(
                NDDataArray(con.k_B) #type:ignore
                ).divide(
                    NDDataArray(molecular_mass*u.kg/u.mol) #type:ignore
                ).divide(
                    self.gravity_acceleration
                )
                ).convert_unit_to(u.m)  # type: ignore
        self.atmospheric_scale_height = ndutils._round_to_precision(self.atmospheric_scale_height)
        
        assert self.atmospheric_scale_height.meta is not None
        
        self.atmospheric_scale_height.meta.update({
            'parameter': 'Atmospheric scale height',
            'Reference': 'calculated'
        })
        logger.info(f'    {self.atmospheric_scale_height}')
        
        return

class CalculationSystem:
    
    def _calculate_gravity_acceleration(self):
        """
        Calculate the gravity acceleration at the planetary surface
        """
        self.Planet._calculate_gravity_acceleration() #type:ignore
        return
    
    def _calculate_atmospheric_scale_height(self):
        """
        Calculate the gravity acceleration at the planetary surface
        """
        self.Planet._calculate_atmospheric_scale_height() #type:ignore
        return
    
    def stellar_model(self) -> sp.Spectrum1D:
        """
        Generate stellar model using expecto (wrapper for PHOENIX library).

        Parameters
        ----------
        vacuum : bool, optional
            Whether to return stellar spectrum with vacuum wavelength, by default False. If False, returns wavelength in air.

        Returns
        -------
        stellar_model : sp.Spectrum1D
            Stellar model extracted using expecto
        """
        return self.Star.stellar_model() #type:ignore
    
    def _calculate_planet_velocity(self,
                                   phase: float | NDDataArray) -> NDDataArray:
        """
        Calculate Keplerian velocity of the planet at given phase.

        Parameters
        ----------
        phase : float | NDDataArray
            Phase at which to calculate Keplerian velocity of the planet. If NDDataArray is given, will propagate uncertainty on the phase as well.

        Returns
        -------
        planet_velocity : NDDataArray
            Planet velocity
        """
        semimajor_axis = self.Planet.semimajor_axis.convert_unit_to(u.m) #type:ignore
        period = self.Ephemeris.period.convert_unit_to(u.s) #type:ignore 
        inclination = self.Planet.inclination.convert_unit_to(u.rad) #type:ignore
        
        # 2pi * a / P * sin(2pi*phase) * sin(i)
        planet_velocity = NDDataArray(2*np.pi,  uncertainty = StdDevUncertainty(0)).multiply(
            semimajor_axis
        ).divide(
            period
        ).multiply(
            ndutils._sin_NDDataArray(NDDataArray(2*np.pi*u.rad, uncertainty = StdDevUncertainty(0)).multiply(
                phase
            ))
        ).multiply(
            ndutils._sin_NDDataArray(inclination)
        )
        return planet_velocity
    
    def _calculate_stellar_velocity(self,
                                    phase: float | NDDataArray) -> NDDataArray:
        """
        Calculate Keplerian velocity of the star at given phase.

        Parameters
        ----------
        phase : float | NDDataArray
            Phase at which to calculate Keplerian velocity of the star. If NDDataArray is given, will propagate uncertainty on the phase as well.

        Returns
        -------
        stellar_velocity : NDDataArray
            Velocity of the star
        """
        
        if np.isnan(self.Planet.argument_of_periastron): #type:ignore
            omega = NDDataArray(90*u.deg, uncertainty = StdDevUncertainty(0))#type:ignore
            omega = omega.convert_unit_to(u.rad) 
        else:
            omega = self.Planet.argument_of_periastron.convert_unit_to(u.rad) #type:ignore
        
        keplerian_semiamplitude = self.Planet.keplerian_semiamplitude.convert_unit_to(u.m/u.s) #type:ignore
        eccentricity = self.Planet.eccentricity #type:ignore
        eccentricity = NDDataArray(eccentricity.data, uncertainty = StdDevUncertainty(eccentricity.uncertainty.array))
        
        # K * (cos(True_anomaly + omega) + e * cos(omega))
        true_anomaly = self._calc_true_anom(phase)
        stellar_velocity = keplerian_semiamplitude.multiply(
            ndutils._cos_NDDataArray(
                true_anomaly.add(omega)
                ).add(
                eccentricity.multiply(
                    ndutils._cos_NDDataArray(omega)
                )
            )
        )
        
        return stellar_velocity
    
    def _local_stellar_velocity(self,
                                phase: float | NDDataArray):
        
        if type(phase) != NDDataArray:
            phase = NDDataArray(phase,
                                uncertainty= StdDevUncertainty(0)
                                )
        
        return (self.Planet.semimajor_axis.divide(self.Star.radius).convert_unit_to(u.dimensionless_unscaled)).multiply( #type:ignore
            (
                (ndutils._sin_NDDataArray(phase.multiply(2*np.pi*u.rad))).multiply(
                    ndutils._cos_NDDataArray(self.Planet.projected_obliquity) #type:ignore
                    ).add(
                ndutils._cos_NDDataArray(phase.multiply(2*np.pi*u.rad)).multiply(
                    ndutils._cos_NDDataArray(self.Planet.inclination)).multiply(#type:ignore
                        ndutils._sin_NDDataArray(self.Planet.projected_obliquity))#type:ignore
                    )
                    ).multiply(
                        self.Star.vsini#type:ignore
                    )
            )
    
    def calculate_local_stellar_velocity(self,
                                          spectrum_list: sp.SpectrumList):
        """
        Calculate local stellar velocity for each spectrum in list.
        
        Equation used comes from Steiner+2023, section 5.2

        Parameters
        ----------
        spectrum_list : sp.SpectrumList
            Spectrum list to calculate local stellar velocity for
        """

        for spectrum in spectrum_list:
            if not(spectrum.meta['Transit_partial']):
                spectrum.meta['velocity_stellar_local'] = None
            else:
                spectrum.meta['velocity_stellar_local'] = (
                    self.Planet.semimajor_axis.divide(self.Star.radius).convert_unit_to(u.dimensionless_unscaled)).multiply(#type:ignore
                        (
                            (ndutils._sin_NDDataArray(spectrum.meta['Phase'].multiply(2*np.pi*u.rad))).multiply(
                                ndutils._cos_NDDataArray(self.Planet.projected_obliquity)#type:ignore
                                ).add(
                            ndutils._cos_NDDataArray(spectrum.meta['Phase'].multiply(2*np.pi*u.rad)).multiply(
                                ndutils._cos_NDDataArray(self.Planet.inclination)).multiply(#type:ignore
                                    ndutils._sin_NDDataArray(self.Planet.projected_obliquity))#type:ignore
                                )
                                ).multiply(
                                    self.Star.vsini#type:ignore
                                )
                        )
        return
    
    
    def _calc_true_anom(self, phase: float | NDDataArray) -> NDDataArray:
        """
        Calculates true anomaly of the orbit at certain phase.
        
        Parameters
        ----------
        phase : float | NDDataArray
            Phase of the planet. If NDDataArray is given, will propagate uncertainty.

        Returns
        -------
        true_anomaly : NDDataArray
            Calculated true anomaly.
        """

        import math
        from scipy.optimize import newton
        if np.isnan(self.Planet.argument_of_periastron):#type:ignore
            omega = NDDataArray(90*u.deg, uncertainty = StdDevUncertainty(0))#type:ignore
            omega = omega.convert_unit_to(u.rad)
        else:
            omega = self.Planet.argument_of_periastron.convert_unit_to(u.rad)#type:ignore
        eccentricity = self.Planet.eccentricity#type:ignore
        
        if type(phase) != NDDataArray:
            phase = NDDataArray(phase, uncertainty = StdDevUncertainty(0))
        
        #Circular orbit
        if math.isclose(eccentricity.data,
                        0.,
                        abs_tol=1e-4):
            
            true_anomaly = phase.multiply(NDDataArray(2.*np.pi* u.rad))
            #Eccentric anomaly
            eccentricity_anomaly = None
            #Eccentric orbit
            #  - by definition the ascending node is where the object goes toward the observer through the plane of sky
            #  - omega_bar is the angle between the ascending node and the periastron, in the orbital plane (>0 counterclockwise)
        else:
            
            #True anomaly of the planet at mid-transit:
            #    - angle counted from 0 at the perisastron, to the star/Earth LOS
            #    - >0 counterclockwise, possibly modulo 2pi
            true_anomaly_TR = NDDataArray(0.5*np.pi *u.rad).subtract(omega)
            
            # true_anom_TR = (np.pi*0.5*u.rad)-sys_para.planet.omega_bar

            #True anomaly at the time of the transit
            #    - corresponds to 'dt_transit' (in years), time from periapsis to transit center
            #    - atan(X) is in -PI/2 ; PI/2
            term_in_tan = ndutils._tan_NDDataArray(true_anomaly_TR.multiply(NDDataArray(0.5))).multiply(
                ndutils._power_NDDataArray(
                    NDDataArray(1).subtract(eccentricity).divide(
                        NDDataArray(1).add(eccentricity)
                        ),
                    (1/2)
                )
            )
            # term_in_tan = np.tan(true_anomaly_TR*0.5)*np.sqrt((1.-sys_para.planet.e)/(1.+sys_para.planet.e))
            arctan = ndutils._arctan_NDDataArray(term_in_tan)
            
            eccentricity_anomaly_TR = NDDataArray(2).multiply(arctan)
            # ecc_anom_TR=2.*np.arctan(np.tan(true_anomaly_TR*0.5)*np.sqrt((1.-sys_para.planet.e)/(1.+sys_para.planet.e)))
            
            sin_eccentricity_anomaly_TR = ndutils._sin_NDDataArray(eccentricity_anomaly_TR)
            eccentricity = NDDataArray(eccentricity.data, uncertainty = StdDevUncertainty(eccentricity.uncertainty.array)) # Removing unit
            subtraction_term =  eccentricity.multiply(
                sin_eccentricity_anomaly_TR
            ).multiply(
                NDDataArray(1, unit = u.rad)
            )
            
            mean_anomaly_TR = eccentricity_anomaly_TR.subtract(subtraction_term)
            
            # mean_anomaly_TR = ecc_anom_TR-sys_para.planet.e*np.sin(ecc_anom_TR)*u.rad
            
            if (mean_anomaly_TR.data < 0.):
                mean_anomaly_TR = mean_anomaly_TR.add(NDDataArray(2.*np.pi*u.rad))

            #Mean anomaly
            #  - time origin of t_mean at the periapsis (t_mean=0 <-> M=0 <-> E=0)
            #  - M(t_mean)=M(dt_transit)+M(t_simu)
            mean_anomaly = NDDataArray(2*np.pi*u.rad).multiply(phase).add(mean_anomaly_TR)
            # mean_anom=2.*np.pi* phase*u.rad +mean_anomaly_TR

            #Eccentric anomaly :
            #  - M = E - e sin(E)
            #    - >0 counterclockwise
            #  - angle, with origin at the ellipse center, between the major axis toward the periapsis and the
            #line crossing the circle with radius 'a_Rs' at its intersection with the perpendicular to
            #the major axis through the planet position
            eccentricity_anomaly = newton(
                _Kepler_func,
                mean_anomaly.data,
                args= (mean_anomaly.data, eccentricity) )

            eccentricity_anomaly = NDDataArray(eccentricity_anomaly * u.rad, uncertainty = StdDevUncertainty(0))
            
            #True anomaly of the planet at current time
            arctan_2_term = ndutils._power_NDDataArray(NDDataArray(1).add(eccentricity).divide(
                NDDataArray(1).subtract(eccentricity)
                ),
                                                  (1/2)
                                                  ).multiply(
                ndutils._tan_NDDataArray(
                    eccentricity_anomaly.divide(NDDataArray(2))
                                         )
                )
            
            true_anomaly = NDDataArray(2).multiply(
                ndutils._arctan_NDDataArray(arctan_2_term))
            
            # true_anomaly=2.*np.arctan(np.sqrt((1.+sys_para.planet.e)/(1.-sys_para.planet.e)) * np.tan(eccentricity_anomaly_TR/2.)) *u.rad
            
        return true_anomaly
    
    def _calculate_velocities_spectrum(self,
                                       spectrum: sp.Spectrum1D | sp.SpectrumCollection) -> None:
        """
        Calculate velocities of planet and star for given spectrum, provided spectrum has a 'Phase' meta parameter

        Parameters
        ----------
        spectrum : sp.Spectrum1D | sp.SpectrumCollection
            _description_
        """
        assert spectrum.meta is not None
        assert ('Phase' in spectrum.meta.keys()), 'Spectrum does not have "Phase" keyword.'
        assert not(spectrum.meta['Phase'] == 'undefined'), 'Spectrum "Phase" keyword is not defined.'
        
        velocity_planet = self._calculate_planet_velocity(spectrum.meta['Phase'])
        velocity_star = self._calculate_stellar_velocity(spectrum.meta['Phase'])
        
        spectrum.meta['velocity_planet'] = velocity_planet
        spectrum.meta['velocity_star'] = velocity_star
        
        return
    @time_function
    def calculate_velocities_list(self,
                                  spectra_list: sp.SpectrumList) -> None:
        """
        Calculate stellar and planetary velocities for given spectrum list, assuming 'Phase' keyword is present.

        Parameters
        ----------
        spectra_list : sp.SpectrumList
            Spectra list for which to calculate the velocities
        """
        
        for spectrum in spectra_list:
            self._calculate_velocities_spectrum(spectrum= spectrum)
    
    @time_function
    def add_velocity_system_in_list(self,
                                    spectra_list: sp.SpectrumList) -> None:
        """
        Adds systemic velocity to all spectra in list based on the self.System.systemic_velocity attribute.

        Parameters
        ----------
        spectra_list : sp.SpectrumList
            Spectra list to which to update the systemic velocity.
        """
        
        for spectrum in spectra_list:
            spectrum.meta['velocity_system'] = self.System.systemic_velocity #type:ignore


    
    def _create_Keplerian_model(self):
        phase = np.linspace(-0.5, 0.5, 1000)
        
        return
    pass

class CalculationTransitLength:
    
    def _update_contact_points(self,
                               force_recalculate = False):
        """
        Contact points based on the Winn 2014 paper.

        Parameters
        ----------
        force_recalculate : bool, optional
            Forcing to recalculate the values, by default False
        """
        # CLEANME Cleanup if possible
        # TODO move to Ephemeris class
        
        if self.Ephemeris.transit_length_partial is None or force_recalculate or np.isnan(self.Ephemeris.transit_length_partial.data): #type:ignore
            # Equation 14 in Winn (2014)
            amplitude = self.Ephemeris.period.divide((np.pi * u.rad)) # P/ pi #type:ignore
            scale = self.Star.radius.divide(self.Planet.semimajor_axis) # R_star / a #type:ignore
            
            one = NDDataArray(1, unit=u.dimensionless_unscaled)
            
            nominator_unrooted_term1 = one.add(self.Ephemeris.transit_depth) # (1+k) #type:ignore
            nominator_unrooted_term1 = ndutils._power_NDDataArray(nominator_unrooted_term1, 2) # (1+k)**2
            nominator_unrooted_term2 = (self.Planet.impact_parameter.multiply(self.Planet.impact_parameter)) # b**2 #type:ignore
            nominator_unrooted = nominator_unrooted_term1.subtract(nominator_unrooted_term2)
            nominator_unrooted = nominator_unrooted.convert_unit_to(u.dimensionless_unscaled)
            
            nominator_squareroot = ndutils._power_NDDataArray(nominator_unrooted, power= (1/2))
            denominator = ndutils._sin_NDDataArray(self.Planet.inclination) #type:ignore
            term_inside_sinus = scale.multiply(nominator_squareroot).divide(denominator).convert_unit_to(u.dimensionless_unscaled)
            
            # Eccentricity scale factor of equation 16 in Winn (2014)
            eccentricity_nominator = ndutils._power_NDDataArray(
                one.subtract(
                    ndutils._power_NDDataArray(
                        self.Planet.eccentricity, #type:ignore
                        2
                        )
                    ),
                (1/2)
                )
            eccentricity_denominator = (
                self.Planet.eccentricity.multiply( #type:ignore
                    ndutils._sin_NDDataArray(self.Planet.argument_of_periastron) #type:ignore
                    )
                ).add(one)
            
            eccentricity_correction = eccentricity_nominator.divide(eccentricity_denominator)
            
            self.Ephemeris.transit_length_partial = amplitude.multiply( #type:ignore
                ndutils._arcsin_NDDataArray(term_inside_sinus)
                ).multiply(
                    eccentricity_correction
                    ).convert_unit_to(
                        u.hour #type:ignore
                        )
                    
            self.Ephemeris.transit_length_partial.meta.update({ #type:ignore
                'reference': 'calculated',
                'formula': 'Winn (2014)',
                'parameter': 'Transit length (partial, T14)'
            })
            self.Ephemeris.transit_length_partial = ndutils._round_to_precision(self.Ephemeris.transit_length_partial) #type:ignore
            
            logger.info('(Re)-calculation of transit length')
            logger.info(f'    Partial transit (T14) length: {self.Ephemeris.transit_length_partial}') #type:ignore
            logger.info(f'recalculated based on: {self.Ephemeris.transit_length_partial.meta["formula"]}') #type:ignore
            

        if self.Ephemeris.transit_length_full is None or force_recalculate or np.isnan(self.Ephemeris.transit_length_full.data): #type:ignore
            amplitude = self.Ephemeris.period.divide((np.pi * u.rad)) # P/ pi #type:ignore
            scale = self.Star.radius.divide(self.Planet.semimajor_axis) # R_star / a #type:ignore
            one = NDDataArray(1, unit=u.dimensionless_unscaled) 
            
            nominator_unrooted_term1 = (one.subtract(self.Ephemeris.transit_depth)) # (1-k) #type:ignore
            nominator_unrooted_term1 = ndutils._power_NDDataArray(nominator_unrooted_term1, 2) # (1-k)**2 
            nominator_unrooted_term2 = (self.Planet.impact_parameter.multiply(self.Planet.impact_parameter)) # b**2 #type:ignore
            nominator_unrooted = nominator_unrooted_term1.subtract(nominator_unrooted_term2)
            nominator_unrooted = nominator_unrooted.convert_unit_to(u.dimensionless_unscaled)
            
            nominator_squareroot = ndutils._power_NDDataArray(nominator_unrooted, power= (1/2))
            denominator = ndutils._sin_NDDataArray(self.Planet.inclination) #type:ignore
            term_inside_sinus = scale.multiply(nominator_squareroot).divide(denominator).convert_unit_to(u.dimensionless_unscaled)
            
            
            # Eccentricity scale factor of equation 16 in Winn (2014)
            eccentricity_nominator = ndutils._power_NDDataArray(
                one.subtract(
                    ndutils._power_NDDataArray(
                        self.Planet.eccentricity, #type:ignore
                        2
                        )
                    ),
                (1/2)
                )
            eccentricity_denominator = (
                self.Planet.eccentricity.multiply( #type:ignore
                    ndutils._sin_NDDataArray(self.Planet.argument_of_periastron) #type:ignore
                    )
                ).add(one)
            
            eccentricity_correction = eccentricity_nominator.divide(eccentricity_denominator)
            
            self.Ephemeris.transit_length_full = amplitude.multiply( #type:ignore
                ndutils._arcsin_NDDataArray(term_inside_sinus)
                ).multiply(
                    eccentricity_correction
                    ).convert_unit_to(
                        u.hour #type:ignore
                        )
            self.Ephemeris.transit_length_full.meta.update({ #type:ignore
                'reference': 'calculated',
                'formula': 'Winn (2014)',
                'parameter': 'Transit length (full, T23)'
            })
            self.Ephemeris.transit_length_full = ndutils._round_to_precision(self.Ephemeris.transit_length_full) #type:ignore
            
            logger.info('(Re)-calculation of transit length')
            logger.info(f'    Full transit (T23) length: {self.Ephemeris.transit_length_full}') #type:ignore
            logger.info(f'recalculated based on: {self.Ephemeris.transit_length_full.meta["formula"]}') #type:ignore


    def _phase_fold(self,
                    bjd: float | u.quantity.Quantity | NDDataArray) -> NDDataArray:
        """
        Phase folds a bjd time stamp into phase.

        Parameters
        ----------
        bjd : float | u.quantity.Quantity | NDDataArray
            BJD time stamp (should be mid-exposure for spectra).  

        Returns
        -------
        phase : NDDataArray
            Phase at given BJD time. Result has uncertainty propagated through ephemeris precision.
        """
        if type(bjd) != NDDataArray:
            if type(bjd) == type(1*u.day): #type:ignore
                bjd = NDDataArray(bjd)
            else:
                bjd = NDDataArray(bjd*u.day) #type:ignore
        
        
        period = self.Ephemeris.period #type:ignore
        transit_center = self.Ephemeris.transit_center #type:ignore
        
        phase = (bjd.subtract(transit_center)).divide(period)
        phase = NDDataArray(phase.data%1, uncertainty = StdDevUncertainty(phase.uncertainty.array))
        
        if (phase.data > 0.5):#type:ignore
            phase = phase.subtract(NDDataArray(1)) 
        
        return phase
    
    def _calculate_contact_points(self):
        """
        Calculates contact points for given Ephemeris.
        """
        
        T14 = self.Ephemeris.transit_length_partial.convert_unit_to(u.h) #type:ignore
        T23 = self.Ephemeris.transit_length_full.convert_unit_to(u.h) #type:ignore
        P = self.Ephemeris.period.convert_unit_to(u.h) #type:ignore
        
        self.Ephemeris.contact_T1 = T14.divide(P).divide(NDDataArray(-1)).divide(NDDataArray(2)) #type:ignore
        self.Ephemeris.contact_T2 = T23.divide(P).divide(NDDataArray(-1)).divide(NDDataArray(2)) #type:ignore
        self.Ephemeris.contact_T3 = T23.divide(P).divide(NDDataArray(2)) #type:ignore
        self.Ephemeris.contact_T4 = T14.divide(P).divide(NDDataArray(2)) #type:ignore
        
        
        return
    
    def plot_contact_points(self,
                            ax: Axes,
                            ls='--',
                            color_full= 'darkgreen',
                            color_partial= 'goldenrod',
                            ):
        """
        Add contact points to given artist as horizontal line. 

        Parameters
        ----------
        ax : plt.Axes
            Artist to plot on
        ls : str, optional
            Linestyle of the contact point line, by default '--'
        color_full : str, optional
            Color of the contact points for T2 and T3 contact points, by default 'darkgreen'.
        color_partial : str, optional
            Color of the contact points for T1 and T4 contact points, by default 'goldenrod'
        """
        ax.axhline(self.Ephemeris.contact_T1.data, ls=ls, color= color_partial) #type:ignore
        ax.axhline(self.Ephemeris.contact_T2.data, ls=ls, color= color_full) #type:ignore
        ax.axhline(self.Ephemeris.contact_T3.data, ls=ls, color= color_full) #type:ignore
        ax.axhline(self.Ephemeris.contact_T4.data, ls=ls, color= color_partial) #type:ignore

        return
    
    def _find_transit(self,
                      phase: NDDataArray) -> tuple[bool, bool]:
        """
        Find whether a given phase is in transit or not.

        Parameters
        ----------
        phase : NDDataArray
            Phase for which to define the transit flags.

        Returns
        -------
        Transit_partial, Transit_full : [bool, bool]
            Whether the phase is partially, respectivelly fully, transiting the star.
        """
        
        t1 = self.Ephemeris.contact_T1 #type:ignore 
        t2 = self.Ephemeris.contact_T2 #type:ignore
        t3 = self.Ephemeris.contact_T3 #type:ignore
        t4 = self.Ephemeris.contact_T4 #type:ignore
        
        if (phase.data < t1) or (phase.data > t4): # Out-of-transit
            Transit_full = False
            Transit_partial = False
        elif (phase.data < t2) or (phase.data > t3): # Partial transit
            Transit_full = False
            Transit_partial = True
        else: # Full in transit
            Transit_full = True
            Transit_partial = True
        
        
        return Transit_partial, Transit_full
    
    def _spectrum_is_transiting(self,
                                spectrum: sp.Spectrum1D | sp.SpectrumCollection) -> None:
        """
        Calculate the phase and assert whether the spectrum is taken during transit or not.

        Parameters
        ----------
        spectrum : sp.Spectrum1D | sp.SpectrumCollection
            Spectrum for which to calculate the flags.

        Raises
        ------
        KeyError
            If BJD keyword is not defined, raise an error.
        """
        assert spectrum.meta is not None
                
        if 'BJD' not in spectrum.meta:
            raise KeyError('There is no BJD keyword in the spectrum!')
        
        
        phase = self._phase_fold(spectrum.meta['BJD'])
        Transit_partial, Transit_full = self._find_transit(phase)
        
        spectrum.meta.update({
            'Phase': phase,
            'Transit_partial': Transit_partial,
            'Transit_full': Transit_full,
        })
        return
    
    @time_function
    def spectra_transit_flags(self,
                              spectra_list: sp.SpectrumList) -> None:
        """
        Calculates phases and transit flags for the entire spectrum list.

        Parameters
        ----------
        spectra_list : sp.SpectrumList
            _description_
        """
        for spectrum in spectra_list:
            self._spectrum_is_transiting(spectrum= spectrum)
        return
    
    
class EquivalenciesTransmission:

    def _custom_transmission_units(self):
        """
        Defines transmission spectrum specific units for given planet. Conversion equations are available at https://www.overleaf.com/read/gkdtkqvwffzn
        """
        # TODO Check everything works for now.
        # Constants for given system
        Rp, Rs = self.Planet.radius.data * self.Planet.radius.unit, self.Star.radius.data * self.Star.radius.unit #type:ignore
        H = self.Planet.atmospheric_scale_height.convert_unit_to(u.km).data # In km #type:ignore
        rat = (Rs/Rp).decompose().value # There is a rat in my code, oh no! 
        Rp_km = Rp.to(u.km).value # Planet radius in km #type:ignore
        
        # Definition of units
        self.unit_Transmission = u.def_unit(['','T','Transmitance','Transmitted flux'])
        self.unit_Absorption = u.def_unit(['','R','Excess atmospheric absorption', 'A'])
        self.unit_PlanetRadius =  u.def_unit(['Rp','Planetary radius'])
        self.unit_TransitDepth =  u.def_unit(['','Wavelength dependent transit depth'])
        self.unit_NumberOfAtmosphericScaleHeights =  u.def_unit(['','Number of scale heights'])
        
        equivalency_transmission = u.Equivalency(
            [
                # Planetary radius (white-light radius assumed)i
                (self.unit_Transmission, self.unit_PlanetRadius,
                    lambda x: rat* (1-x+x/rat**2)**(1/2),
                    lambda x: (x**2 - rat**2)/(1-rat**2)
                ),
                # R = (1- Flam)
                (self.unit_Transmission, self.unit_Absorption,
                lambda x: 1-x,
                lambda x: 1-x
                ),
                # Transit depth
                (self.unit_Transmission, self.unit_TransitDepth,
                lambda x: 1-x+x*rat**(-2),
                lambda x: (x-1)/(-1+rat**(-2))
                ),
                # R to planetary radius
                (self.unit_Absorption, self.unit_PlanetRadius,
                lambda x: rat * np.sqrt(x + rat**(-2) - x*rat**(-2)),
                lambda x: (x**2-1 ) / (rat**2 -1)
                ),
                # R to transit depth
                (self.unit_Absorption, self.unit_TransitDepth,
                lambda x: x + rat**(-2) - rat**(-2)*x,
                lambda x: (x-rat**(-2)) / (1-rat**(-2))
                ),
                # Rp_lam to transit depth 
                (self.unit_PlanetRadius, self.unit_TransitDepth,
                lambda x: x**2 * (rat**(-2) -1) / (1-rat**(2)) ,
                lambda x: np.sqrt( (x*(1-rat**2) + rat**2 - rat**(-2)) / (rat**(-2)-1) )
                ),
                # H_num
                (self.unit_Transmission, self.unit_NumberOfAtmosphericScaleHeights,
                lambda x: (rat* (1-x+x/rat**2)**(1/2)-1)/H,
                lambda x: ((x*H+1)**2*rat**(-2)-(1))/(-1+rat**(-2))
                ),
                (self.unit_Absorption, self.unit_NumberOfAtmosphericScaleHeights,
                lambda x: 1-((rat* (1-x+x/rat**2)**(1/2)-1)/H),
                lambda x: 1-((x*H+1)**2*rat**(-2)-(1))/(-1+rat**(-2)) 
                ),
                # Use this equivalency, please
                (self.unit_PlanetRadius, self.unit_NumberOfAtmosphericScaleHeights,
                lambda x: (x-1)*Rp_km/H,
                lambda x: (x*H / Rp_km)+1
                ),
                (self.unit_TransitDepth, self.unit_NumberOfAtmosphericScaleHeights,
                lambda x: (rat*x**(1/2)-1)/H,
                lambda x: (x*H+1)**2*rat**(-2)
                ),
            ],
            "Transmission",
        )
        return
    
    
    
class StellarModel():
    def stellar_model(self,
                      vacuum= False) -> sp.Spectrum1D:
        """
        Generate stellar model using expecto (wrapper for PHOENIX library).

        Parameters
        ----------
        vacuum : bool, optional
            Whether to return stellar spectrum with vacuum wavelength, by default False. If False, returns wavelength in air.

        Returns
        -------
        stellar_model : sp.Spectrum1D
            Stellar model extracted using expecto

        """
        
        from expecto import get_spectrum
        
        stellar_spectrum = get_spectrum(
            T_eff=self.temperature.data,
            log_g=self.logg.data,
            cache=True,
            vacuum= vacuum
            )
        
        stellar_spectrum = sp.Spectrum1D(
            spectral_axis= stellar_spectrum.spectral_axis,
            flux = stellar_spectrum.flux,
            uncertainty = StdDevUncertainty(
                np.zeros_like(stellar_spectrum.flux)
                ),
            mask = np.zeros_like(stellar_spectrum.flux),
            meta= {
                'Type': 'Stellar model',
                'Stellar type': self.stellar_type.data,
                'Model': 'PHOENIX',
                'Creation': 'expecto package',
                },
            )
        return stellar_spectrum
    
class LimbDarkening:
    
    def calculate_limb_darkening_coefficients(self):
        """
        Response functions can be downloaded here: http://svo2.cab.inta-csic.es/theory/fps/index.php?id=SLOAN/SDSS.u&&mode=browse&gname=SLOAN&gname2=SDSS#filter
        
        Better way is to create a proxy-filter for given wavelength range of an instrument.
        
        TODO: Create a proxy-filter based on instrument data
        """
        import sys
        from uncertainties import ufloat
        #DOCUMENTME
        # FIXME: This is dumb!
        from ..setup_filenames import LDCU_location
        sys.path.append(LDCU_location)
        # sys.path.append('/media/chamaeleontis/Observatory_main/Code/LDCU-main')
        
        import get_lds_with_errors_v3 as glds #type:ignore

        star = {"Teff": ufloat(self.temperature.data, #type:ignore
                               self.temperature.uncertainty.array),       # K #type:ignore
                "logg": ufloat(self.logg.data, #type:ignore
                               self.logg.uncertainty.array),     # cm/s2 (= log g) #type:ignore
                "M_H": ufloat(self.metallicity.data, #type:ignore
                              self.metallicity.uncertainty.array),      # dex (= M/H) #type:ignore
                "vturb": None}                  # km/s

        # list of response functions (pass bands) to be used
        RF_list = ["espresso_uniform.dat"]

        # query the ATLAS and PHOENIX database and build up a grid of available models
        #   (to be run only once each)
        glds.update_atlas_grid()
        glds.update_phoenix_grid()

        # compute the limb-darkening coefficients
        ldc = glds.get_lds_with_errors(**star, RF=RF_list)

        # print and/or store the results
        header = glds.get_header(**star)
        summary = glds.get_summary(ldc)
        logger.print(summary) #type:ignore
        logger.print('Taking the values of "Quadratic" law, Merged, ALL') #type:ignore
        logger.print(f"    {ldc['espresso_uniform.dat']['quadratic']['Merged']['ALL'][0][0]}") #type:ignore
        logger.print(f"    {ldc['espresso_uniform.dat']['quadratic']['Merged']['ALL'][1][0]}") #type:ignore
        logger.warning("Don't forget to cite LDCU code!") 
        return ldc['espresso_uniform.dat']['quadratic']['Merged']['ALL'][0][0], ldc['espresso_uniform.dat']['quadratic']['Merged']['ALL'][1][0]

class ModellingLightCurve:
    
    @save_and_load
    def model_light_curve(self,
                          force_load=False,
                          force_skip=False,
                          pkl_name='light_curve_model.pkl'):
        """
        Creates a model light-curve with quadratic limb-darkening.
        
        The limb-darkening coefficients are calculated with LDCU.
        
        
        """
        
        
        import batman
        #DOCUMENTME
        try:
            import pickle
            with open('./saved_data/limb_darkening_coefficients.pkl', 'rb') as input_file:
                u1, u2 =  pickle.load(input_file)
        except:        
            u1, u2 = self.Star.calculate_limb_darkening_coefficients() #type:ignore
            import pickle
            with open('./saved_data/limb_darkening_coefficients.pkl', 'wb') as output_file:
                pickle.dump([u1,u2],output_file)
                
        
        params = batman.TransitParams()
        params.t0 = self.Ephemeris.transit_center.data                       #time of inferior conjunction #type:ignore
        params.per = self.Ephemeris.period.data                      #orbital period #type:ignore
        params.rp = self.Planet.rprs.data                      #planet radius (in units of stellar radii) #type:ignore
        params.a = self.Planet.a_rs_ratio.data                       #semi-major axis (in units of stellar radii) #type:ignore
        params.inc = self.Planet.inclination.data                     #orbital inclination (in degrees) #type:ignore
        params.ecc = self.Planet.eccentricity.data                      #eccentricity #type:ignore
        if np.isnan(self.Planet.argument_of_periastron.data): #type:ignore
            argument_of_periastron = 90
        else:
            argument_of_periastron = self.Planet.argument_of_periastron.data #type:ignore
        params.w = argument_of_periastron                       #longitude of periastron (in degrees) #type:ignore
        params.u = [u1, u2]                #limb darkening coefficients [u1, u2] #type:ignore
        params.limb_dark = "quadratic"       #limb darkening model #type:ignore

        self.Star.LimbDarkening_u1 = u1 #type:ignore
        self.Star.LimbDarkening_u2 = u2 #type:ignore
        
        self.lc_params = params
        return
    
    def add_transit_depth_value(self,
                                spectrum_list: sp.SpectrumList):
        #DOCUMENTME
        import batman
        
        t = np.asarray([spectrum.meta['BJD'].value for spectrum in spectrum_list])
        lc_model = batman.TransitModel(self.lc_params, t)    #initializes model
        transit_depth = lc_model.light_curve(self.lc_params)          #calculates light curve
        
        for ind, spectrum in enumerate(spectrum_list):
            spectrum.meta['light_curve_flux'] = transit_depth[ind]
            spectrum.meta['delta'] = 1-transit_depth[ind]
        
        return