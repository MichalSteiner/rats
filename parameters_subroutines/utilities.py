#%% Importing libraries
import astropy.constants as con
import astropy.units as u
import specutils as sp
from astropy.nddata import NDData, StdDevUncertainty, NDDataArray
import logging
from rats.utilities import default_logger_format
import rats.ndarray_utilities as ndutils
import numpy as np
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
    
    def _calculate_gravity_acceleration(self):
        """
        Calculate the gravity acceleration at the planetary surface
        """
        
        logger.info('Calculation of gravity acceleration:')
        # g = G * M / R / R
        self.gravity_acceleration = NDDataArray(con.G, unit= con.G.unit).multiply(
            self.mass.divide(
                ndutils._power_NDDataArray(self.radius, 2)
                )
            )
        self.gravity_acceleration = self.gravity_acceleration.convert_unit_to(u.m/u.s/u.s)
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
        self.atmospheric_scale_height = (NDDataArray(con.N_A).multiply(
            self.equilibrium_temperature
            ).multiply(
                NDDataArray(con.k_B)
                ).divide(
                    NDDataArray(molecular_mass*u.kg/u.mol)
                ).divide(
                    self.gravity_acceleration
                )
                ).convert_unit_to(u.m)
        self.atmospheric_scale_height = ndutils._round_to_precision(self.atmospheric_scale_height)
        self.atmospheric_scale_height.meta = {
            'parameter': 'Atmospheric scale height',
            'Reference': 'calculated'
        }
        logger.info(f'    {self.atmospheric_scale_height}')
        
        return

class CalculationSystem:
    
    def _calculate_gravity_acceleration(self):
        """
        Calculate the gravity acceleration at the planetary surface
        """
        self.Planet._calculate_gravity_acceleration
        return
    
    def _calculate_atmospheric_scale_height(self):
        """
        Calculate the gravity acceleration at the planetary surface
        """
        self.Planet._calculate_atmospheric_scale_height
        return
    
    def _calculate_TSM(self):
        #TODO
        logger.info('Calculation of gravity acceleration:')
        
        logger.info('')
        
        return
    
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
        semimajor_axis = self.Planet.semimajor_axis.convert_unit_to(u.m)
        period = self.Ephemeris.period.convert_unit_to(u.s)
        inclination = self.Planet.inclination.convert_unit_to(u.rad)
        
        # 2pi * a / P * sin(2pi*phase) * sin(i)
        planet_velocity = NDDataArray(2*np.pi).multiply(
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
        
        omega = self.Planet.argument_of_periastron.convert_unit_to(u.rad)
        keplerian_semiamplitude = self.Planet.keplerian_semiamplitude.convert_unit_to(u.m/u.s)
        eccentricity = self.Planet.eccentricity
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
        
        # vel_star = sys_para.planet.semiamp_s.to(u.m/u.s) * \
        # (np.cos(true_anomaly+omega) + sys_para.planet.e * np.cos(omega))
        return stellar_velocity
    
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
        #CLEANME
        import math
        from scipy.optimize import newton
        
        eccentricity = self.Planet.eccentricity
        omega = self.Planet.argument_of_periastron.convert_unit_to(u.rad)
        
        if type(phase) == float:
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
            mean_anom = NDDataArray(2*np.pi*u.rad).multiply(phase).add(mean_anomaly_TR)
            # mean_anom=2.*np.pi* phase*u.rad +mean_anomaly_TR

            #Eccentric anomaly :
            #  - M = E - e sin(E)
            #    - >0 counterclockwise
            #  - angle, with origin at the ellipse center, between the major axis toward the periapsis and the
            #line crossing the circle with radius 'a_Rs' at its intersection with the perpendicular to
            #the major axis through the planet position
            eccentricity_anomaly = newton(
                _Kepler_func,
                mean_anom.data,
                args= (mean_anom.data, eccentricity) )

            eccentricity_anomaly = NDDataArray(eccentricity_anomaly * u.rad, uncertainty = StdDevUncertainty(0))
            
            #True anomaly of the planet at current time
            arctan_2_term = ndutils._power_NDDataArray(NDDataArray(1).add(eccentricity).divide(
                NDDataArray(1).subtract(eccentricity)
                ),
                                                  (1/2)
                                                  )
            
            true_anomaly = NDDataArray(2).multiply(
                ndutils._arctan_NDDataArray(arctan_2_term).multiply(
                ndutils._tan_NDDataArray(
                    eccentricity_anomaly.divide(NDDataArray(2))
                                         )
                ))
            # FIXME make sure this implementation is correct.
            logger.critical('Calculation of Keplerian velocity using eccentric orbit. Please double-check the output is correct.')
            
            
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
        assert ('Phase' in spectrum.meta.keys()), 'Spectrum does not have "Phase" keyword.'
        assert not(spectrum.meta['Phase'] == 'undefined'), 'Spectrum "Phase" keyword is not defined.'
        
        velocity_planet = self._calculate_planet_velocity(spectrum.meta['Phase'])
        velocity_star = self._calculate_stellar_velocity(spectrum.meta['Phase'])
        
        spectrum.meta['velocity_planet'] = velocity_planet
        spectrum.meta['velocity_star'] = velocity_star
        
        return
    
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
        
        if self.Ephemeris.transit_length_partial is None or force_recalculate or np.isnan(self.Ephemeris.transit_length_partial.data):
            # Equation 14 in Winn (2014)
            amplitude = self.Ephemeris.period.divide((np.pi * u.rad)) # P/ pi
            scale = self.Star.radius.divide(self.Planet.semimajor_axis) # R_star / a
            
            one = NDDataArray(1, unit=u.dimensionless_unscaled)
            
            nominator_unrooted_term1 = one.add(self.Ephemeris.transit_depth) # (1+k)
            nominator_unrooted_term1 = ndutils._power_NDDataArray(nominator_unrooted_term1, 2) # (1+k)**2
            nominator_unrooted_term2 = (self.Planet.impact_parameter.multiply(self.Planet.impact_parameter)) # b**2
            nominator_unrooted = nominator_unrooted_term1.subtract(nominator_unrooted_term2)
            nominator_unrooted = nominator_unrooted.convert_unit_to(u.dimensionless_unscaled)
            
            nominator_squareroot = ndutils._power_NDDataArray(nominator_unrooted, power= (1/2))
            denominator = ndutils._sin_NDDataArray(self.Planet.inclination)
            term_inside_sinus = scale.multiply(nominator_squareroot).divide(denominator)
            
            # Eccentricity scale factor of equation 16 in Winn (2014)
            eccentricity_nominator = ndutils._power_NDDataArray(
                one.subtract(
                    ndutils._power_NDDataArray(
                        self.Planet.eccentricity,
                        2
                        )
                    ),
                (1/2)
                )
            eccentricity_denominator = (
                self.Planet.eccentricity.multiply(
                    ndutils._sin_NDDataArray(self.Planet.argument_of_periastron)
                    )
                ).add(one)
            
            eccentricity_correction = eccentricity_nominator.divide(eccentricity_denominator)
            
            self.Ephemeris.transit_length_partial = amplitude.multiply(
                ndutils._arcsin_NDDataArray(term_inside_sinus)
                ).multiply(
                    eccentricity_correction
                    ).convert_unit_to(
                        u.hour
                        )
            self.Ephemeris.transit_length_partial.meta.update({
                'reference': 'calculated',
                'formula': 'Winn (2014)',
                'parameter': 'Transit length (partial, T14)'
            })
            self.Ephemeris.transit_length_partial = ndutils._round_to_precision(self.Ephemeris.transit_length_partial)
            
            logger.info('(Re)-calculation of transit length')
            logger.info(f'    Partial transit (T14) length: {self.Ephemeris.transit_length_partial}')
            logger.info(f'recalculated based on: {self.Ephemeris.transit_length_partial.meta["formula"]}')
            

        if self.Ephemeris.transit_length_full is None or force_recalculate or np.isnan(self.Ephemeris.transit_length_full.data):
            amplitude = self.Ephemeris.period.divide((np.pi * u.rad)) # P/ pi
            scale = self.Star.radius.divide(self.Planet.semimajor_axis) # R_star / a
            one = NDDataArray(1, unit=u.dimensionless_unscaled)
            
            nominator_unrooted_term1 = (one.subtract(self.Ephemeris.transit_depth)) # (1-k)
            nominator_unrooted_term1 = ndutils._power_NDDataArray(nominator_unrooted_term1, 2) # (1-k)**2
            nominator_unrooted_term2 = (self.Planet.impact_parameter.multiply(self.Planet.impact_parameter)) # b**2
            nominator_unrooted = nominator_unrooted_term1.subtract(nominator_unrooted_term2)
            nominator_unrooted = nominator_unrooted.convert_unit_to(u.dimensionless_unscaled)
            
            nominator_squareroot = ndutils._power_NDDataArray(nominator_unrooted, power= (1/2))
            denominator = ndutils._sin_NDDataArray(self.Planet.inclination)
            term_inside_sinus = scale.multiply(nominator_squareroot).divide(denominator)
            
            
            # Eccentricity scale factor of equation 16 in Winn (2014)
            eccentricity_nominator = ndutils._power_NDDataArray(
                one.subtract(
                    ndutils._power_NDDataArray(
                        self.Planet.eccentricity,
                        2
                        )
                    ),
                (1/2)
                )
            eccentricity_denominator = (
                self.Planet.eccentricity.multiply(
                    ndutils._sin_NDDataArray(self.Planet.argument_of_periastron)
                    )
                ).add(one)
            
            eccentricity_correction = eccentricity_nominator.divide(eccentricity_denominator)
            
            self.Ephemeris.transit_length_full = amplitude.multiply(
                ndutils._arcsin_NDDataArray(term_inside_sinus)
                ).multiply(
                    eccentricity_correction
                    ).convert_unit_to(
                        u.hour
                        )
            self.Ephemeris.transit_length_full.meta.update({
                'reference': 'calculated',
                'formula': 'Winn (2014)',
                'parameter': 'Transit length (full, T23)'
            })
            self.Ephemeris.transit_length_full = ndutils._round_to_precision(self.Ephemeris.transit_length_full)
            
            logger.info('(Re)-calculation of transit length')
            logger.info(f'    Full transit (T23) length: {self.Ephemeris.transit_length_full}')
            logger.info(f'recalculated based on: {self.Ephemeris.transit_length_full.meta["formula"]}')


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
            if type(bjd) == type(1*u.day):
                bjd = NDDataArray(bjd)
            else:
                bjd = NDDataArray(bjd*u.day)
        
        
        period = self.Ephemeris.period
        transit_center = self.Ephemeris.transit_center
        
        phase = (bjd.subtract(transit_center)).divide(period)
        phase = NDDataArray(phase.data%1, uncertainty = StdDevUncertainty(phase.uncertainty.array))
        
        if (phase.data > 0.5):
            phase = phase.subtract(NDDataArray(1))
        
        return phase
    
    def _calculate_contact_points(self):
        """
        Calculates contact points for given Ephemeris.
        """
        
        T14 = self.Ephemeris.transit_length_partial.convert_unit_to(u.h)
        T23 = self.Ephemeris.transit_length_full.convert_unit_to(u.h)
        P = self.Ephemeris.period.convert_unit_to(u.h)
        
        self.Ephemeris.contact_T1 = T14.divide(P).divide(NDDataArray(-1))
        self.Ephemeris.contact_T2 = T23.divide(P).divide(NDDataArray(-1))
        self.Ephemeris.contact_T3 = T23.divide(P)
        self.Ephemeris.contact_T4 = T14.divide(P)
        
        return
    
    def _find_transit(self,
                      phase: NDDataArray) -> [bool, bool]:
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
        
        t1 = self.Ephemeris.contact_T1
        t2 = self.Ephemeris.contact_T2
        t3 = self.Ephemeris.contact_T3
        t4 = self.Ephemeris.contact_T4
        
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
        if 'BJD' not in spectrum.meta:
            raise KeyError('There is no BJD keyword in the spectrum!')
        
        # TODO Check on precision of ephemeris.
        
        phase = self._phase_fold(spectrum.meta['BJD'])
        Transit_partial, Transit_full = self._find_transit(phase)
        
        spectrum.meta.update({
            'Phase': phase,
            'Transit_partial': Transit_partial,
            'Transit_full': Transit_full,
        })
        return
    
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
        Rp, Rs = self.Planet.radius.data * self.Planet.radius.unit, self.Star.radius.data * self.Star.radius.unit
        H = self.Planet.atmospheric_scale_height.convert_unit_to(u.km).data # In km
        rat = (Rs/Rp).decompose().value # There is a rat in my code, oh no!
        Rp_km = Rp.to(u.km).value # Planet radius in km
        
        # Definition of units
        self.unit_Transmission = u.def_unit(['','T','Transmitance','Transmitted flux'])
        self.unit_Absorption = u.def_unit(['','R','Excess atmospheric absorption', 'A'])
        self.unit_PlanetRadius =  u.def_unit(['Rp','Planetary radius'])
        self.unit_TransitDepth =  u.def_unit(['','Wavelength dependent transit depth'])
        self.unit_NumberOfAtmosphericScaleHeights =  u.def_unit(['','Number of scale heights'])
        
        equivalency_transmission = u.Equivalency(
            [
                # Planetary radius (white-light radius assumed)
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